import deepspeed
import deepspeed.comm as dist
import intel_extension_for_pytorch as ipex
try:
    ipex._C.disable_jit_linear_repack()
except Exception:
    pass
import json
import os
import torch

from deepspeed import get_accelerator
from huggingface_hub import snapshot_download
from pathlib import Path
from torch.nn.modules import Module
from transformers.models.bloom.modeling_bloom import BloomBlock as BloomBlock
from transformers.utils import is_offline_mode
from transformers import (
    # pipeline,
    AutoConfig,
    AutoModelForCausalLM,
    # AutoModel,
    T5ForConditionalGeneration,
    AutoTokenizer,
)

from .inference_transformer_base import DSTransformerModelBase
from ..inference_utils import ActivationType, DtypeEnum
from ..ragged import RaggedBatchWrapper
from ..modules.configs import *


# supported models now
MODEL_CLASSES = {
    "gpt-j": (AutoModelForCausalLM, AutoTokenizer),
    "gpt-neox": (AutoModelForCausalLM, AutoTokenizer),
    "opt": (AutoModelForCausalLM, AutoTokenizer),
    "bloom": (AutoModelForCausalLM, AutoTokenizer),
    "llama": (AutoModelForCausalLM, AutoTokenizer),
    "t5": (T5ForConditionalGeneration, AutoTokenizer),
    "falcon": (AutoModelForCausalLM, AutoTokenizer),
    "auto": (AutoModelForCausalLM, AutoTokenizer),
    # "chatglm": (AutoModel, AutoTokenizer),
}

# the Deepspeed team made these so it's super fast to load (~1 minute), rather than wait 10-20min loading time.
TP_PRESHARDED_MODELS = [
    "microsoft/bloom-deepspeed-inference-int8",
    "microsoft/bloom-deepspeed-inference-fp16",
]
NO_META_SUPPORT_MODELS = ["falcon"]


def get_int_from_env(env_keys, default):
    """Returns the first positive env value found in the `env_keys` list or the default."""
    for e in env_keys:
        val = int(os.environ.get(e, -1))
        if val >= 0:
            return val
    return default


def get_repo_root(model_name_or_path, local_rank=0):
    if os.path.isdir(model_name_or_path):
        return model_name_or_path
    # checks if online or not
    if is_offline_mode():
        print("Offline mode: forcing local_files_only=True")
    # download only on first process
    if local_rank == 0:
        snapshot_download(
            model_name_or_path,
            local_files_only=is_offline_mode(),
            cache_dir=os.getenv("TRANSFORMERS_CACHE", None),
            ignore_patterns=["*.safetensors", "*.msgpack", "*.h5"],
            resume_download=True,
        )

    dist.barrier()

    return snapshot_download(
        model_name_or_path,
        local_files_only=is_offline_mode(),
        cache_dir=os.getenv("TRANSFORMERS_CACHE", None),
        ignore_patterns=["*.safetensors", "*.msgpack", "*.h5"],
        resume_download=True,
    )


def get_checkpoint_files(model_name_or_path, local_rank=0):
    cached_repo_dir = get_repo_root(model_name_or_path, local_rank)
    # extensions: .bin | .pt
    # creates a list of paths from all downloaded files in cache dir
    file_list = [
        str(entry)
        for entry in Path(cached_repo_dir).rglob("*.[bp][it][n]")
        if entry.is_file()
    ]
    return file_list


def write_checkpoints_json(model_name_or_path, checkpoints_json="checkpoints.json"):
    checkpoint_files = get_checkpoint_files(model_name_or_path)
    data = {"type": model_name_or_path, "checkpoints": checkpoint_files, "version": 1.0}
    json.dump(data, open(checkpoints_json, "w"))


class XPUModel(DSTransformerModelBase, Module):
    def __init__(self, config, policy, base_mp_group, **kwargs):
        Module.__init__(self)
        self._config = config
        self._policy = policy
        self._base_mp_group = base_mp_group

        self.model_name_or_path = kwargs.get("model_name_or_path", "/datadisk/share/llama2-7b")
        self.local_rank = get_int_from_env(
            ["LOCAL_RANK", "MPI_LOCALRANKID", "PALS_LOCAL_RANKID"], "0"
        )
        # self.world_size = get_int_from_env(
        #     ["WORLD_SIZE", "PMI_SIZE", "PALS_LOCAL_SIZE"], "1"
        # )
        self.world_size = self._config.tensor_parallel.tp_size
        self.port = get_int_from_env(["MASTER_PORT"], 29500)
        print(f"*** local_rank={self.local_rank} world_size={self.world_size}")
        self.kernel_inject = False
        self.jit = False
        self.dtype = kwargs.get("dtype", "float16")
        self._parse_dtype(self.dtype)
        self.model: Module = None

    def _parse_dtype(self, dtype="float16"):
        if dtype == "float16":
            self.load_dtype = torch.half
            self.infer_dtype = torch.half
        elif dtype == "bfloat16":
            self.load_dtype = torch.bfloat16
            self.infer_dtype = torch.bfloat16
        elif dtype == "int8":
            self.load_dtype = torch.half
            self.infer_dtype = torch.int8
        elif dtype == "float32":
            self.load_dtype = torch.float32
            self.infer_dtype = torch.float32

    """
    Inference model implementation for ragged batching for Llama-2 models.
    """

    def build_model(self):
        # load model
        model_type = next(
            (x for x in MODEL_CLASSES.keys() if x in self.model_name_or_path.lower()),
            "auto",
        )
        model_class = MODEL_CLASSES[model_type]
        config = AutoConfig.from_pretrained(
            self.model_name_or_path, torchscript=self.jit, trust_remote_code=True
        )
        kwargs = dict(
            replace_with_kernel_inject=self.kernel_inject,
            injection_policy=None
        )
        is_meta_support = (not model_type in NO_META_SUPPORT_MODELS) and (
            kwargs["injection_policy"] or self.kernel_inject or self.world_size > 1
        )  # support meta and w/ (1.specified TP or 2.Kernel injection or 3.Auto-TP)
        with deepspeed.OnDevice(
            dtype=self.load_dtype, device="meta", enabled=is_meta_support
        ):
            if model_class[0] == AutoModelForCausalLM and is_meta_support:  # -> meta
                model = model_class[0].from_config(
                    config, torch_dtype=self.load_dtype, trust_remote_code=True
                )
            else:  # -> host
                model = model_class[0].from_pretrained(
                    self.model_name_or_path,
                    config=config,
                    low_cpu_mem_usage=True,
                    torch_dtype=self.load_dtype,
                    trust_remote_code=True,
                )
        model = model.eval().to(memory_format=torch.channels_last)
        # write ckpt.json
        repo_root = get_repo_root(self.model_name_or_path, self.local_rank)
        if (self.model_name_or_path in TP_PRESHARDED_MODELS) and self.kernel_inject:
            checkpoints_json = os.path.join(repo_root, "ds_inference_config.json")
        else:
            checkpoints_json = "checkpoints.json"
            if self.local_rank == 0:
                write_checkpoints_json(self.model_name_or_path, checkpoints_json)
            # dist.barrier()
        # init DS::EngineV1
        model = deepspeed.init_inference(
            model,
            mp_size=self.world_size,
            base_dir=repo_root,
            dtype=self.infer_dtype,
            checkpoint=checkpoints_json if is_meta_support else None,
            **kwargs,
        )
        # optimize model
        model = ipex.optimize_transformers(
            model.eval(),
            dtype=self.infer_dtype,
            device=get_accelerator().device_name(),
            inplace=(
                "low_precision_checkpoint"
                in ipex.optimize_transformers.__code__.co_varnames
            ),
        )
        # convert DS::EngineV1 -> nn.Module
        if isinstance(model, deepspeed.InferenceEngine):
            self.model = model.module

    """
    Properties ineherited from `DSInferenceModelBase`
    """

    @property
    def max_sequence_length(self) -> int:
        return self._config.max_seq_length

    """
    Properties ineherited from `DSTransformerModelBase`
    """

    @property
    def num_layers(self) -> int:
        return self._config.num_hidden_layers

    @property
    def model_dim(self) -> int:
        return self._config.hidden_size

    @property
    def vocab_size(self) -> int:
        return self._config.vocab_size

    @property
    def head_size(self) -> int:
        return self.model_dim // self.n_heads

    @property
    def n_heads(self) -> int:
        return self._config.num_attention_heads

    @property
    def intermediate_dim(self) -> int:
        return self._config.intermediate_size

    @property
    def n_heads_kv(self) -> int:
        return self._config.num_key_value_heads

    @property
    def activation_dtype(self) -> DtypeEnum:
        if self._config.torch_dtype == torch.float16:
            return DtypeEnum.fp16
        elif self._config.torch_dtype == torch.bfloat16:
            return DtypeEnum.bf16
        else:
            raise NotImplementedError("Only fp16 and bf16 are supported")

    @property
    def mlp_activation_fn(self) -> ActivationType:
        activation = self._config.hidden_act.lower()
        # llama model family is special and is always gated so force gated versions of relu, gelu, silu
        if activation == "gelu":
            return ActivationType.GEGLU
        elif activation == "relu":
            return ActivationType.ReGLU
        elif activation == "gegelu":
            return ActivationType.GEGLU
        elif activation == "silu":
            return ActivationType.SiGLU
        else:
            raise NotImplementedError(f"Activation {activation} not supported")

    @property
    def norm_type(self) -> NormTypeEnum:
        return NormTypeEnum.RMSNorm

    @property
    def positional_embedding_type(self) -> PositionalEmbeddingType:
        return PositionalEmbeddingType.rotate_half

    """
    Forward implementations
    """

    # TODO: overwrite v2 model forward
    def forward(self, wrapped_batch: RaggedBatchWrapper) -> torch.Tensor:
        pass
