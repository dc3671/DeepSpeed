# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import intel_extension_for_pytorch as ipex

import deepspeed
import deepspeed.comm as dist

try:
    ipex._C.disable_jit_linear_repack()
except Exception:
    pass
import json
import os
from pathlib import Path
from typing import Optional, Tuple, List

import torch
from huggingface_hub import snapshot_download
from torch.nn.modules import Module
from transformers import (AutoModelForCausalLM, AutoTokenizer, T5ForConditionalGeneration)
from transformers.models.llama import LlamaForCausalLM
from transformers.utils import is_offline_mode

from deepspeed import get_accelerator

from ..inference_utils import ActivationType, DtypeEnum, ceil_div
from ..modules.configs import (NormTypeEnum, PositionalEmbeddingType, RotateHalfConfig)
from ..ragged import DSSequenceDescriptor, KVCacheConfig, RaggedBatchWrapper
from .inference_transformer_base import DSTransformerModelBase

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
    file_list = [str(entry) for entry in Path(cached_repo_dir).rglob("*.[bp][it][n]") if entry.is_file()]
    return file_list


def write_checkpoints_json(model_name_or_path, checkpoints_json="checkpoints.json"):
    checkpoint_files = get_checkpoint_files(model_name_or_path)
    data = {"type": "BLOOM", "checkpoints": checkpoint_files, "version": 1.0}
    json.dump(data, open(checkpoints_json, "w"))


class XPUModel(DSTransformerModelBase, Module):

    def __init__(self, config, policy, base_mp_group, **kwargs):
        Module.__init__(self)
        self._policy = policy
        self._config = policy._model_config
        self._engine_config = config
        self._base_mp_group = base_mp_group
        self._kv_cache_config = None

        # Set to None until the Policy sets the model parameters
        self._non_transformer = None
        self._transformer = None
        self._flattened_param_buffer = None
        self._flattened_param_metadata = None

        self.model_name_or_path = policy._checkpoint_engine.model_name_or_path

        self.local_rank = get_int_from_env(["LOCAL_RANK", "MPI_LOCALRANKID", "PALS_LOCAL_RANKID"], "0")
        self.world_size = get_int_from_env(["WORLD_SIZE", "PMI_SIZE", "PALS_LOCAL_SIZE"], "1")
        # self.world_size = self.engine_config.tensor_parallel.tp_size
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
        config = self._config
        is_meta_support = model_type not in NO_META_SUPPORT_MODELS
        with deepspeed.OnDevice(dtype=self.load_dtype, device="meta", enabled=is_meta_support):
            if model_class[0] == AutoModelForCausalLM and is_meta_support:  # -> meta
                model = model_class[0].from_config(config, torch_dtype=self.load_dtype)
            else:  # -> host
                model = model_class[0].from_pretrained(
                    self.model_name_or_path,
                    config=config,
                    low_cpu_mem_usage=True,
                    torch_dtype=self.load_dtype,
                )
        # write ckpt.json
        repo_root = get_repo_root(self.model_name_or_path, self.local_rank)
        if (self.model_name_or_path in TP_PRESHARDED_MODELS) and self.kernel_inject:
            checkpoints_json = os.path.join(repo_root, "ds_inference_config.json")
        else:
            checkpoints_json = "checkpoints.json"
            if self.local_rank == 0:
                write_checkpoints_json(self.model_name_or_path, checkpoints_json)
            dist.barrier()
        # init DS::EngineV1
        model = deepspeed.init_inference(
            model,
            mp_size=self.world_size,
            base_dir=repo_root,
            dtype=self.infer_dtype,
            checkpoint=checkpoints_json if is_meta_support else None,
        )
        # optimize model
        model = ipex.optimize_transformers(
            model.eval(),
            dtype=self.infer_dtype,
            device=get_accelerator().device_name(),
            inplace=True,
        )
        # convert DS::EngineV1 -> nn.Module
        if isinstance(model, deepspeed.InferenceEngine):
            model = model.module
        self.model: LlamaForCausalLM = model

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

    @property
    def positional_embedding_config(self) -> Optional[RotateHalfConfig]:
        return RotateHalfConfig(theta_base=self._config.rope_theta)

    """
    Properties for compatibility
    """

    @property
    def q_block_size(self):
        return 8

    @property
    def kv_block_size(self):
        if self.head_size <= 64:
            return 128
        elif self.head_size != 160:
            return 64
        else:
            return 32

    def get_kv_requirements(self, sequence: DSSequenceDescriptor, max_new_tokens: int,
                            max_new_blocks: int) -> Tuple[int, torch.Tensor]:
        """
        See ``DSInferenceModelBase.get_kv_requirements`` for documentation.

        This method assumes an autoregressive dense attention pattern. Override this method
        if this does not match the model's attention pattern.
        """
        total_tokens = sequence.seen_tokens + max_new_tokens
        req_blocks = ceil_div(total_tokens, self.kv_block_size)
        block_lim = req_blocks - sequence.cur_allocated_blocks

        if block_lim <= max_new_blocks:
            return max_new_tokens, block_lim

        token_capacity = (max_new_blocks + sequence.cur_allocated_blocks) * self.kv_block_size - sequence.seen_tokens

        return token_capacity, torch.tensor([max_new_blocks])

    def kv_cache_config(self) -> Tuple[KVCacheConfig, ...]:
        """
        See ``DSInferenceModelBase.kv_cache_config`` for documentation.

        This method assumes an autoregressive dense attention pattern. Override this method
        if this does not match the model's attention pattern.
        """
        if self._kv_cache_config is None:
            cache_shape = (self.num_layers, self.n_heads_kv_local, self.head_size)
            max_blocks = ceil_div(self.max_sequence_length, self.kv_block_size)
            self._kv_cache_config = KVCacheConfig(
                block_size=self.kv_block_size,
                cache_shape=cache_shape,
                cache_dtype=self.activation_dtype,
                max_blocks_per_allocation_group=max_blocks,
            )
        return (self._kv_cache_config, )

    def get_remaining_block_capacity(self, sequence: DSSequenceDescriptor) -> int:
        return sequence.seen_tokens % self.kv_block_size

    def prepare_batch(self, wrapped_batch: RaggedBatchWrapper) -> None:
        pass

    """
    Forward implementations
    """

    def _prepare_position_ids(self, batch_size: int, seq_data: torch.Tensor):
        position_ids = []
        for seq_id in range(batch_size):
            # [H0 ~ H0+4], [H1, H1+1], ...
            n_tokens = seq_data[seq_id][1].item()
            seen_tokens = seq_data[seq_id][2].item()
            position_ids.extend(range(seen_tokens, seen_tokens + n_tokens))
        position_ids = torch.tensor(position_ids, device=get_accelerator().device_name(),
                                    dtype=torch.long).view(1, len(position_ids))
        return position_ids

    def _prepare_block_tables(self, kv_buffer: List[torch.Tensor], batch_size: int, max_blocks: int):
        block_tables = torch.empty((batch_size, max_blocks),
                                   device="cpu",
                                   dtype=torch.int32)
        for i in range(batch_size):
            block_tables[i] = kv_buffer[i][0, :max_blocks]
        return block_tables.to(device=get_accelerator().device_name())

    def _prepare_atoms(self, batch_size: int, seq_data: torch.Tensor) -> torch.Tensor:
        atoms = []
        n_atoms = 0
        decode_only = True
        for seq_id in range(batch_size):
            start_idx = seq_data[seq_id][0].item()
            n_tokens = seq_data[seq_id][1].item()
            if n_tokens > 1:
                decode_only = False
            seen_tokens = seq_data[seq_id][2].item()

            seq_atoms = ceil_div(n_tokens, self.q_block_size)

            for _ in range(seq_atoms):
                atom = [0] * 8
                # seq_id
                atom[0] = seq_id
                # q_start_idx
                atom[1] = start_idx
                # q_len
                q_len = min(n_tokens, self.q_block_size)
                atom[2] = q_len
                # kv_blocks = current total q_len with global start_idx
                atom[3] = -(-(seen_tokens + q_len) // self.kv_block_size)
                # total_extent
                atom[4] = seen_tokens + q_len
                # global_q_idx
                atom[5] = seen_tokens

                # update seq_data
                start_idx += q_len
                seen_tokens += q_len
                n_tokens -= q_len
                n_atoms += 1
                atoms.append(atom)
        return torch.tensor(atoms, dtype=torch.int, device=get_accelerator().device_name()), decode_only

    def _prepare_slot_mapping(self, batch_size: int, seq_data: torch.Tensor,
                              kv_buffer: List[torch.Tensor]) -> torch.Tensor:

        slot_mapping = []
        for seq_id in range(batch_size):
            n_tokens = seq_data[seq_id][1].item()
            seen_tokens = seq_data[seq_id][2].item()

            start_block_idx = seen_tokens // self.kv_block_size
            end_block_id = (n_tokens + seen_tokens + self.kv_block_size - 1) // self.kv_block_size
            local_start_idx = 0

            for block_idx in range(start_block_idx, end_block_id):
                mapped_block_id = kv_buffer[seq_id][0, block_idx].item()
                block_start_idx = seen_tokens % self.kv_block_size
                n_tokens_to_check = min(self.kv_block_size - block_start_idx, n_tokens - local_start_idx)
                block_end_idx = block_start_idx + n_tokens_to_check
                slot_mapping += [
                    i + mapped_block_id * self.kv_block_size for i in range(block_start_idx, block_end_idx)
                ]
                seen_tokens += n_tokens_to_check
                local_start_idx += n_tokens_to_check
        return torch.tensor(slot_mapping, dtype=torch.int, device=get_accelerator().device_name())

    def _forward_embed(self, ragged_batch: RaggedBatchWrapper) -> torch.Tensor:
        """
        Performs the embedding lookup prior to running the transformer of the model.

        Arguments:
            ragged_batch (RaggedBatchWrapper): The batch to embed.

        Returns:
            torch.Tensor: The embedded batch.
        """
        embed = self.model.model.embed_tokens(ragged_batch.input_ids())

        if embed.shape[-1] != self.model_dim:
            raise ValueError(f"Embedding output shape {embed.shape} does not match model_dim {self.model_dim}")

        return embed

    def _forward_unembed(self, hidden_states: torch.Tensor, ragged_batch_info: RaggedBatchWrapper) -> torch.Tensor:
        """
        Performs unembedding of the hidden states to logits. This will only sample the final
        token of each sequence.
        """
        return self.model.lm_head(hidden_states)

    def forward(self, wrapped_batch: RaggedBatchWrapper) -> torch.Tensor:
        # os.environ["PTI_ENABLE_COLLECTION"] = "1"
        input_ids = wrapped_batch.input_ids()
        batch_size = wrapped_batch.current_sequences
        seq_data = wrapped_batch.inflight_seq_descriptors(False)
        kv_buffer = wrapped_batch.kv_buffer()
        seqlen_kv = [seq_data[i][1].item() + seq_data[i][2].item() for i in range(batch_size)]
        seq_ids = [seq_data[i][0].item() + seq_data[i][1].item() - 1 for i in range(batch_size)]
        max_seqlen_kv = max(seqlen_kv)
        max_blocks = ceil_div(max_seqlen_kv, self.kv_block_size)

        block_tables = self._prepare_block_tables(kv_buffer, batch_size, max_blocks)
        slot_mapping = self._prepare_slot_mapping(batch_size, seq_data, kv_buffer)
        position_ids = self._prepare_position_ids(batch_size, seq_data)
        atoms, decode_only = self._prepare_atoms(batch_size, seq_data)

        hidden_states = self.model.model.embed_tokens(input_ids)

        for idx, decoder_layer in enumerate(self.model.model.layers):
            # cache_shape: (num_blocks, config.block_size, 2, num_heads, head_size)
            kv_cache = self.state_manager.get_cache(idx)
            hidden_states = decoder_layer(
                hidden_states,
                position_ids=position_ids,
                kv_cache=kv_cache,
                max_seqlen_kv=max_seqlen_kv,
                block_tables=block_tables,  # [num_seqs, max_blocks_per_seq]
                atoms=atoms,
                slot_mapping=slot_mapping,  # [num_tokens] per kv token -> block_idx
                decode_only=decode_only,
            )

        hidden_states = self.model.model.norm(hidden_states)
        # gather token output according to wrapped_batch
        hidden_states = hidden_states[:, seq_ids, :].view(batch_size, 1, -1)
        hidden_states = self.model.lm_head(hidden_states)
        # [bs*beam, seq, hidden_size] -> [seq, hidden_size]
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        # os.unsetenv("PTI_ENABLE_COLLECTION")
        return hidden_states
