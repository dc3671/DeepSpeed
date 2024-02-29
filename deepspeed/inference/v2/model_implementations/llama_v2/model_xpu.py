# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

#import intel_extension_for_pytorch as ipex
from intel_extension_for_pytorch.transformers.models.xpu.optimize_transformers.modules.transformer_modules.Mlp import \
    IPEXTransformerMLPOptimizedFp16SiluLlama
from intel_extension_for_pytorch.transformers.models.xpu.optimize_transformers.modules.transformer_modules.Norm import \
    LlamaRMSNorm

from ...inference_utils import ActivationType, DtypeEnum
from ...modules.configs import *
from ...modules.interfaces import *
from .. import *
from .model import Llama2InferenceModel


class Llama2InferenceXPUModel(Llama2InferenceModel):

    def make_norm_layer(self) -> None:
        assert self.norm_type == NormTypeEnum.RMSNorm
        self.norm = LlamaRMSNorm(self.model_dim)

    def make_qkv_layer(self) -> None:
        pass

    def make_attn_layer(self) -> None:
        pass

    def make_attn_out_layer(self) -> None:
        pass

    def make_mlp_1_layer(self) -> None:
        dtype = self.activation_dtype
        activation = self.mlp_activation_fn
        assert dtype == DtypeEnum.fp16
        assert activation == ActivationType.SiGLU
        return
        self.mlp_1 = IPEXTransformerMLPOptimizedFp16SiluLlama(self._config)

    def make_mlp_2_layer(self) -> None:
        dtype = self.activation_dtype
        activation = self.mlp_activation_fn
        assert dtype == DtypeEnum.fp16
        assert activation == ActivationType.SiGLU
        return
        self.mlp_1 = IPEXTransformerMLPOptimizedFp16SiluLlama()

    def make_embedding_layer(self) -> None:
        pass

    def make_unembedding_layer(self) -> None:
        pass
