# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from ..config import DeepSpeedInferenceConfig
from .base import BaseOp


class GELUGemmOp(BaseOp):

    def __init__(self, config: DeepSpeedInferenceConfig):
        super(GELUGemmOp, self).__init__(config)
        try:
            if self.config.fp16:
                self.fused_gemm_gelu = self.inference_module.fused_gemm_gelu_fp16
            elif self.config.bf16:
                self.fused_gemm_gelu = self.inference_module.fused_gemm_gelu_bf16
            else:
                self.fused_gemm_gelu = self.inference_module.fused_gemm_gelu_fp32
        except AttributeError:
            self.fused_gemm_gelu = None

        TGREEN =  '\033[32m' # Green Text
        ENDC = '\033[m' # reset to the defaults

        # self.fused_gemm_gelu = None
        if self.fused_gemm_gelu == None:
            print('<<<< fused_gemm_gelu fallback path <<<<')
        else :
            print(TGREEN + '<<<< fused_gemm_gelu kernel injection path <<<<' + ENDC)

    def forward(self,
                input: torch.Tensor,
                weight: torch.Tensor,
                bias: torch.Tensor,
                weight_out: torch.Tensor,
                async_op: bool = False):
        if self.fused_gemm_gelu != None:
            output = self.fused_gemm_gelu(input, weight, weight.scale if hasattr(weight, "scale") else torch.empty(1),
                                          bias, weight_out,
                                          weight_out.scale if hasattr(weight_out, "scale") else torch.empty(1),
                                          self.config.epsilon, self.config.pre_layer_norm, self.config.q_int8,
                                          async_op, self.config.transposed_mode)
        else:
            # fallback
            raise NotImplementedError
        return output
