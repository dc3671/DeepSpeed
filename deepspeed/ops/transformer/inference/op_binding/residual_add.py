# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from ..config import DeepSpeedInferenceConfig
from .base import BaseOp


class ResidualAddOp(BaseOp):

    def __init__(self, config: DeepSpeedInferenceConfig):
        super(ResidualAddOp, self).__init__(config)
        try:
            if self.config.fp16 or self.config.q_int8:
                self.residual_add_func = self.inference_module.residual_add_bias_fp16
            elif self.config.bf16:
                self.residual_add_func = self.inference_module.residual_add_bias_bf16
            else:
                self.residual_add_func = self.inference_module.residual_add_bias_fp32
        except AttributeError:
            self.residual_add_func = None

        TGREEN =  '\033[32m' # Green Text
        ENDC = '\033[m' # reset to the defaults

        # self.residual_add_func = None
        if self.residual_add_func == None:
            print('<<<< residual_add_func fallback path <<<<')
        else :
            print(TGREEN + '<<<< residual_add_func kernel injection path <<<<' + ENDC)

    def forward(self, hidden_state: torch.Tensor, residual: torch.Tensor, attention_output: torch.Tensor,
                attention_bias: torch.Tensor, final_bias: torch.Tensor, add_bias: bool, residual_add: torch.Tensor):

        if not self.config.pre_layer_norm and residual_add is not None:
            # only use residual add if its set and we are not pre layer norm
            residual = residual_add

        if self.residual_add_func != None:
            self.residual_add_func(hidden_state, residual, attention_output, attention_bias, final_bias,
                                   self.config.mp_size, self.config.mlp_after_attn, add_bias,
                                   self.config.pre_layer_norm)
        else:
            # fallback
            if self.config.mlp_after_attn:
                if self.config.pre_layer_norm:
                    tmp = (residual.float() + attention_output.float() + attention_bias.float() +
                           final_bias.float()) / self.config.mp_size + hidden_state.float()
                else:
                    tmp = residual.float() + hidden_state.float() + final_bias.float()

                input_dtype = hidden_state.dtype
                residual = tmp.to(input_dtype)
            else:
                raise NotImplementedError
        return residual
