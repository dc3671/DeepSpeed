# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

from ... import DSKernelBase
from ....inference_utils import ActivationType, DtypeEnum
from deepspeed.ops.op_builder import InferenceCutlassBuilder

from typing import Optional

PYTORCH_ACT_FN_MAP = {
    ActivationType.GELU: torch.nn.functional.gelu,
    ActivationType.SILU: torch.nn.functional.silu,
    ActivationType.RELU: torch.nn.functional.relu
}

class MoEGEMM(DSKernelBase):
    """
    CUTLASS implementation of MoE GEMM.
    """

    supported_dtypes = [DtypeEnum.fp16, DtypeEnum.bf16]
    supported_act_fns = [ActivationType.GELU, ActivationType.SILU, ActivationType.RELU, ActivationType.IDENTITY]

    def __init__(self, fp_dtype: DtypeEnum, act_fn: ActivationType) -> None:

        if not isinstance(fp_dtype, DtypeEnum):
            fp_dtype = DtypeEnum(fp_dtype)

        if fp_dtype not in MoEGEMM.supported_dtypes:
            raise ValueError("Unsupported data type: {}, supported_dtypes are {}".format(
                fp_dtype, MoEGEMM.supported_dtypes))

        if act_fn not in MoEGEMM.supported_act_fns:
            raise ValueError("Unsupported activation function: {}, supported_act_fns are {}".format(
                act_fn, MoEGEMM.supported_act_fns))

        # inf_module = InferenceCutlassBuilder().load()
        # self.kernel = inf_module.moe_gemm
        self.act_fn = act_fn
        
    def ref_moegemm(self,
                 ordered_output: torch.Tensor,
                 activations: torch.Tensor,
                 weights: torch.Tensor,
                 cumsum_rows: torch.Tensor,
                 biases: Optional[torch.Tensor] = None) -> None:
        """
        Arguments:
            ordered_output (torch.Tensor): The output of the MoE GEMM of shape [n_tokens, out_neurons].
            ordered_input (torch.Tensor): The direct input for the MoE GEMM of shape [n_tokens, in_neurons].
            weights (torch.Tensor): The weights of shape [n_experts, in_neurons, out_neurons]. These weights must be contiguous.
            total_rows_before_expert (torch.Tensor): The total number of rows before each expert of shape [n_experts].
            biases (torch.Tensor): The biases of shape [n_experts, out_neurons]. These biases must be contiguous.

        Returns:
            ordered_output
        """                 
        n_experts = weights.shape[0]
        for expert_idx in range(n_experts):
            start = cumsum_rows[expert_idx - 1] if expert_idx > 0 else 0
            end = cumsum_rows[expert_idx]
            activations_slice = activations[start:end]
            weights_slice = weights[expert_idx]
            output = torch.matmul(activations_slice, weights_slice)
            if biases is not None:
                output += biases[expert_idx]
            ordered_output[start:end] = output

        if self.act_fn != ActivationType.IDENTITY:
            act_fn_fn = PYTORCH_ACT_FN_MAP[self.act_fn]
            ordered_output = act_fn_fn(ordered_output)        

    def __call__(self,
                 ordered_output: torch.Tensor,
                 ordered_input: torch.Tensor,
                 weights: torch.Tensor,
                 total_rows_before_expert: torch.Tensor,
                 biases: Optional[torch.Tensor] = None) -> None:
        """
            Performs a MoE GEMM. Note that the stride between token inputs 
            must be even (the distance between byte 1 of token 0 and token 1 must be the same as the distance between byte 1 of token 1 and token 2).

            Arguments:
                ordered_output (torch.Tensor): The output of the MoE GEMM of shape [n_tokens, out_neurons].
                ordered_input (torch.Tensor): The direct input for the MoE GEMM of shape [n_tokens, in_neurons].
                weights (torch.Tensor): The weights of shape [n_experts, in_neurons, out_neurons]. These weights must be contiguous.
                total_rows_before_expert (torch.Tensor): The total number of rows before each expert of shape [n_experts].
                biases (torch.Tensor): The biases of shape [n_experts, out_neurons]. These biases must be contiguous.

            Returns:
                ordered_output
            """
        # self.kernel(ordered_output, ordered_input, weights, biases, total_rows_before_expert, self.act_fn)
        self.ref_moegemm(ordered_output, ordered_input, weights, total_rows_before_expert, biases)
        return ordered_output
