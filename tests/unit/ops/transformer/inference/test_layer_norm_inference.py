"""
Copyright 2022 The Microsoft DeepSpeed Team
"""

import deepspeed
import numpy as np
import torch
import pytest
from deepspeed.accelerator import get_accelerator
from deepspeed.ops.op_builder import InferenceBuilder
import sys

np.set_printoptions(threshold=sys.maxsize)

if not deepspeed.ops.__compatible_ops__[InferenceBuilder.NAME]:
    pytest.skip("Inference ops are not available on this system",
                allow_module_level=True)

inference_module = None


def allclose(x, y):
    assert x.dtype == y.dtype
    rtol, atol = {torch.float32: (5e-4, 5e-5), torch.float16: (3e-2, 2e-3)}[x.dtype]
    # print(((x - y).abs() < 5e-5).cpu().numpy())
    np.testing.assert_allclose(x.cpu().numpy(),
                                      y.cpu().numpy(),
                                      rtol=rtol,
                                      atol=atol)


def run_layer_norm_reference(
    input: torch.Tensor,
    dim: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    epsilon: float,
):
    input_dtype = input.dtype
    output = torch.nn.functional.layer_norm(input,
                                            (input.shape[-1], ),
                                            gamma, beta, epsilon).to(input_dtype)
    return output


def run_layer_norm_ds(
    input: torch.Tensor,
    dim: tuple,
    gamma: float,
    beta: float,
    epsilon: float,
):
    global inference_module
    if inference_module is None:
        inference_module = InferenceBuilder().load()

    if input.dtype == torch.float16:
        layer_norm_func = inference_module.layer_norm_fp16
        allocate_workspace = inference_module.allocate_workspace_fp16
    elif attn_scores.dtype == torch.bfloat16:
        layer_norm_func = inference_module.layer_norm_bf16
        allocate_workspace = inference_module.allocate_workspace_bf16
    else:
        layer_norm_func = inference_module.layer_norm_fp32
        allocate_workspace = inference_module.allocate_workspace_fp32

    allocate_workspace(1024, 16, 33, 1, 1, 1, True, 0, 32, 0)

    return layer_norm_func(input, gamma, beta, epsilon)


@pytest.mark.inference_ops
@pytest.mark.parametrize('execution_number', range(1))
@pytest.mark.parametrize("dtype", [torch.float16])
def test_layer_norm(dtype, execution_number):
    num_batch = 1
    seq_length = 33
    hidden_states = 1024
    input = torch.randn(
        (num_batch,
         seq_length,
         hidden_states),
        dtype=dtype,
        device=get_accelerator().device_name(),
    )
    gamma = torch.randn((hidden_states), dtype=dtype, device=get_accelerator().device_name())
    beta = torch.randn((hidden_states), dtype=dtype, device=get_accelerator().device_name())
    epsilon = 1e-05

    ds_out = run_layer_norm_ds(input, (input.shape[2], ), gamma, beta, epsilon)
    ref_out = run_layer_norm_reference(input, (input.shape[2], ), gamma, beta, epsilon)

    allclose(ds_out, ref_out)
