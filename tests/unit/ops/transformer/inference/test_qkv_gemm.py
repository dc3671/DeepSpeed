"""
Copyright 2022 The Microsoft DeepSpeed Team
"""

import sys

import numpy as np
import pytest
import torch

import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed.ops.op_builder import InferenceBuilder

np.set_printoptions(threshold=sys.maxsize)

if not deepspeed.ops.__compatible_ops__[InferenceBuilder.NAME]:
    pytest.skip(
        "Inference ops are not available on this system", allow_module_level=True
    )

inference_module = None


def allclose(x, y):
    assert x.dtype == y.dtype
    rtol, atol = {torch.float32: (5e-4, 5e-5), torch.float16: (3e-2, 2e-3)}[x.dtype]
    # print(((x - y).abs() < 5e-5).cpu().numpy())
    np.testing.assert_allclose(x.cpu().numpy(), y.cpu().numpy(), rtol=rtol, atol=atol)


def run_qkv_gemm_reference(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    add_bias: bool,
    num_layers: int,
    num_heads: int = None,
    max_out_tokens: int = None,
):
    inp_norm = torch.nn.functional.layer_norm(
        input, (input.shape[2],), gamma, beta, 1e-6
    )
    tmp = torch.matmul(inp_norm, weight)
    if add_bias:
        tmp += bias
    output = [tmp, inp_norm]
    return output


def run_qkv_gemm_ds(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    add_bias: bool,
    num_layers: int,
    num_heads: int = None,
    max_out_tokens: int = None,
):
    global inference_module
    if inference_module is None:
        inference_module = InferenceBuilder().load()

    if input.dtype == torch.float16:
        qkv_gemm_func = inference_module.qkv_gemm_fp16
        allocate_workspace = inference_module.allocate_workspace_fp16
    elif input.dtype == torch.bfloat16:
        qkv_gemm_func = inference_module.qkv_gemm_bf16
        allocate_workspace = inference_module.allocate_workspace_bf16
    else:
        qkv_gemm_func = inference_module.qkv_gemm_fp32
        allocate_workspace = inference_module.allocate_workspace_fp32

    allocate_workspace(32, 32, 32, 32, 0, 1, True, 0, 32, 0)

    output = qkv_gemm_func(
        input,
        weight,
        torch.tensor(1.0),
        bias,
        gamma,
        beta,
        1e-6,
        add_bias,
        num_layers,
        False,
        1,
        0,
        False,
        False,
    )
    return output


@pytest.mark.inference_ops
@pytest.mark.parametrize('execution_number', range(5))
@pytest.mark.parametrize("dtype", [torch.float16])
def test_qkv_gemm(dtype, execution_number):
    input = torch.ones(
        (1, 1, 32),
        dtype=dtype,
        device=get_accelerator().device_name(),
    )
    weight = torch.randn(
        (32, 48),
        dtype=dtype,
        device=get_accelerator().device_name(),
    )
    bias = torch.randn(
        (48,),
        dtype=dtype,
        device=get_accelerator().device_name(),
    )
    gamma = torch.ones(
        (32,),
        dtype=dtype,
        device=get_accelerator().device_name(),
    )
    beta = torch.ones(
        (32,),
        dtype=dtype,
        device=get_accelerator().device_name(),
    )
    # input = torch.tensor(
    #    torch.arange(32).reshape(1, 1, 32),
    #    dtype=dtype,
    #    device=get_accelerator().device_name(),
    # )

    ds_out = run_qkv_gemm_ds(input, weight, bias, gamma, beta, False, 24, 8, None)
    ref_out = run_qkv_gemm_reference(
        input, weight, bias, gamma, beta, False, 24, 8, None
    )
    allclose(ds_out[1], ref_out[1])
    allclose(ds_out[0], ref_out[0])
