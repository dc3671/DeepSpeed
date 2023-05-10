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
    rtol, atol = {torch.float32: (5e-4, 5e-5), torch.float16: (3e-2, 2e-3), torch.bfloat16: (3e-2, 2e-3)}[x.dtype]
    # print(((x - y).abs() < 5e-5).cpu().numpy())
    np.testing.assert_allclose(x.cpu().numpy(), y.cpu().numpy(), rtol=rtol, atol=atol)


def run_vector_matmul_reference(
    input: torch.Tensor,
    weight: torch.Tensor,
    async_op: bool = False,
):
    return torch.matmul(input, weight)


def run_vector_matmul_ds(
    input: torch.Tensor,
    weight: torch.Tensor,
    async_op: bool = False,
):
    global inference_module
    if inference_module is None:
        inference_module = InferenceBuilder().load()

    if input.dtype == torch.float16:
        vector_matmul_func = inference_module.vector_matmul_fp16
        allocate_workspace = inference_module.allocate_workspace_fp16
    elif input.dtype == torch.bfloat16:
        vector_matmul_func = inference_module.vector_matmul_bf16
        allocate_workspace = inference_module.allocate_workspace_bf16
    else:
        vector_matmul_func = inference_module.vector_matmul_fp32
        allocate_workspace = inference_module.allocate_workspace_fp32

    allocate_workspace(32, 32, 32, 32, 0, 1, True, 0, 32, 0)

    return vector_matmul_func(input, weight, async_op, torch.tensor(1.0), False, False)


@pytest.mark.inference_ops
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_vector_matmul(dtype):
    input = torch.randn(
        (1, 1, 32),
        dtype=dtype,
        device=get_accelerator().device_name(),
    )
    weight = torch.randn(
        (32, 32),
        dtype=dtype,
        device=get_accelerator().device_name(),
    )
    # input = torch.tensor(
    #    torch.arange(32).reshape(1, 1, 32),
    #    dtype=dtype,
    #    device=get_accelerator().device_name(),
    # )

    ds_out = run_vector_matmul_ds(input, weight)
    ref_out = run_vector_matmul_reference(input, weight)
    allclose(ds_out, ref_out)
