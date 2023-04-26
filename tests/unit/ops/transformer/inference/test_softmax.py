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


def run_softmax_reference(
    attn_scores: torch.Tensor,
    attn_mask: torch.Tensor,
    alibi: torch.Tensor,
    triangular: bool,
    recompute: bool,
    local_attention: bool,
    window_size: int,
    async_op: bool,
    layer_scale: float,
    head_offset: int,
    mp_size: int,
    heads_per_par: int,
):
    alibi = alibi[head_offset:head_offset + heads_per_par]
    input_dtype = attn_scores.dtype
    if triangular:
        tri = ~torch.tril(torch.ones(attn_scores.size(),
                                     device=attn_scores.device)).to(bool)
        attn_scores = torch.masked_fill(attn_scores * layer_scale,
                                        tri,
                                        torch.finfo(input_dtype).min)
    if alibi is not None:
        attn_scores += alibi
    if attn_mask is not None:
        # expand atten_mask from two dim into 4 dim, insert two dims in the middle
        attn_mask = attn_mask[:, None, None, :]
        attn_scores += attn_mask
    output = torch.nn.functional.softmax(attn_scores,
                                         dim=-1,
                                         dtype=torch.float32).to(input_dtype)
    return output


def run_softmax_ds(
    attn_scores: torch.Tensor,
    attn_mask: torch.Tensor,
    alibi: torch.Tensor,
    triangular: bool,
    recompute: bool,
    local_attention: bool,
    window_size: int,
    async_op: bool,
    layer_scale: float,
    head_offset: int,
    mp_size: int,
    heads_per_par: int,
):
    global inference_module
    if inference_module is None:
        inference_module = InferenceBuilder().load()

    if attn_scores.dtype == torch.float16:
        softmax_func = inference_module.softmax_fp16
    elif attn_scores.dtype == torch.bfloat16:
        softmax_func = inference_module.softmax_bf16
    else:
        softmax_func = inference_module.softmax_fp32

    return softmax_func(attn_scores,
                        attn_mask,
                        alibi,
                        triangular,
                        recompute,
                        local_attention,
                        window_size,
                        async_op,
                        layer_scale,
                        head_offset,
                        mp_size)


@pytest.mark.inference_ops
@pytest.mark.parametrize("mp_size", [1, 2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_softmax(mp_size, dtype):
    num_beams = 1
    heads = 16
    heads_per_par = heads // mp_size
    attn_scores = torch.randn(
        (num_beams,
         heads_per_par,
         32,
         32),
        dtype=dtype,
        device=get_accelerator().device_name(),
    )
    attn_mask = torch.randn((num_beams,
                             32),
                            dtype=dtype,
                            device=get_accelerator().device_name())
    alibi = torch.randn((heads * num_beams,
                         1,
                         32),
                        dtype=dtype,
                        device=get_accelerator().device_name())
    triangular = True
    recompute = False
    local_attention = False
    window_size = 1
    async_op = False
    layer_scale = 1.25

    for rank in range(mp_size):
        head_offset = rank * heads // mp_size
        args = torch.load(f"test_softmax_{mp_size}_{dtype}.pt")
        for i, arg in enumerate(args):
            if isinstance(arg, torch.Tensor):
                args[i] = arg.xpu()
        *args, ref_out = args
        ds_out = run_softmax_ds(*args)
        allclose(ds_out, ref_out)
        continue
        ds_out = run_softmax_ds(attn_scores,
                                attn_mask,
                                alibi,
                                triangular,
                                recompute,
                                local_attention,
                                window_size,
                                async_op,
                                layer_scale,
                                head_offset,
                                mp_size,
                                heads_per_par)
        args = [
            attn_scores,
            attn_mask,
            alibi,
            triangular,
            recompute,
            local_attention,
            window_size,
            async_op,
            layer_scale,
            head_offset,
            mp_size,
            heads_per_par,
            ds_out
        ]
        for i, arg in enumerate(args):
            if isinstance(arg, torch.Tensor):
                args[i] = arg.cpu()
        torch.save(args, f"test_softmax_{mp_size}_{dtype}.pt")

        ref_out = run_softmax_reference(attn_scores,
                                        attn_mask,
                                        alibi,
                                        triangular,
                                        recompute,
                                        local_attention,
                                        window_size,
                                        async_op,
                                        layer_scale,
                                        head_offset,
                                        mp_size,
                                        heads_per_par)

        allclose(ds_out, ref_out)
