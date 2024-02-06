# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

from ....inference_utils import DtypeEnum
from deepspeed.ops.op_builder import RaggedOpsBuilder
from ....ragged import RaggedBatchWrapper
from ... import DSKernelBase
from deepspeed.accelerator import get_accelerator


class BlockedRotaryEmbeddings(DSKernelBase):
    """
    CUDA Kernel implementation that will perform rotary position embeddings on the queries and keys
    before copying into a blocked KV cache.
    """

    supported_dtypes = [DtypeEnum.fp16, DtypeEnum.bf16]
    supported_head_sizes = [64, 80, 128]
    supported_q_ratios = [1, 2, 4, 5, 8, 16, 29, 35, 36, 71]

    def __init__(self, head_size: int, n_q_heads: int, n_kv_heads: int, dtype: torch.dtype, rotary_dim: int,
                 theta_base: float) -> None:
        """
        Args:
            head_size: The size of the attention head.
            q_ratio: Ratio of q heads to kv heads (for GQA)
            dtype: Data type for the input/output. Supported values are torch.float16 and torch.bfloat16.
        """

        q_ratio = n_q_heads // n_kv_heads

        if head_size not in BlockedRotaryEmbeddings.supported_head_sizes:
            raise ValueError("Unsupported head size: {}, supported_head_sizes are {}".format(
                head_size, BlockedRotaryEmbeddings.supported_head_sizes))

        if q_ratio not in BlockedRotaryEmbeddings.supported_q_ratios:
            raise ValueError("Unsupported q_ratio: {}, supported_q_ratios are {}".format(
                q_ratio, BlockedRotaryEmbeddings.supported_q_ratios))

        if not isinstance(dtype, DtypeEnum):
            dtype = DtypeEnum(dtype)

        if dtype not in BlockedRotaryEmbeddings.supported_dtypes:
            raise ValueError("Unsupported data type: {}, supported_dtypes are {}".format(
                dtype, BlockedRotaryEmbeddings.supported_dtypes))

        inf_module = RaggedOpsBuilder().load()
        self.kernel = inf_module.kv_rotary_embeddings
        self.head_size = head_size
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.rotary_dim = rotary_dim
        self.theta_base = theta_base
        llama_max_position_embeddings = 2048
        self.cos, self.sin, _ = self.make_cos_sin_emb(llama_max_position_embeddings)        
    
    def make_cos_sin_emb(self, seq_len: int):
        t = torch.arange(seq_len, dtype=torch.float32, device=get_accelerator().current_device_name())
        inv_freq = (1.0 / (self.theta_base ** (torch.arange(
            0, self.rotary_dim, 2, dtype=torch.float32, device=get_accelerator().current_device_name()) / self.rotary_dim)))

        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        return torch.cos(emb)[:, None, :], torch.sin(emb)[:, None, :], inv_freq

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., :x.shape[-1] // 2]), dim=-1)
        
        
    def ref_rope_and_store_to_cache(self, kv_cache, q, k, v, rotary_dim, theta_base, batch_data, 
                     inflight_seq, kv_ptr_list):
        """                     
        kv_cache (torch.Tensor): Pre-allocated KV cache of [num_blocks, block_size, 2, n_kv_heads, head_size]                     
        """

        block_size = kv_cache.shape[1]
        head_size = kv_cache.shape[4]        
        n_heads_q = q.shape[1] // head_size
        n_heads_kv = k.shape[1] // head_size     
        
        num_seqs = batch_data[1]
        for seq_id in range(num_seqs):
            start_idx = inflight_seq[seq_id][0]
            n_tokens = inflight_seq[seq_id][1]
            seen_tokens = inflight_seq[seq_id][2]

            q_src = q[start_idx:start_idx + n_tokens].reshape(n_tokens, n_heads_q, head_size).float()
            k_src = k[start_idx:start_idx + n_tokens].reshape(n_tokens, n_heads_kv, head_size).float()
            v_src = v[start_idx:start_idx + n_tokens].reshape(n_tokens, n_heads_kv, head_size)
            freq_start_offset = inflight_seq[seq_id][2]

            q_src_rot = q_src[:, :, :rotary_dim]
            k_src_rot = k_src[:, :, :rotary_dim]

            cos_chunk = self.cos[range(freq_start_offset, freq_start_offset + n_tokens)]
            sin_chunk = self.sin[range(freq_start_offset, freq_start_offset + n_tokens)]

            q_rot = q_src_rot * cos_chunk + self.rotate_half(q_src_rot) * sin_chunk
            k_rot = k_src_rot * cos_chunk + self.rotate_half(k_src_rot) * sin_chunk

            q_emb = torch.cat((q_rot, q_src[:, :, rotary_dim:]), dim=-1)
            k_emb = torch.cat((k_rot, k_src[:, :, rotary_dim:]), dim=-1)

            q[start_idx:start_idx + n_tokens] = q_emb.reshape(n_tokens, n_heads_q * head_size).to(q.dtype)
            k[start_idx:start_idx + n_tokens] = k_emb.reshape(n_tokens, n_heads_q * head_size).to(q.dtype)
            
            # store KV to KV cache
            start_block_idx = seen_tokens // block_size
            end_block_id = (n_tokens + seen_tokens + block_size - 1) // block_size
            global_start_idx = seen_tokens // 1
            local_start_idx = 0 
            
            for block_idx in range(start_block_idx, end_block_id):
                mapped_block_id = kv_ptr_list[seq_id][0][block_idx]
                block_start_idx = global_start_idx % block_size
                n_tokens_to_check = min(block_size - block_start_idx, n_tokens - local_start_idx)
                block_end_idx = block_start_idx + n_tokens_to_check             
                kv_cache[mapped_block_id, block_start_idx:block_end_idx, 0, :, :] = \
                    k_emb[local_start_idx:local_start_idx + n_tokens_to_check].reshape(
                        n_tokens_to_check, n_heads_kv, head_size).to(kv_cache.dtype)
                kv_cache[mapped_block_id, block_start_idx:block_end_idx, 1, :, :] = \
                    v_src[local_start_idx:local_start_idx + n_tokens_to_check].reshape(
                        n_tokens_to_check, n_heads_kv, head_size).to(kv_cache.dtype)
                global_start_idx += n_tokens_to_check
                local_start_idx += n_tokens_to_check
                
                
    def ref_ipex_rope(self, kv_cache, q, k, v, rotary_dim, theta_base, batch_data, 
                     inflight_seq, kv_ptr_list):
        """                     
        kv_cache (torch.Tensor): Pre-allocated KV cache of [num_blocks, block_size, 2, n_kv_heads, head_size]                     
        """

        block_size = kv_cache.shape[1]
        head_size = kv_cache.shape[4]        
        n_heads_q = q.shape[1] // head_size
        n_heads_kv = k.shape[1] // head_size     
        
        total_tokens = batch_data[0]
        num_seqs = batch_data[1]

        position_id = []
        for seq_id in range(num_seqs):
            n_tokens = inflight_seq[seq_id][1]
            seen_tokens = inflight_seq[seq_id][2]
            position_id.extend(range(seen_tokens, seen_tokens + n_tokens))
        cos_chunk = self.cos[position_id]
        sin_chunk = self.sin[position_id]
        
        q_src = torch.reshape(q, [total_tokens, n_heads_q, head_size])
        q_src_rot = q_src[:, :, :rotary_dim]
        cos_q = cos_chunk.expand(q_src_rot.shape)
        sin_q = sin_chunk.expand(q_src_rot.shape)        
        torch.ops.torch_ipex.apply_rotary_embedding_half(q_src_rot, sin_q, cos_q, q_src_rot)
        
        k_src = torch.reshape(k, [total_tokens, n_heads_kv, head_size])
        k_src_rot = k_src[:, :, :rotary_dim]
        cos_k = cos_chunk.expand(k_src_rot.shape)
        sin_k = sin_chunk.expand(k_src_rot.shape)        
        torch.ops.torch_ipex.apply_rotary_embedding_half(k_src_rot, sin_k, cos_k, k_src_rot)        
        
        q = torch.cat((q_src_rot, q_src[:, :, rotary_dim:]), dim=-1).reshape(total_tokens, n_heads_q * head_size)
        k = torch.cat((k_src_rot, k_src[:, :, rotary_dim:]), dim=-1).reshape(total_tokens, n_heads_kv * head_size)

        # store KV to KV cache
        for seq_id in range(num_seqs):
            start_idx = inflight_seq[seq_id][0]
            n_tokens = inflight_seq[seq_id][1]
            seen_tokens = inflight_seq[seq_id][2]            
            
            k_src = k[start_idx:start_idx + n_tokens].reshape(n_tokens, n_heads_kv, head_size)
            v_src = v[start_idx:start_idx + n_tokens].reshape(n_tokens, n_heads_kv, head_size)                        
            
            start_block_idx = seen_tokens // block_size
            end_block_id = (n_tokens + seen_tokens + block_size - 1) // block_size
            global_start_idx = seen_tokens // 1
            local_start_idx = 0 
            
            for block_idx in range(start_block_idx, end_block_id):
                mapped_block_id = kv_ptr_list[seq_id][0][block_idx]
                block_start_idx = global_start_idx % block_size
                n_tokens_to_check = min(block_size - block_start_idx, n_tokens - local_start_idx)
                block_end_idx = block_start_idx + n_tokens_to_check             
                kv_cache[mapped_block_id, block_start_idx:block_end_idx, 0, :, :] = \
                    k_src[local_start_idx:local_start_idx + n_tokens_to_check].reshape(
                        n_tokens_to_check, n_heads_kv, head_size).to(kv_cache.dtype)
                kv_cache[mapped_block_id, block_start_idx:block_end_idx, 1, :, :] = \
                    v_src[local_start_idx:local_start_idx + n_tokens_to_check].reshape(
                        n_tokens_to_check, n_heads_kv, head_size).to(kv_cache.dtype)
                global_start_idx += n_tokens_to_check
                local_start_idx += n_tokens_to_check                
            

    def __call__(self, kv_cache: torch.Tensor, qkv: torch.Tensor, ragged_batch: RaggedBatchWrapper) -> None:
        """
        Perform rotary embeddings on the queries and keys before copying into a blocked KV cache.

        Args:
            kv_cache (torch.Tensor): Pre-allocated KV cache of [num_blocks, block_size, 2, n_kv_heads, head_size]
            qkv: Input tensor of shape [num_tokens, head_size * (n_q_heads + 2 * n_kv_heads)]
            ragged_batch: Wrapper for the ragged batch.
        """
        # TODO(zw): add code for kv_seq_len > max_position_embedding
        q = qkv[:, :self.head_size * self.n_q_heads]
        k = qkv[:, self.head_size * self.n_q_heads:self.head_size * (self.n_q_heads + self.n_kv_heads)]
        v = qkv[:, self.head_size * (self.n_q_heads + self.n_kv_heads):]
        
        self.ref_ipex_rope(kv_cache, q, k, v,
                self.rotary_dim, self.theta_base, ragged_batch.batch_metadata_buffer(), 
                     ragged_batch.inflight_seq_descriptors(), ragged_batch.kv_buffer())
        # self.kernel(kv_cache, q, k, v, self.rotary_dim, self.theta_base, ragged_batch.batch_metadata_buffer(),
        #             ragged_batch.inflight_seq_descriptors(), ragged_batch.tokens_to_seq(), ragged_batch.kv_ptrs())
