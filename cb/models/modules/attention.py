from typing import Optional

import torch
from torch import nn


def repeat_kv(hidden_states: torch.Tensor, num_reps):
    batch_size, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    if num_reps == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch_size, num_key_value_heads, num_reps, seq_len, head_dim
    )
    hidden_states = hidden_states.reshape(
        batch_size, num_key_value_heads * num_reps, seq_len, head_dim
    )
    return hidden_states


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def torch_sdpa_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query, key_states, value_states, attn_mask=attention_mask, dropout_p=dropout, scale=scaling
    )
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, None


ATTENTION_IMPLEMENTATION = {"eager": eager_attention_forward, "sdpa": torch_sdpa_attention_forward}


if __name__ == "__main__":
    query = torch.randn(2, 6, 4, 8)
    key = torch.randn(2, 6, 4, 8)
    value = torch.randn(2, 6, 4, 8)

    batch_size = query.shape[0]
    seq_len = query.shape[-2]
    causal_mask = torch.tril(torch.ones(seq_len, seq_len))[None, None, :, :].expand(
        batch_size, 1, seq_len, seq_len
    )
    causal_mask = (1.0 - causal_mask) * torch.finfo(torch.float).min

    module = nn.Module()
    module.num_key_value_groups = 1

    output, _ = eager_attention_forward(
        module, query, key, value, attention_mask=causal_mask, scaling=1
    )
    sdpa_output, _ = torch_sdpa_attention_forward(
        module, query, key, value, attention_mask=causal_mask, scaling=1
    )
    print(output.shape)
    print(sdpa_output.shape)
    result = torch.allclose(output, sdpa_output)
    print(result)
