from dataclasses import dataclass

import torch


@dataclass
class Qwen2_5Config:
    attention_dropout: float = 0.0
    bos_token_id: int = 151643
    eos_token_id: int = 151643
    pad_token_id: int = 151643
    hidden_act: str = "silu"
    hidden_size: int = 896
    intermediate_size: int = 4864
    max_position_embeddings: int = 32768
    max_window_layers: int = 24
    num_attention_heads: int = 14
    num_hidden_layers: int = 24
    num_key_value_heads: int = 2
    rms_norm_eps: float = 1e-6
    vocab_size: int = 151936
    rope_theta: int = 1000000
    attn_implementation: str = "eager"
    hf_repo: str = "Qwen/Qwen2.5-0.5B"
    model_name: str = "qwen2.5"
    dtype = torch.bfloat16


@dataclass
class QwenDummyConfig(Qwen2_5Config):
    bos_token_id: int = 15
    eos_token_id: int = 15
    pad_token_id: int = 15
    hidden_size: int = 32
    intermediate_size: int = 32
    max_position_embeddings: int = 128
    num_attention_heads: int = 4
    num_hidden_layers: int = 2
    num_key_value_heads: int = 4
    vocab_size: int = 128
    attn_implementation: str = "sdpa"
