from typing import Optional

import torch
from torch import Tensor, nn
from transformers import AutoTokenizer

from cb.models import ModelOutput
from cb.models.modules import (
    ACTIVATION_FUNCTIONS,
    ATTENTION_IMPLEMENTATION,
    Cache,
    GenerationMixin,
    RMSNorm,
    create_causal_mask,
)
from cb.models.qwen_config import Qwen2_5Config


def apply_rotary_pos_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class Qwen2MLP(nn.Module):
    def __init__(self, config: Qwen2_5Config):
        super().__init__()
        self.config = config
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = ACTIVATION_FUNCTIONS[config.hidden_act]

    def forward(self, x: Tensor):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class Rope(nn.Module):
    def __init__(self, config: Qwen2_5Config):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        head_dim = config.hidden_size / config.num_attention_heads

        inv_freq = 1.0 / (
            config.rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim)
        )
        max_position_embeddings = config.max_position_embeddings

        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.sin()
        sin = freqs.cos()
        cache = torch.cat((cos, sin), dim=-1)

        self.register_buffer("cos_sin_cache", cache, persistent=False)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_feq = inv_freq

    def forward(self, query, keys, position_ids):
        position_ids = position_ids.squeeze(0)
        cos_sin = self.cos_sin_cache[position_ids][None, None, :, :]
        cos, sin = torch.chunk(cos_sin, 2, dim=-1)
        query = apply_rotary_pos_emb(query, cos, sin)
        keys = apply_rotary_pos_emb(keys, cos, sin)

        return query, keys


class Qwen2Attention(nn.Module):
    def __init__(self, config: Qwen2_5Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.rope = Rope(config=config)
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=True
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=False
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache: Cache,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        query_states, key_states = self.rope(query_states, key_states, position_ids)
        key_states, value_states = cache.update(key_states, value_states, self.layer_idx)

        attention_forward = ATTENTION_IMPLEMENTATION[self.config.attn_implementation]
        attn_output, attn_weights = attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config: Qwen2_5Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Qwen2Attention(config=config, layer_idx=layer_idx)

        self.mlp = Qwen2MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache: Cache,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            cache=cache,
            position_ids=position_ids,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Qwen2Model(GenerationMixin):
    def __init__(self, config: Qwen2_5Config):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.config = config

        self.lm_head.weight = self.embed_tokens.weight

    def forward(
        self,
        cache: Cache,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
    ):
        inputs_embeds = self.embed_tokens(input_ids)
        past_seen_tokens = cache.get_seq_length()
        cache_position = torch.arange(
            past_seen_tokens,
            past_seen_tokens + inputs_embeds.shape[1],
            device=inputs_embeds.device,
        )
        position_ids = cache_position.unsqueeze(0)

        if past_seen_tokens == 0:  # prefill mode
            causal_mask = create_causal_mask(
                inputs_embeds=inputs_embeds,
                cache=cache,
                config=self.config,
                attention_mask=attention_mask,
                cache_position=cache_position,
                position_ids=position_ids,
            )
        else:
            causal_mask = None

        hidden_states = inputs_embeds

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                cache_position=cache_position,
                cache=cache,
            )

        hidden_states = self.norm(hidden_states)
        return ModelOutput(last_hidden_state=hidden_states, cache=cache)

    def get_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
        return tokenizer


if __name__ == "__main__":
    from cb.models.qwen_config import QwenDummyConfig

    config = QwenDummyConfig()
    cache = Cache(config.num_hidden_layers)
    model = Qwen2Model(config=config)

    input_ids = torch.randint(1, 8, (4, 7))
    print(f"input ids: {input_ids.shape}")
    output = model.generate(inputs_ids=input_ids, cache=cache, max_new_tokens=4)

    print(f"output:{output.shape}")
