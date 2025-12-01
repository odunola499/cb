from typing import Dict, Optional

import torch
from torch import Tensor


def create_causal_mask(
    inputs_embeds, cache, attention_mask, cache_position=None, position_ids=None, config=None
):
    batch_size = inputs_embeds.shape[0]
    device = inputs_embeds.device
    dtype = inputs_embeds.dtype

    if cache is not None:
        seq_len = cache.get_seq_length() + inputs_embeds.shape[1]
    else:
        seq_len = inputs_embeds.shape[1]
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device))[None, None, :, :].expand(
        batch_size, 1, seq_len, seq_len
    )
    if attention_mask is not None:
        attention_mask = attention_mask[:, None, None, :]
        causal_mask = causal_mask * attention_mask

    causal_mask = (1.0 - causal_mask) * torch.finfo(dtype).min
    return causal_mask


class CacheLayer:
    def __init__(self):
        self.cached_keys = None
        self.cached_values = None
        self.cache_kwargs = None

        self.seq_length = 0

    def update(self, keys: Tensor, values: Tensor, cache_kwargs: Dict[str, torch.Tensor]):
        if self.cached_keys is not None:
            self.cached_keys = torch.cat([self.cached_keys, keys], dim=2)
        else:
            self.cached_keys = keys

        if self.cached_values is not None:
            self.cached_values = torch.cat([self.cached_values, values], dim=2)
        else:
            self.cached_values = values
        self.cache_kwargs = cache_kwargs

        self.seq_length += keys.shape[2]
        return self.cached_keys, self.cached_values

    def clear(self):
        self.cached_keys = self.cached_values = self.cache_kwargs = None
        self.seq_length = 0


class Cache:
    def __init__(self, num_layers: int):
        self.layer_caches = {i: CacheLayer() for i in range(num_layers)}

    def get_seq_length(self):
        return self.layer_caches[0].seq_length

    def update(self, keys: Tensor, values: Tensor, layer_idx: int, cache_kwargs: Optional = None):
        return self.layer_caches[layer_idx].update(keys, values, cache_kwargs)

    def clear(self, layer_idx: int):
        if layer_idx < 1:
            self.layer_caches = {i: CacheLayer() for i in range(len(self.layer_caches))}

        else:
            self.layer_caches[layer_idx].clear()
