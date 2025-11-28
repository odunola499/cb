from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from cb.models.modules import Cache


@dataclass
class ModelOutput:
    last_hidden_state: Tensor
    cache: Optional[Cache]


class ModelWrapper(ABC, nn.Module):
    @abstractmethod
    def get_tokenizer(self):
        pass

    def prefill(self, input_ids, cache: Cache, temperature=0):
        embeds = self.embed_tokens(input_ids)
        out = self(
            inputs_embeds=embeds,
            cache=cache,
            use_cache=True,
        )
        hidden = out.last_hidden_state
        last_hidden_token = hidden[:, -1, :]
        logits = self.lm_head(last_hidden_token)
        next_token = self.sample(logits, temperature)
        return next_token

    def sample(self, logits, temperature):
        logits /= max(0.01, temperature)
        probs = F.softmax(logits, dim=-1)
        next_token = torch.argmax(probs, dim=-1)
        return next_token

    def decode(self, input_ids, cache: Cache, max_new_tokens, temperature):
        next_token = input_ids  # B, 1, E
        generated = [next_token]

        for _ in range(max_new_tokens - 1):
            embeds = self.embed_tokens(next_token)
            out = self(
                inputs_embeds=embeds,
                cache=cache,
                use_cache=True,
            )
            hidden = out.last_hidden_state
            logits = self.lm_head(hidden)
            next_token = self.sample(logits, temperature=temperature)
            generated.append(next_token)

        generated = torch.cat([input_ids] + generated, dim=-1)
        return generated

    def generate(
        self,
        inputs_ids: Tensor,
        cache: Optional[Cache] = None,
        attention_mask: Optional[Tensor] = None,
        max_new_tokens=10,
        temperature=1,
    ):
        if cache is None:
            cache = Cache(num_layers=self.config.num_hidden_layers)
        next_token = self.prefill(inputs_ids, cache=cache, temperature=temperature)
        generated_result = self.decode(
            next_token, cache=cache, max_new_tokens=max_new_tokens, temperature=temperature
        )
        return generated_result
