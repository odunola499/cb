from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from cb.models.modules import Cache


class GenerationMixin(ABC, nn.Module):
    def prefill(self, input_ids, cache: Cache, temperature=0):
        print("Prefill")
        out = self(
            input_ids=input_ids,
            cache=cache,
        )
        hidden = out.last_hidden_state
        last_hidden_token = hidden[:, -1, :]
        logits = self.lm_head(last_hidden_token)
        next_token = self.sample(logits, temperature)
        print("End of Prefill")
        return next_token

    @abstractmethod
    def get_tokenizer(self):
        pass

    def sample(self, logits, temperature):
        if temperature < 0.01:
            return torch.argmax(logits, dim=-1)

        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def decode(self, input_ids, cache: Cache, max_new_tokens, temperature):
        print("Decode")
        next_token = input_ids
        generated = []

        for _ in range(max_new_tokens - 1):
            out = self(input_ids=next_token, cache=cache)
            logits = self.lm_head(out.last_hidden_state[:, -1, :])
            next_token = self.sample(logits, temperature).unsqueeze(-1)
            generated.append(next_token)

        generated = torch.cat([input_ids] + generated, dim=-1)
        print("End of Decode")
        return generated

    def generate(
        self,
        input_ids: Tensor,
        cache: Optional[Cache] = None,
        max_new_tokens=10,
        temperature=1,
    ):
        if cache is None:
            cache = Cache(num_layers=self.config.num_hidden_layers)
        next_token = self.prefill(input_ids, cache=cache, temperature=temperature).unsqueeze(-1)
        generated_result = self.decode(
            next_token, cache=cache, max_new_tokens=max_new_tokens, temperature=temperature
        )
        return generated_result
