from dataclasses import dataclass
from typing import Optional

from torch import Tensor

from cb.models.modules import Cache
from cb.models.qwen2 import Qwen2Model
from cb.models.qwen_config import Qwen2_5Config


@dataclass
class ModelOutput:
    last_hidden_state: Tensor
    cache: Optional[Cache]


MODELS = {"qwen2.5": (Qwen2_5Config, Qwen2Model)}
