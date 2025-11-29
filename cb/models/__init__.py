from dataclasses import dataclass
from typing import Optional

from torch import Tensor

from cb.models.modules import Cache


@dataclass
class ModelOutput:
    last_hidden_state: Tensor
    cache: Optional[Cache]
