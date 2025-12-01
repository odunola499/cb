from torch.nn import functional as F

from cb.models.modules.attention import ATTENTION_IMPLEMENTATION
from cb.models.modules.cache import Cache, CacheLayer, create_causal_mask
from cb.models.modules.generation import GenerationMixin
from cb.models.modules.norm import RMSNorm

ACTIVATION_FUNCTIONS = {"silu": F.silu, "relu": F.relu, "leaky_relu": F.leaky_relu}
