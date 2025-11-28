from torch.nn import functional as F

from cb.models.modules.attention import ATTENTION_IMPLEMENTATION
from cb.models.modules.cache import Cache, CacheLayer, create_causal_mask
from cb.models.modules.norm import RMSNorm
from cb.models.modules.utils import (
    apply_rotary_pos_emb,
    compute_default_rope_parameters,
    rotate_half,
)

ACTIVATION_FUNCTIONS = {"silu": F.silu, "relu": F.relu, "leaky_relu": F.leaky_relu}
