from enum import Enum
from typing import Literal

from pydantic import Field, field_validator
from pydantic.dataclasses import dataclass


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    policy: Literal["fcfs", "priority"] = "fcfs"
    stream_interval: int = Field(default=1, ge=1)
