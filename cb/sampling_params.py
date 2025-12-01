from dataclasses import dataclass


@dataclass
class SamplingParams:
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 0.0
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    max_tokens: int = 16
    min_tokens: int = 0
    ignore_eos: bool = False
