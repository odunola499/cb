import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, embed_dim: int, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.embed_dim = embed_dim
        self.gamma = nn.Parameter(torch.ones(embed_dim))

    def forward(self, x: torch.Tensor):
        assert x.shape[-1] == self.embed_dim, (
            f"embed dim {self.embed_dim} does not match input shape {x.shape}"
        )
        rms = torch.sqrt(self.eps + (x**2).mean(dim=-1, keepdim=True))
        return (x / rms) * self.gamma


if __name__ == "__main__":
    inputs = torch.randn(1, 4, 8)
    local_rms = RMSNorm(8)
    nn_rms = nn.RMSNorm(8, eps=1e-6)

    local_output = local_rms(inputs)
    nn_output = nn_rms(inputs)

    result = torch.allclose(local_output, nn_output)
    print(result)
