"""
RMSNorm — Root Mean Square Layer Normalization
Lebih efisien dari LayerNorm karena tidak perlu hitung mean.
Dipakai di LLaMA, Gemma, Mistral.

Paper: "Root Mean Square Layer Normalization" (Zhang & Sennrich, 2019)
Dibuat oleh Syahril Haryono
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        """
        Args:
            d_model: dimensi input
            eps: nilai kecil untuk stabilitas numerik
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))  # learnable scale (gamma)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        # RMS = sqrt(mean(x²))
        # normalized = x / RMS
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, d_model]
        Returns:
            [B, T, d_model] — normalized + scaled
        """
        # Hitung dalam float32 untuk stabilitas, lalu cast balik
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


# =====================
# Quick test
# =====================
if __name__ == "__main__":
    norm = RMSNorm(d_model=512)
    x = torch.randn(2, 10, 512)
    out = norm(x)
    print(f"Input shape  : {x.shape}")
    print(f"Output shape : {out.shape}")
    print(f"Params       : {sum(p.numel() for p in norm.parameters())}")
    print("RMSNorm OK ✅")
