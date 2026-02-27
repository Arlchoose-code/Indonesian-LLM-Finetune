"""
RoPE — Rotary Positional Embedding
Posisi di-encode langsung ke dalam Q dan K via rotasi,
bukan ditambahkan ke embedding seperti positional encoding biasa.

Keunggulan:
- Relative position awareness
- Bisa di-extend context length tanpa retrain
- Dipakai di LLaMA, Gemma, Mistral, Qwen

Paper: "RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al., 2021)
Dibuat oleh Syahril Haryono
"""

import torch
import torch.nn as nn
from typing import Tuple


def precompute_rope_freqs(
    head_dim: int,
    context_length: int,
    theta: float = 10000.0,
    device: torch.device = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute cosine dan sine frequencies untuk RoPE.

    Args:
        head_dim     : dimensi per attention head
        context_length: panjang sequence maksimum
        theta        : base frequency (default 10000)
        device       : CUDA/CPU

    Returns:
        cos, sin : masing-masing [context_length, head_dim]
    """
    assert head_dim % 2 == 0, "head_dim harus genap untuk RoPE"

    # Hitung inverse frequencies: θ_i = 1 / (theta^(2i/d))
    # Shape: [head_dim/2]
    i = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (theta ** (i / head_dim))

    # Posisi 0 sampai context_length-1
    # Shape: [context_length]
    positions = torch.arange(context_length, dtype=torch.float32, device=device)

    # Outer product: posisi × frekuensi
    # Shape: [context_length, head_dim/2]
    freqs = torch.outer(positions, inv_freq)

    # Duplikat: [context_length, head_dim]
    freqs = torch.cat([freqs, freqs], dim=-1)

    cos = freqs.cos()
    sin = freqs.sin()

    return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotasi separuh dimensi untuk RoPE.
    x = [x1, x2, ..., x_{d/2}, x_{d/2+1}, ..., x_d]
    → [-x_{d/2+1}, ..., -x_d, x1, ..., x_{d/2}]
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Terapkan RoPE ke Query dan Key.

    Args:
        q   : [B, n_heads, T, head_dim]
        k   : [B, n_kv_heads, T, head_dim]
        cos : [T, head_dim]
        sin : [T, head_dim]

    Returns:
        q_rotated, k_rotated dengan shape sama
    """
    # Ambil sesuai panjang sequence aktual
    T = q.shape[2]
    cos = cos[:T].unsqueeze(0).unsqueeze(0)  # [1, 1, T, head_dim]
    sin = sin[:T].unsqueeze(0).unsqueeze(0)  # [1, 1, T, head_dim]

    # Terapkan rotasi: x_rotated = x * cos + rotate_half(x) * sin
    q_rotated = (q * cos) + (rotate_half(q) * sin)
    k_rotated = (k * cos) + (rotate_half(k) * sin)

    return q_rotated, k_rotated


class RoPE(nn.Module):
    """
    RoPE module yang bisa di-attach ke model.
    Precompute frequencies saat init, apply saat forward.
    """
    def __init__(self, head_dim: int, context_length: int, theta: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.context_length = context_length
        self.theta = theta

        # Precompute dan register sebagai buffer (tidak di-train, tapi ikut .to(device))
        cos, sin = precompute_rope_freqs(head_dim, context_length, theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return apply_rope(q, k, self.cos, self.sin)


# =====================
# Quick test
# =====================
if __name__ == "__main__":
    rope = RoPE(head_dim=64, context_length=1024)

    B, T = 2, 10
    q = torch.randn(B, 8, T, 64)   # 8 query heads
    k = torch.randn(B, 4, T, 64)   # 4 KV heads (GQA)

    q_rot, k_rot = rope(q, k)
    print(f"Q input  : {q.shape}  → Q rotated : {q_rot.shape}")
    print(f"K input  : {k.shape}  → K rotated : {k_rot.shape}")
    print("RoPE OK ✅")
