"""
Grouped Query Attention (GQA) dengan RoPE dan Causal Mask
- n_heads query heads
- n_kv_heads key/value heads (lebih sedikit → hemat VRAM ~30%)
- Causal masking: token hanya bisa lihat token sebelumnya

Paper: "GQA: Training Generalized Multi-Query Transformer Models" (Ainslie et al., 2023)
Dibuat oleh Syahril Haryono
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model.rope import RoPE


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        context_length: int,
        dropout: float = 0.1,
        rope_theta: float = 10000.0
    ):
        """
        Args:
            d_model        : dimensi model (512)
            n_heads        : jumlah query heads (8)
            n_kv_heads     : jumlah KV heads untuk GQA (4)
            head_dim       : dimensi per head (64)
            context_length : max sequence length (1024)
            dropout        : attention dropout
            rope_theta     : RoPE base frequency
        """
        super().__init__()

        assert d_model == n_heads * head_dim
        assert n_heads % n_kv_heads == 0

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.n_rep = n_heads // n_kv_heads  # berapa kali KV di-repeat untuk match Q
        self.dropout_p = dropout

        # Linear projections — no bias (modern LLM style)
        self.q_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.out_proj = nn.Linear(n_heads * head_dim, d_model, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # RoPE
        self.rope = RoPE(head_dim, context_length, rope_theta)

        # Causal mask: upper triangle = -inf (tidak boleh lihat ke depan)
        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.full((context_length, context_length), float("-inf")),
                diagonal=1
            ),
            persistent=False
        )

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """
        Repeat KV heads agar jumlahnya match dengan Q heads (untuk GQA).
        [B, n_kv_heads, T, head_dim] → [B, n_heads, T, head_dim]
        """
        if self.n_rep == 1:
            return x
        B, n_kv, T, hd = x.shape
        x = x.unsqueeze(2).expand(B, n_kv, self.n_rep, T, hd)
        return x.reshape(B, n_kv * self.n_rep, T, hd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, d_model]
        Returns:
            [B, T, d_model]
        """
        B, T, _ = x.shape

        # === Linear projections ===
        q = self.q_proj(x)   # [B, T, n_heads * head_dim]
        k = self.k_proj(x)   # [B, T, n_kv_heads * head_dim]
        v = self.v_proj(x)   # [B, T, n_kv_heads * head_dim]

        # === Reshape ke [B, heads, T, head_dim] ===
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # === Apply RoPE ke Q dan K ===
        q, k = self.rope(q, k)

        # === Repeat KV untuk GQA ===
        k = self._repeat_kv(k)  # [B, n_heads, T, head_dim]
        v = self._repeat_kv(v)  # [B, n_heads, T, head_dim]

        # === Scaled Dot-Product Attention ===
        scale = math.sqrt(self.head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / scale  # [B, n_heads, T, T]

        # Apply causal mask
        attn_scores = attn_scores + self.causal_mask[:T, :T]

        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Context vector
        out = torch.matmul(attn_weights, v)  # [B, n_heads, T, head_dim]

        # === Merge heads ===
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)

        # === Output projection ===
        out = self.resid_dropout(self.out_proj(out))

        return out


# =====================
# Quick test
# =====================
if __name__ == "__main__":
    attn = CausalSelfAttention(
        d_model=512,
        n_heads=8,
        n_kv_heads=4,
        head_dim=64,
        context_length=1024,
        dropout=0.1
    )

    x = torch.randn(2, 10, 512)
    out = attn(x)
    print(f"Input  : {x.shape}")
    print(f"Output : {out.shape}")
    print(f"Params : {sum(p.numel() for p in attn.parameters()):,}")
    print("CausalSelfAttention OK ✅")
