"""
Transformer Block — satu lapisan penuh Aibys
Terdiri dari:
  Pre-norm Attention + Residual
  Pre-norm FFN + Residual

Menggunakan Pre-LayerNorm (norm sebelum sublayer),
lebih stabil saat training dibanding Post-LayerNorm.

Dibuat oleh Syahril Haryono
"""

import torch
import torch.nn as nn
from model.rmsnorm import RMSNorm
from model.attention import CausalSelfAttention
from model.ffn import SwiGLUFFN
from model.config import AibysConfig


class TransformerBlock(nn.Module):
    def __init__(self, config: AibysConfig):
        super().__init__()

        # Pre-norm sebelum attention
        self.norm1 = RMSNorm(config.d_model)

        # Causal Self-Attention dengan GQA + RoPE
        self.attn = CausalSelfAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_kv_heads=config.n_kv_heads,
            head_dim=config.head_dim,
            context_length=config.context_length,
            dropout=config.dropout,
            rope_theta=config.rope_theta
        )

        # Pre-norm sebelum FFN
        self.norm2 = RMSNorm(config.d_model)

        # SwiGLU FFN
        self.ffn = SwiGLUFFN(
            d_model=config.d_model,
            ffn_dim=config.ffn_dim,
            dropout=config.dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, d_model]
        Returns:
            [B, T, d_model]

        Alur:
            x → norm1 → attention → + x  (residual)
              → norm2 → ffn       → + x  (residual)
        """
        # Attention sublayer dengan residual connection
        x = x + self.attn(self.norm1(x))

        # FFN sublayer dengan residual connection
        x = x + self.ffn(self.norm2(x))

        return x


# =====================
# Quick test
# =====================
if __name__ == "__main__":
    from model.config import AibysConfig

    config = AibysConfig()
    block = TransformerBlock(config)

    x = torch.randn(2, 10, 512)
    out = block(x)

    print(f"Input  : {x.shape}")
    print(f"Output : {out.shape}")
    print(f"Params : {sum(p.numel() for p in block.parameters()):,}")
    print("TransformerBlock OK ✅")
