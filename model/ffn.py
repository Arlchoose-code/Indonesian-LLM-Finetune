"""
SwiGLU Feed-Forward Network
Lebih expressive dari FFN biasa (ReLU/GELU) karena ada gating mechanism.
Dipakai di LLaMA, Gemma, PaLM, Mistral.

Formula:
  FFN(x) = (SiLU(W_gate(x)) ⊙ W_up(x)) · W_down

Paper: "GLU Variants Improve Transformer" (Noam Shazeer, 2020)
Dibuat oleh Syahril Haryono
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.config import AibysConfig


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        d_model: int,
        ffn_dim: int,
        dropout: float = 0.1
    ):
        """
        Args:
            d_model : dimensi input/output (512)
            ffn_dim : dimensi inner layer (1365)
            dropout : dropout rate
        """
        super().__init__()

        # Tiga linear layers — no bias
        self.gate_proj = nn.Linear(d_model, ffn_dim, bias=False)  # W_gate
        self.up_proj   = nn.Linear(d_model, ffn_dim, bias=False)  # W_up
        self.down_proj = nn.Linear(ffn_dim, d_model, bias=False)  # W_down

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, d_model]
        Returns:
            [B, T, d_model]

        Formula:
            gate = SiLU(W_gate(x))    ← gating signal
            up   = W_up(x)            ← information
            hidden = gate ⊙ up        ← element-wise multiply (filtering)
            out  = W_down(hidden)
        """
        gate = F.silu(self.gate_proj(x))  # SiLU = x * sigmoid(x)
        up   = self.up_proj(x)
        hidden = gate * up                # gating: pilih info yang relevan
        out  = self.down_proj(hidden)
        out  = self.dropout(out)
        return out


# =====================
# Quick test
# =====================
if __name__ == "__main__":
    ffn = SwiGLUFFN(d_model=512, ffn_dim=1365, dropout=0.1)

    x = torch.randn(2, 10, 512)
    out = ffn(x)

    print(f"Input  : {x.shape}")
    print(f"Output : {out.shape}")
    print(f"Params : {sum(p.numel() for p in ffn.parameters()):,}")
    print("SwiGLUFFN OK ✅")
