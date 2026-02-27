"""
Aibys-268M — Indonesian Decoder-only Language Model
Arsitektur lengkap dari token embedding sampai LM head.

Dibuat oleh Syahril Haryono
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
from model.config import AibysConfig
from model.rmsnorm import RMSNorm
from model.block import TransformerBlock


class Aibys(nn.Module):
    def __init__(self, config: AibysConfig):
        super().__init__()
        self.config = config

        # === Token Embedding ===
        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)

        # === Transformer Blocks (stacked) ===
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        # === Final RMSNorm ===
        self.norm_final = RMSNorm(config.d_model)

        # === LM Head — project ke vocab ===
        # no bias, weight tying dengan token_embed (hemat ~16M params)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: LM head berbagi weight dengan token embedding
        # Ini teknik standar yang dipakai GPT-2, LLaMA, dll
        self.lm_head.weight = self.token_embed.weight

        # === Inisialisasi weights ===
        self.apply(self._init_weights)

        # Hitung total params
        total = sum(p.numel() for p in self.parameters())
        print(f"✅ Aibys-268M initialized: {total:,} parameters ({total/1e6:.1f}M)")

    def _init_weights(self, module: nn.Module):
        """Inisialisasi weights standar GPT-style."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor = None
    ):
        """
        Args:
            input_ids : [B, T] — token IDs
            targets   : [B, T] — token IDs untuk dihitung loss (optional)

        Returns:
            logits    : [B, T, vocab_size]
            loss      : scalar cross-entropy loss (jika targets diberikan)
        """
        B, T = input_ids.shape
        assert T <= self.config.context_length, \
            f"Sequence length {T} melebihi context_length {self.config.context_length}"

        # === Token Embedding ===
        x = self.token_embed(input_ids)  # [B, T, d_model]

        # === Transformer Blocks ===
        for block in self.blocks:
            x = block(x)

        # === Final Norm ===
        x = self.norm_final(x)  # [B, T, d_model]

        # === LM Head ===
        logits = self.lm_head(x)  # [B, T, vocab_size]

        # === Hitung Loss (jika training) ===
        loss = None
        if targets is not None:
            # Cross-entropy: flatten untuk efisiensi
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=-1  # ignore padding
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """
        Generate teks baru dengan sampling.

        Args:
            input_ids      : [1, T] — prompt awal
            max_new_tokens : jumlah token baru yang di-generate
            temperature    : 1.0 = normal, <1.0 = lebih deterministic, >1.0 = lebih random
            top_k          : ambil top-k token sebelum sampling
            top_p          : nucleus sampling threshold

        Returns:
            [1, T + max_new_tokens]
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Potong jika melebihi context length
            ids = input_ids if input_ids.size(1) <= self.config.context_length \
                  else input_ids[:, -self.config.context_length:]

            # Forward pass
            logits, _ = self(ids)

            # Ambil logits token terakhir
            logits = logits[:, -1, :]  # [1, vocab_size]

            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature

            # Top-k filtering
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[sorted_indices_to_remove] = float("-inf")
                logits = torch.scatter(logits, 1, sorted_indices, sorted_logits)

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

    def save(self, path: str):
        """Simpan model weights + config."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({
            "model_state_dict": self.state_dict(),
            "config": self.config.to_dict(),
        }, path)
        print(f"💾 Model disimpan ke: {path}")

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "Aibys":
        """Load model dari file."""
        checkpoint = torch.load(path, map_location=device)
        config = AibysConfig.from_dict(checkpoint["config"])
        model = cls(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        print(f"📂 Model di-load dari: {path}")
        return model

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# =====================
# Quick test
# =====================
if __name__ == "__main__":
    config = AibysConfig()
    model = Aibys(config)

    # Test forward pass
    B, T = 2, 64
    input_ids = torch.randint(0, config.vocab_size, (B, T))
    targets = torch.randint(0, config.vocab_size, (B, T))

    logits, loss = model(input_ids, targets)
    print(f"Input  : {input_ids.shape}")
    print(f"Logits : {logits.shape}")
    print(f"Loss   : {loss.item():.4f}")

    # Test generate
    prompt = torch.randint(0, config.vocab_size, (1, 5))
    generated = model.generate(prompt, max_new_tokens=20, temperature=0.8)
    print(f"Generated : {generated.shape}")
    print("Aibys model OK ✅")
