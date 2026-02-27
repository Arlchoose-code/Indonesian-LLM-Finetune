"""
Aibys-268M Configuration
Dibuat oleh Syahril Haryono
"""

from dataclasses import dataclass


@dataclass
class AibysConfig:
    # === Model Architecture ===
    # Target: ~268M params
    # Actual: 268,729,344 (~268.7M) ✅
    vocab_size: int = 32000
    d_model: int = 1024
    n_layers: int = 15
    n_heads: int = 16
    n_kv_heads: int = 8
    head_dim: int = 64
    ffn_dim: int = 4096
    context_length: int = 512      # ← turun jadi 512
    dropout: float = 0.1

    # === RoPE ===
    rope_theta: float = 10000.0

    # === Training ===
    # Disesuaikan untuk model 268M di RTX 4060 8GB
    # VRAM estimate: ~2.5 GB (aman!)
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_steps: int = 2000
    batch_size: int = 4
    grad_accum_steps: int = 8      # effective batch = 32
    max_steps: int = 103150
    eval_interval: int = 500
    save_interval: int = 1000
    log_interval: int = 10         # ← ganti jadi 10!

    # === Paths ===
    data_dir: str = "data"
    tokenizer_path: str = "tokenizer/aibys.model"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

    # === Metadata ===
    model_name: str = "Aibys-268M"
    author: str = "Syahril Haryono"
    language: str = "Indonesian"
    version: str = "1.0.0"

    def __post_init__(self):
        assert self.d_model == self.n_heads * self.head_dim, \
            f"d_model ({self.d_model}) harus = n_heads ({self.n_heads}) * head_dim ({self.head_dim})"
        assert self.n_heads % self.n_kv_heads == 0, \
            f"n_heads ({self.n_heads}) harus habis dibagi n_kv_heads ({self.n_kv_heads})"

    def to_dict(self):
        import dataclasses
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict):
        return cls(**d)


# Config default siap pakai
DEFAULT_CONFIG = AibysConfig()