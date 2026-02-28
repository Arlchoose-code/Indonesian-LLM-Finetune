# 🎯 Indonesian LLM Fine-tune

Kit untuk melakukan **fine-tuning** model LLM Bahasa Indonesia menggunakan teknik **LoRA (Low-Rank Adaptation)**. Dikembangkan oleh **Syahril Haryono** sebagai kelanjutan dari Indonesian LLM Starter.

> 💡 **Repo ini adalah lanjutan dari [Indonesian LLM Starter](https://github.com/syhrlhyn834/Indonesian-LLM-Starter) — pastikan kamu sudah punya model hasil pre-training sebelum fine-tuning di sini.**

---

## 🔗 Ekosistem Aibys

| Repo | Fungsi |
|---|---|
| 📦 [Aibys Data Collector](https://github.com/syhrlhyn834/Aibys-Data-Collector) | Kumpulkan & siapkan dataset untuk training |
| 🏗️ [Indonesian LLM Starter](https://github.com/syhrlhyn834/Indonesian-LLM-Starter) | Pre-training LLM dari scratch |
| 🎯 **Indonesian LLM Fine-tune** (repo ini) | Fine-tuning model hasil pre-training dengan LoRA |

**Alur lengkap:**
```
Aibys Data Collector    →    Indonesian LLM Starter    →    Indonesian LLM Fine-tune
(kumpul & siap data)         (pre-train model)               (fine-tune jadi assistant)
        ↓                            ↓                                  ↓
  train_shuffled.txt    →      aibys_final.pt           →        model siap chat
```

---

## 🎯 Untuk Apa Repo Ini?

Model hasil pre-training dari [Indonesian LLM Starter](https://github.com/syhrlhyn834/Indonesian-LLM-Starter) sudah bisa generate teks, tapi belum bisa diajak ngobrol seperti assistant. Repo ini mengubahnya jadi model yang bisa **menjawab pertanyaan dan mengikuti instruksi** — dengan cara yang efisien menggunakan LoRA.

---

## ✨ Kenapa LoRA?

Fine-tuning model 500M parameter secara penuh butuh VRAM yang besar dan waktu lama. LoRA menyelesaikan ini dengan cara:

- **Freeze** semua parameter base model (tidak dilatih ulang)
- **Tambah** matriks kecil A dan B di layer attention
- **Latih** hanya matriks kecil tersebut (~1-2% dari total parameter)
- **Hasilnya** hampir sama bagusnya dengan full fine-tuning, tapi jauh lebih cepat dan hemat VRAM

```
Full fine-tuning  : ~500M params dilatih  ❌ berat
LoRA fine-tuning  : ~5-10M params dilatih ✅ ringan
```

---

## 📐 Cara Kerja LoRA

LoRA menyisipkan matriks tambahan ke dalam layer attention:

```
# Sebelum LoRA
output = W_base(x)

# Sesudah LoRA
output = W_base(x) + (A @ B)(x) * scaling

# Dimana:
# W_base  = frozen, tidak berubah
# A, B    = matriks kecil yang dilatih (rank=16)
# scaling = alpha / rank
```

LoRA di-inject ke 4 projection layer di setiap attention block:
- `q_proj` — Query projection
- `k_proj` — Key projection
- `v_proj` — Value projection
- `out_proj` — Output projection

---

## 🗂️ Struktur Project

```
aibys-finetune/
│
├── finetune_lora.py      # 🚀  Entry point — jalankan ini untuk fine-tuning
├── merge_lora.py         # 🔀  Merge LoRA weights ke base model jadi satu file
├── chat_lora.py          # 💬  Chat interaktif dengan model LoRA (tanpa merge)
├── test_finetuned.py     # 🧪  Test model hasil fine-tune (chat / batch test)
│
├── model/                # Arsitektur model (sama seperti Indonesian LLM Starter)
│   ├── config.py
│   ├── aibys.py
│   ├── block.py
│   ├── attention.py
│   ├── ffn.py
│   ├── rmsnorm.py
│   ├── rope.py
│   └── __init__.py
│
├── checkpoints/          # Output fine-tuning (tidak di-commit)
│   ├── best_lora.pt      # Checkpoint dengan val loss terbaik
│   ├── final_lora.pt     # Checkpoint step terakhir
│   └── final_lora_merged.pt  # Hasil merge (siap deploy)
│
├── base_model.pt         # Base model dari pre-training (tidak di-commit)
└── tokenizer.model       # Tokenizer SentencePiece (tidak di-commit)
```

---

## ⚙️ Penjelasan Tiap File

### `finetune_lora.py` — Fine-tuning Utama
Script utama yang:
1. Load base model hasil pre-training
2. Inject LoRA ke layer attention
3. Freeze semua parameter kecuali LoRA
4. Load dataset `cahya/instructions_indonesian` dari HuggingFace
5. Fine-tune dengan training loop lengkap (mixed precision, LR scheduler, checkpoint)
6. Otomatis ganti nama AI lain (ChatGPT, Claude, Gemini, dll) jadi "Aibys" di data training

**LoRA Config default:**
```python
lora_rank = 16       # Rank matriks A dan B
lora_alpha = 32      # Scaling factor
lora_dropout = 0.05  # Dropout pada LoRA
```

### `merge_lora.py` — Gabungkan LoRA ke Base Model
Setelah fine-tuning, jalankan ini untuk menggabungkan LoRA weights ke base model:
```
W_final = W_base + (A @ B) * (alpha / rank)
```
Hasilnya satu file `.pt` yang bisa dipakai langsung tanpa perlu LoRA lagi.

### `chat_lora.py` — Chat Interaktif (dengan LoRA terpisah)
Load base model + LoRA weights secara terpisah dan langsung bisa chat. Cocok untuk quick test sebelum merge.

Format prompt yang dipakai:
```
Pengguna: {pertanyaan kamu}
Aibys: {jawaban model}
```

### `test_finetuned.py` — Test Model Hasil Merge
Test model yang sudah di-merge dengan dua mode:
- **`--mode chat`** — Chat interaktif bebas
- **`--mode test`** — Batch test dengan 6 pertanyaan preset
- **`--mode both`** — Keduanya

---

## 🚀 Cara Pakai dari Awal

### Prerequisite
- Sudah punya model hasil pre-training dari [Indonesian LLM Starter](https://github.com/syhrlhyn834/Indonesian-LLM-Starter)
- File `aibys_final.pt` (atau checkpoint terakhir) dari pre-training
- File `tokenizer/aibys.model` dari pre-training

### Step 1 — Clone & Install
```bash
git clone https://github.com/syhrlhyn834/indonesian-llm-finetune.git
cd indonesian-llm-finetune
pip install -r requirements.txt
```

### Step 2 — Siapkan File yang Dibutuhkan
Copy file dari hasil pre-training ke folder ini:
```bash
# Copy base model
cp ../Indonesian-LLM-Starter/checkpoints/aibys_final.pt ./base_model.pt

# Copy tokenizer
cp ../Indonesian-LLM-Starter/tokenizer/aibys.model ./tokenizer.model
```

### Step 3 — Fine-tuning
```bash
python finetune_lora.py
```

Training akan otomatis:
- Download dataset `cahya/instructions_indonesian` dari HuggingFace
- Log progress setiap step
- Evaluasi val loss setiap 250 steps
- Simpan checkpoint terbaik ke `checkpoints/best_lora.pt`
- Simpan checkpoint final ke `checkpoints/final_lora.pt`

### Step 4 — Merge LoRA ke Base Model
```bash
python merge_lora.py \
  --base base_model.pt \
  --lora checkpoints/best_lora.pt \
  --output checkpoints/final_lora_merged.pt
```

### Step 5 — Chat dengan Model
```bash
# Chat dengan model yang sudah di-merge (rekomendasi)
python test_finetuned.py --checkpoint checkpoints/final_lora_merged.pt --mode chat

# Atau chat langsung dengan LoRA (tanpa merge)
python chat_lora.py
```

### Step 6 — Batch Test
```bash
python test_finetuned.py --checkpoint checkpoints/final_lora_merged.pt --mode both
```

---

## 🔧 Konfigurasi

Edit `FineTuneConfig` di `finetune_lora.py`:

```python
class FineTuneConfig:
    # LoRA
    lora_rank = 16          # Lebih besar = lebih ekspresif, lebih berat
    lora_alpha = 32         # Biasanya 2x rank
    lora_dropout = 0.05

    # Training
    batch_size = 4
    grad_accum_steps = 4    # Effective batch = 4 x 4 = 16
    learning_rate = 3e-4
    max_steps = 5000        # 5K steps cukup untuk fine-tune
    warmup_steps = 100

    # Dataset
    max_samples = None      # None = pakai semua data
    context_length = 512
```

**Tips penyesuaian GPU:**

| VRAM | `batch_size` | `grad_accum_steps` | `lora_rank` |
|---|---|---|---|
| 16GB | 8 | 2 | 16 |
| 8GB | 4 | 4 | 16 |
| 6GB | 2 | 8 | 8 |

---

## 📦 Dependencies

```
torch>=2.1.0
sentencepiece>=0.1.99
datasets>=2.14.0
numpy>=1.24.0
tqdm>=4.66.0
```

```bash
pip install -r requirements.txt
```

---

## 🗺️ Roadmap

- [x] LoRA injection ke attention layers
- [x] Freeze base model, latih hanya LoRA
- [x] Dataset instruksi Bahasa Indonesia
- [x] Auto-replace nama AI lain jadi Aibys
- [x] Merge LoRA ke base model
- [x] Chat interaktif
- [x] Batch testing
- [ ] Support multi-turn conversation
- [ ] DPO / RLHF alignment
- [ ] Export ke GGUF setelah merge
- [ ] Support dataset custom (bawa data sendiri)

---

## 👤 Author

**Syahril Haryono** — Developer independen asal Indonesia.

---

## 📄 License

Apache 2.0 — bebas digunakan, dimodifikasi, dan didistribusikan dengan atribusi.

---

*Lanjutan dari [Indonesian LLM Starter](https://github.com/syhrlhyn834/Indonesian-LLM-Starter) — bangun AI assistant Bahasa Indonesia kamu sendiri. 🚀*
