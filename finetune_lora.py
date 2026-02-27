"""
Aibys Fine-tuning dengan LoRA
Dataset: cahya/instructions_indonesian dari HuggingFace
Author: Syahril Haryono
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import sentencepiece as spm
from pathlib import Path
import json
from tqdm import tqdm
import os

# ============================================================================
# CONFIG
# ============================================================================

class FineTuneConfig:
    # Paths
    base_checkpoint = "base_model.pt"  # Base model (copied from latest.pt)
    tokenizer_path = "tokenizer.model"
    output_dir = "checkpoints"  # Output di folder ini
    
    # LoRA Config
    lora_rank = 16  # Rank untuk LoRA (8, 16, 32)
    lora_alpha = 32  # Alpha untuk scaling
    lora_dropout = 0.05
    
    # Training Config
    batch_size = 4
    grad_accum_steps = 4  # Effective batch = 4 x 4 = 16
    learning_rate = 3e-4
    max_steps = 5000  # 5K steps cukup untuk fine-tune
    warmup_steps = 100
    eval_interval = 250
    save_interval = 500
    
    # Model Config (harus sama dengan pre-training!)
    context_length = 512
    
    # Dataset
    max_samples = None  # None = pakai semua, atau angka (e.g., 10000)
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# ============================================================================
# LoRA LAYERS
# ============================================================================

class LoRALinear(nn.Module):
    """
    LoRA (Low-Rank Adaptation) layer
    W = W0 + BA where B is rank x d_out, A is d_in x rank
    """
    def __init__(self, in_features, out_features, rank=16, alpha=32, dropout=0.05):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.dropout = nn.Dropout(dropout)
        
        # Initialize
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)
        
        self.scaling = alpha / rank
    
    def forward(self, x, original_output):
        # LoRA forward: original + (dropout(x) @ A @ B) * scaling
        lora_out = (self.dropout(x) @ self.lora_A @ self.lora_B) * self.scaling
        return original_output + lora_out


def inject_lora(model, config):
    """
    Inject LoRA layers and register them as sub-modules
    """
    for name, module in model.named_modules():
        # Target: attention projections
        if isinstance(module, nn.Linear) and any(
            proj in name for proj in ['.q_proj', '.k_proj', '.v_proj', '.out_proj']
        ):
            in_features = module.in_features
            out_features = module.out_features
            
            # 1. Buat layer LoRA
            lora = LoRALinear(
                in_features, 
                out_features,
                rank=config.lora_rank,
                alpha=config.lora_alpha,
                dropout=config.lora_dropout
            )
            
            # 2. PENTING: Daftarkan lora sebagai sub-module agar terdeteksi model.parameters()
            # Kita tempelkan ke module linear aslinya
            module.lora_layer = lora 
            
            # 3. Simpan original forward
            module.original_linear_forward = module.forward
            
            # 4. Ganti forward module tersebut dengan closure yang benar
            def make_new_forward(m):
                def new_forward(x):
                    # Jalanin linear aslinya (W0)
                    orig_out = m.original_linear_forward(x)
                    # Tambahin hasil LoRA (BA) menggunakan atribut m.lora_layer
                    return m.lora_layer(x, orig_out)
                return new_forward
            
            module.forward = make_new_forward(module)
            print(f"✅ Injected LoRA into {name}")


def freeze_base_model(model):
    """
    Freeze all parameters except LoRA
    """
    for name, param in model.named_parameters():
        if 'lora' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
    
    # Count trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    
    print(f"\n📊 Parameters:")
    print(f"   Total: {total:,} ({total/1e6:.1f}M)")
    print(f"   Trainable (LoRA): {trainable:,} ({trainable/1e6:.1f}M)")
    print(f"   Frozen: {total - trainable:,} ({(total-trainable)/1e6:.1f}M)")
    print(f"   Trainable ratio: {trainable/total*100:.2f}%")


# ============================================================================
# DATASET
# ============================================================================

class InstructionDataset(Dataset):
    """
    Dataset untuk instruction following
    Format: cahya/instructions_indonesian dengan kolom 'label'
    """
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # AI names to filter/replace
        self.ai_names_to_replace = [
            "Asisten:", "Assistant:", "ChatGPT:", "GPT:", "AI:", 
            "Claude:", "Bard:", "LLaMA:", "Llama:", "Gemini:",
            "Copilot:", "Bing:", "Alexa:", "Siri:", "Google Assistant:"
        ]
    
    def clean_text(self, text):
        """
        Clean text: replace AI names dengan Aibys
        """
        # Replace semua AI assistant names dengan Aibys
        for ai_name in self.ai_names_to_replace:
            text = text.replace(ai_name, "Aibys:")
        
        # Tambahkan signature Aibys di akhir jika belum ada
        if "Aibys" not in text and "dibuat oleh" not in text.lower():
            # Hanya tambahkan di akhir conversation
            pass  # Skip untuk sekarang, biar natural
        
        return text
    
    def format_prompt(self, conversation_text):
        """
        Format prompt dari conversation text
        conversation_text sudah dalam format "Pengguna: ... Aibys: ..."
        """
        # Clean AI names
        conversation_text = self.clean_text(conversation_text)
        
        # Pastikan format konsisten
        conversation_text = conversation_text.strip()
        
        return conversation_text
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Data format: item['label'] adalah list dengan 1 element berisi conversation
        # Example: ["Pengguna: ... Aibys: ..."]
        
        if 'label' in item and isinstance(item['label'], list) and len(item['label']) > 0:
            conversation_text = item['label'][0]
        else:
            # Fallback jika format berbeda
            print(f"⚠️  Warning: Unexpected format at index {idx}")
            return self.__getitem__((idx + 1) % len(self.data))  # Skip to next
        
        # Format dan clean
        full_text = self.format_prompt(conversation_text)
        
        # Tokenize
        tokens = self.tokenizer.encode(full_text, add_bos=True, add_eos=True)
        
        # Truncate or pad
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        
        # Convert to tensor
        tokens = torch.tensor(tokens, dtype=torch.long)
        
        # Create input and target (shift by 1)
        x = tokens[:-1]
        y = tokens[1:]
        
        # Pad if needed
        if len(x) < self.max_length - 1:
            pad_len = self.max_length - 1 - len(x)
            x = torch.cat([x, torch.zeros(pad_len, dtype=torch.long)])
            y = torch.cat([y, torch.full((pad_len,), -100, dtype=torch.long)])  # -100 = ignore in loss
        
        return x, y


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train(model, train_loader, val_loader, config):
    """
    Training loop dengan LoRA
    """
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Optimizer (only LoRA params)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1
    )
    
    # LR scheduler with warmup
    def get_lr(step):
        if step < config.warmup_steps:
            return config.learning_rate * (step + 1) / config.warmup_steps
        else:
            # Cosine decay
            progress = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
            return config.learning_rate * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
    
    # Training state
    step = 0
    best_val_loss = float('inf')
    
    # Loss function
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    # Mixed precision
    scaler = torch.amp.GradScaler('cuda') if config.device == "cuda" else None
    
    print("\n🚀 Starting fine-tuning...")
    print(f"   Max steps: {config.max_steps}")
    print(f"   Batch size: {config.batch_size} x {config.grad_accum_steps} = {config.batch_size * config.grad_accum_steps}")
    print(f"   Learning rate: {config.learning_rate}")
    print(f"   Device: {config.device}\n")
    
    model.train()
    optimizer.zero_grad()
    
    pbar = tqdm(total=config.max_steps, desc="Fine-tuning")
    
    while step < config.max_steps:
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(config.device), y.to(config.device)
            
            # Forward
            # Forward
            with torch.amp.autocast('cuda', dtype=config.dtype) if config.device == "cuda" else torch.no_grad():
                # Tangkap output sebagai tuple
                outputs = model(x)
                # Ambil hanya logits-nya (elemen pertama)
                logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
                
                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                loss = loss / config.grad_accum_steps
            
            # Backward
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update every grad_accum_steps
            if (batch_idx + 1) % config.grad_accum_steps == 0:
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                optimizer.zero_grad()
                
                # Update LR
                lr = get_lr(step)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                
                step += 1
                pbar.update(1)
                pbar.set_postfix({'loss': f'{loss.item() * config.grad_accum_steps:.4f}', 'lr': f'{lr:.2e}'})
                
                # Eval
                if step % config.eval_interval == 0:
                    val_loss = evaluate(model, val_loader, criterion, config)
                    print(f"\n📊 Step {step} | Val Loss: {val_loss:.4f}")
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_checkpoint(model, optimizer, step, val_loss, config, best=True)
                    
                    model.train()
                
                # Save
                if step % config.save_interval == 0:
                    save_checkpoint(model, optimizer, step, loss.item(), config)
                
                if step >= config.max_steps:
                    break
        
        if step >= config.max_steps:
            break
    
    pbar.close()
    
    # Final save
    save_checkpoint(model, optimizer, step, loss.item(), config, final=True)
    print(f"\n✅ Fine-tuning complete! Best val loss: {best_val_loss:.4f}")


def evaluate(model, val_loader, criterion, config):
    """
    Evaluate on validation set
    """
    model.eval()
    total_loss = 0
    count = 0
    
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(config.device), y.to(config.device)
            
            with torch.amp.autocast('cuda', dtype=config.dtype) if config.device == "cuda" else torch.no_grad():
                outputs = model(x)
                logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
                
                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            
            total_loss += loss.item()
            count += 1
    
    return total_loss / count if count > 0 else 0


def save_checkpoint(model, optimizer, step, loss, config, best=False, final=False):
    """
    Save LoRA checkpoint (only trainable params)
    """
    # Extract LoRA params only
    lora_state = {
        name: param.data.clone()
        for name, param in model.named_parameters()
        if 'lora' in name
    }
    
    checkpoint = {
        'step': step,
        'lora_state_dict': lora_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': {
            'lora_rank': config.lora_rank,
            'lora_alpha': config.lora_alpha,
            'lora_dropout': config.lora_dropout,
        }
    }
    
    # Save paths
    if best:
        path = Path(config.output_dir) / "best_lora.pt"
        print(f"💾 Saving BEST checkpoint to {path}")
    elif final:
        path = Path(config.output_dir) / "final_lora.pt"
        print(f"💾 Saving FINAL checkpoint to {path}")
    else:
        path = Path(config.output_dir) / f"lora_step_{step:06d}.pt"
    
    torch.save(checkpoint, path)


# ============================================================================
# MAIN
# ============================================================================

def main():
    config = FineTuneConfig()
    
    print("="*60)
    print("  Aibys Fine-tuning dengan LoRA")
    print("  Dataset: cahya/instructions_indonesian")
    print("="*60)
    
    # 1. Load tokenizer
    print("\n📝 Loading tokenizer...")
    
    # Auto-detect tokenizer path
    tokenizer_candidates = [
        config.tokenizer_path,
        "tokenizer.model",
        "tokenizer/aibys.model",
        "aibys.model",
    ]
    
    tokenizer_path = None
    for candidate in tokenizer_candidates:
        if Path(candidate).exists():
            tokenizer_path = candidate
            break
    
    if not tokenizer_path:
        print(f"❌ Error: Tokenizer tidak ditemukan!")
        print(f"   Dicari di: {tokenizer_candidates}")
        return
    
    sp = spm.SentencePieceProcessor()
    sp.load(tokenizer_path)
    print(f"   Loaded: {tokenizer_path}")
    print(f"   Vocab size: {sp.vocab_size()}")
    
    # 2. Load base model
    print("\n📦 Loading base model...")
    checkpoint = torch.load(config.base_checkpoint, map_location=config.device)
    
    # Import model (assumes model code is available)
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from model.aibys import Aibys
    from model.config import AibysConfig
    
    model_config = AibysConfig()
    model = Aibys(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"   ✅ Loaded from step {checkpoint.get('step', 'unknown')}")
    
    # 3. Inject LoRA
    print("\n🔧 Injecting LoRA layers...")
    inject_lora(model, config)
    freeze_base_model(model)

    model = model.to(config.device) 
    print(f"🚀 Model & LoRA moved to {config.device}")
    
    # 4. Load dataset
    print("\n📥 Loading dataset from HuggingFace...")
    dataset = load_dataset("cahya/instructions_indonesian")
    
    if config.max_samples:
        train_data = dataset['train'].select(range(min(config.max_samples, len(dataset['train']))))
    else:
        train_data = dataset['train']
    
    # Split train/val (90/10)
    split_idx = int(len(train_data) * 0.9)
    train_split = train_data.select(range(split_idx))
    val_split = train_data.select(range(split_idx, len(train_data)))
    
    print(f"   Train samples: {len(train_split)}")
    print(f"   Val samples: {len(val_split)}")
    
    # 5. Create datasets
    print("\n🔄 Creating PyTorch datasets...")
    train_dataset = InstructionDataset(train_split, sp, config.context_length)
    val_dataset = InstructionDataset(val_split, sp, config.context_length)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,  # Windows compatibility
        pin_memory=True if config.device == "cuda" else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if config.device == "cuda" else False
    )
    
    # 6. Train!
    train(model, train_loader, val_loader, config)
    
    print("\n" + "="*60)
    print("✅ FINE-TUNING COMPLETE!")
    print(f"📁 Checkpoints saved in: {config.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
