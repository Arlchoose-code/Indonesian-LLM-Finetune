import torch
import sentencepiece as spm
from pathlib import Path
import sys

# Tambahkan path ke folder model agar bisa import Aibys
sys.path.insert(0, str(Path(__file__).parent))
from model.aibys import Aibys
from model.config import AibysConfig

def generate_response(model, sp, prompt, max_new_tokens=100, device="cuda"):
    model.eval()
    # Format sesuai dataset training: "Pengguna: ... Aibys:"
    full_prompt = f"Pengguna: {prompt} Aibys:"
    
    tokens = torch.tensor([sp.encode(full_prompt, add_bos=True)], dtype=torch.long).to(device)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(tokens)
            logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
            logits = logits[:, -1, :] / 0.7
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            tokens = torch.cat([tokens, next_token], dim=1)
            
            if next_token.item() == sp.eos_id():
                break
                
    result = sp.decode(tokens[0].tolist())
    # Ambil hanya jawaban setelah kata "Aibys:"
    if "Aibys:" in result:
        return result.split("Aibys:")[-1].strip()
    return result

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = "checkpoints/best_lora.pt"
    base_model_path = "base_model.pt"
    tokenizer_path = "tokenizer.model"

    print("🤖 Menyiapkan Aibys untuk pengetesan...")
    
    # 1. Load Tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load(tokenizer_path)
    
    # 2. Inisialisasi Base Model
    config = AibysConfig()
    model = Aibys(config)
    base_checkpoint = torch.load(base_model_path, map_location=device)
    model.load_state_dict(base_checkpoint['model_state_dict'])
    
    # 3. Load & Apply LoRA Weights
    print(f"📦 Menyuntikkan LoRA dari {checkpoint_path}...")
    lora_checkpoint = torch.load(checkpoint_path, map_location=device)
    lora_state_dict = lora_checkpoint['lora_state_dict']
    
    # Masukkan weights LoRA ke parameter model yang sesuai
    model.load_state_dict(lora_state_dict, strict=False)
    model.to(device)
    
    print("\n✅ Aibys siap! Ketik 'keluar' untuk berhenti.")
    print("-" * 30)

    while True:
        user_input = input("Kamu: ")
        if user_input.lower() in ['keluar', 'exit', 'quit']:
            break
            
        response = generate_response(model, sp, user_input, device=device)
        print(f"Aibys: {response}\n")

if __name__ == "__main__":
    main()