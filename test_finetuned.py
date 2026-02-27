"""
Test Fine-tuned Aibys Model
Interactive chat interface
"""

import torch
import sentencepiece as spm
from pathlib import Path
import sys

# Add model path
sys.path.insert(0, str(Path(__file__).parent))
from model.aibys import Aibys
from model.config import AibysConfig


def load_finetuned_model(checkpoint_path, tokenizer_path, device="cuda"):
    """
    Load fine-tuned model (merged atau dengan LoRA)
    """
    print(f"📦 Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load(tokenizer_path)
    
    # Load model
    config = AibysConfig()
    model = Aibys(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded from step {checkpoint.get('step', 'unknown')}")
    if 'merged_from_lora' in checkpoint:
        print(f"   (Merged from LoRA with {checkpoint['merged_from_lora']['merged_layers']} layers)")
    
    return model, sp


def format_chat_prompt(user_message):
    """
    Format prompt untuk chat (conversational style)
    """
    prompt = f"""Pengguna: {user_message}
Aibys: """
    
    return prompt


@torch.no_grad()
def generate(model, sp, prompt, max_tokens=256, temperature=0.7, top_k=40, device="cuda"):
    """
    Generate response dari model
    """
    # Tokenize prompt
    tokens = sp.encode(prompt, add_bos=True, add_eos=False)
    tokens = torch.tensor([tokens], dtype=torch.long).to(device)
    
    generated = []
    
    for _ in range(max_tokens):
        # Forward
        logits = model(tokens)
        logits = logits[:, -1, :] / temperature
        
        # Top-k sampling
        if top_k > 0:
            values, indices = torch.topk(logits, top_k)
            logits[logits < values[:, -1:]] = float('-inf')
        
        # Sample
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Check EOS
        if next_token.item() == sp.eos_id():
            break
        
        generated.append(next_token.item())
        tokens = torch.cat([tokens, next_token], dim=1)
        
        # Truncate if too long (sliding window)
        if tokens.size(1) > 512:
            tokens = tokens[:, -512:]
    
    # Decode
    response = sp.decode(generated)
    return response


def interactive_chat(model, sp, device="cuda"):
    """
    Interactive chat loop
    """
    print("\n" + "="*60)
    print("  💬 AIBYS CHAT (Fine-tuned)")
    print("  Dibuat oleh Syahril Haryono")
    print("="*60)
    print("\nKetik 'quit' atau 'exit' untuk keluar")
    print("Ketik 'clear' untuk reset\n")
    
    while True:
        try:
            # Get user input
            user_msg = input("👤 Pengguna: ").strip()
            
            if not user_msg:
                continue
            
            if user_msg.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Sampai jumpa!")
                break
            
            if user_msg.lower() == 'clear':
                print("\n" + "="*60)
                print("  Conversation cleared")
                print("="*60 + "\n")
                continue
            
            # Format prompt
            prompt = format_chat_prompt(user_msg)
            
            # Generate
            print("🤖 Aibys: ", end="", flush=True)
            response = generate(
                model, sp, prompt,
                max_tokens=256,
                temperature=0.7,
                top_k=40,
                device=device
            )
            print(response)
            print()
        
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


def batch_test(model, sp, device="cuda"):
    """
    Test dengan beberapa pertanyaan pre-defined
    """
    test_questions = [
        "Siapa presiden Indonesia?",
        "Jelaskan apa itu fotosintesis",
        "Bagaimana cara membuat nasi goreng?",
        "Apa perbedaan antara kucing dan anjing?",
        "Tulis puisi pendek tentang malam",
        "Berikan 3 tips untuk belajar efektif",
    ]
    
    print("\n" + "="*60)
    print("  🧪 BATCH TEST")
    print("="*60)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n📝 Test {i}/{len(test_questions)}")
        print(f"Q: {question}")
        
        prompt = format_chat_prompt(question)
        response = generate(
            model, sp, prompt,
            max_tokens=150,
            temperature=0.7,
            top_k=40,
            device=device
        )
        
        print(f"A: {response}")
        print("-" * 60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test fine-tuned Aibys model")
    parser.add_argument("--checkpoint", default="checkpoints/final_lora_merged.pt", help="Path to model checkpoint")
    parser.add_argument("--tokenizer", default="tokenizer.model", help="Path to tokenizer")
    parser.add_argument("--mode", choices=['chat', 'test', 'both'], default='chat', help="Mode: chat, test, or both")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    
    args = parser.parse_args()
    
    print("="*60)
    print("  Aibys Fine-tuned Model Tester")
    print("="*60)
    
    # Load model
    model, sp = load_finetuned_model(args.checkpoint, args.tokenizer, args.device)
    
    # Run
    if args.mode in ['test', 'both']:
        batch_test(model, sp, args.device)
    
    if args.mode in ['chat', 'both']:
        interactive_chat(model, sp, args.device)


if __name__ == "__main__":
    main()
