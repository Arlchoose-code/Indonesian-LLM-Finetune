"""
Merge LoRA weights ke base model
Hasilnya: Full model yang bisa dipakai langsung tanpa LoRA
"""

import torch
from pathlib import Path
import sys

def merge_lora_to_base(base_checkpoint_path, lora_checkpoint_path, output_path):
    """
    Merge LoRA weights ke base model
    """
    print("="*60)
    print("  Merge LoRA → Base Model")
    print("="*60)
    
    # 1. Load base model
    print(f"\n📦 Loading base model from {base_checkpoint_path}...")
    base_ckpt = torch.load(base_checkpoint_path, map_location='cpu')
    base_state = base_ckpt['model_state_dict']
    
    # 2. Load LoRA checkpoint
    print(f"📦 Loading LoRA from {lora_checkpoint_path}...")
    lora_ckpt = torch.load(lora_checkpoint_path, map_location='cpu')
    lora_state = lora_ckpt['lora_state_dict']
    lora_config = lora_ckpt['config']
    
    print(f"\n🔧 LoRA Config:")
    print(f"   Rank: {lora_config['lora_rank']}")
    print(f"   Alpha: {lora_config['lora_alpha']}")
    print(f"   Scaling: {lora_config['lora_alpha'] / lora_config['lora_rank']}")
    
    # 3. Merge LoRA into base weights
    print(f"\n🔄 Merging LoRA weights...")
    
    # Group LoRA params by layer
    lora_layers = {}
    for name, param in lora_state.items():
        # Format: blocks.0.attn.c_attn.lora_A atau lora_B
        parts = name.split('.')
        layer_name = '.'.join(parts[:-1])  # Remove lora_A/lora_B
        param_type = parts[-1]  # lora_A or lora_B
        
        if layer_name not in lora_layers:
            lora_layers[layer_name] = {}
        lora_layers[layer_name][param_type] = param
    
    # Merge each LoRA layer
    scaling = lora_config['lora_alpha'] / lora_config['lora_rank']
    merged_count = 0
    
    for layer_name, lora_params in lora_layers.items():
        if 'lora_A' in lora_params and 'lora_B' in lora_params:
            # Find corresponding base weight
            # LoRA layer format: blocks.0.attn.c_attn
            # Base weight format: blocks.0.attn.c_attn.weight
            base_weight_name = layer_name + '.weight'
            
            if base_weight_name in base_state:
                # Compute LoRA delta: BA * scaling
                lora_A = lora_params['lora_A']  # [in_features, rank]
                lora_B = lora_params['lora_B']  # [rank, out_features]
                delta = (lora_A @ lora_B) * scaling  # [in_features, out_features]
                
                # Merge: W_new = W_base + delta^T (transpose because Linear uses transposed weights)
                base_state[base_weight_name] = base_state[base_weight_name] + delta.T
                
                merged_count += 1
                print(f"   ✅ Merged {layer_name}")
    
    print(f"\n✅ Merged {merged_count} LoRA layers")
    
    # 4. Create merged checkpoint
    merged_checkpoint = {
        'model_state_dict': base_state,
        'step': lora_ckpt['step'],
        'loss': lora_ckpt['loss'],
        'config': base_ckpt.get('config', {}),
        'merged_from_lora': {
            'base_checkpoint': str(base_checkpoint_path),
            'lora_checkpoint': str(lora_checkpoint_path),
            'lora_config': lora_config,
            'merged_layers': merged_count
        }
    }
    
    # 5. Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving merged model to {output_path}...")
    torch.save(merged_checkpoint, output_path)
    
    # Print size
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"   Size: {size_mb:.1f} MB")
    
    print("\n" + "="*60)
    print("✅ MERGE COMPLETE!")
    print(f"📁 Merged model: {output_path}")
    print("\nSekarang model bisa dipakai langsung tanpa LoRA!")
    print("="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Merge LoRA weights to base model")
    parser.add_argument("--base", required=True, help="Path to base model checkpoint")
    parser.add_argument("--lora", required=True, help="Path to LoRA checkpoint")
    parser.add_argument("--output", required=True, help="Path to save merged model")
    
    args = parser.parse_args()
    
    merge_lora_to_base(args.base, args.lora, args.output)
