#!/usr/bin/env python3
"""
Script to load a checkpoint and generate text completions.
"""

import sys
import json
from pathlib import Path
import argparse
import torch
from contextlib import nullcontext

# Add src to path to import modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

from models.utils import get_model
from data.utils import get_dataset, DataReader
from optim.utils import eval, get_batch


def find_checkpoint_file(checkpoint_path):
    """Find the checkpoint file from various possible paths."""
    checkpoint_path = Path(checkpoint_path)
    
    if checkpoint_path.is_file() and checkpoint_path.suffix == ".pt":
        # If checkpoint is in ckpts/latest/main.pt, exp_dir is parent of ckpts
        if checkpoint_path.parent.name == "latest" and checkpoint_path.parent.parent.name == "ckpts":
            exp_dir = checkpoint_path.parent.parent.parent
        # If checkpoint is in ckpts/<step>/main.pt, exp_dir is parent of ckpts
        elif checkpoint_path.parent.parent.name == "ckpts":
            exp_dir = checkpoint_path.parent.parent.parent
        # Otherwise, exp_dir is parent of the checkpoint file
        else:
            exp_dir = checkpoint_path.parent
        return checkpoint_path, exp_dir
    
    if checkpoint_path.is_dir():
        # If we're in a ckpts subdirectory, go up to exp_dir
        if checkpoint_path.name == "latest" and checkpoint_path.parent.name == "ckpts":
            exp_dir = checkpoint_path.parent.parent
            ckpt_file = checkpoint_path / "main.pt"
            if ckpt_file.exists():
                return ckpt_file, exp_dir
        elif checkpoint_path.name == "ckpts":
            exp_dir = checkpoint_path.parent
            ckpt_file = checkpoint_path / "latest" / "main.pt"
            if ckpt_file.exists():
                return ckpt_file, exp_dir
        
        # Try common checkpoint locations
        for ckpt_file in [
            checkpoint_path / "ckpts" / "latest" / "main.pt",
            checkpoint_path / "main.pt",
            checkpoint_path / "ckpt.pt",
        ]:
            if ckpt_file.exists():
                exp_dir = checkpoint_path
                return ckpt_file, exp_dir
    
    raise FileNotFoundError(
        f"Could not find checkpoint file. Tried:\n"
        f"  - {checkpoint_path / 'ckpts' / 'latest' / 'main.pt'}\n"
        f"  - {checkpoint_path / 'main.pt'}\n"
        f"  - {checkpoint_path / 'ckpt.pt'}"
    )


def load_config(checkpoint_path):
    """Load config from summary.json in the experiment directory."""
    ckpt_file, exp_dir = find_checkpoint_file(checkpoint_path)
    
    summary_file = exp_dir / "summary.json"
    if not summary_file.exists():
        raise FileNotFoundError(
            f"Could not find summary.json at {summary_file}. "
            f"This file is required to load the model configuration."
        )
    
    with open(summary_file, "r") as f:
        summary = json.load(f)
    
    if "args" not in summary:
        raise ValueError(f"summary.json does not contain 'args' key")
    
    # Convert dict to Namespace
    config = argparse.Namespace(**summary["args"])
    return config, ckpt_file


def load_model(checkpoint_path, device="cuda:0"):
    """Load model from checkpoint."""
    print(f"Loading model from: {checkpoint_path}")
    
    # Load config
    config, ckpt_file = load_config(checkpoint_path)
    
    # Override settings for inference
    config.device = device
    config.dropout = 0.0
    if hasattr(config, "compile"):
        config.compile = False
    
    # Disable distributed settings
    if hasattr(config, 'world_size'):
        config.world_size = 1
    if hasattr(config, 'distributed_backend'):
        config.distributed_backend = None
    
    print(f"\nModel configuration (from summary.json):")
    print(f"  Model type: {config.model}")
    print(f"  n_layer: {config.n_layer}")
    print(f"  n_embd: {config.n_embd}")
    print(f"  n_head: {config.n_head}")
    print(f"  vocab_size: {config.vocab_size}")
    print(f"  sequence_length: {config.sequence_length}")
    if hasattr(config, 'mlp_hidden_dim') and config.mlp_hidden_dim is not None:
        print(f"  mlp_hidden_dim: {config.mlp_hidden_dim}")
    elif hasattr(config, 'mlp_dim_exp_factor'):
        mlp_dim = int(config.n_embd * 4 * config.mlp_dim_exp_factor)
        print(f"  mlp_hidden_dim: {mlp_dim} (4 * n_embd * {config.mlp_dim_exp_factor})")
    if hasattr(config, 'qkv_dim') and config.qkv_dim is not None:
        print(f"  qkv_dim: {config.qkv_dim}")
    if hasattr(config, 'bias'):
        print(f"  bias: {config.bias}")
    if hasattr(config, 'weight_tying'):
        print(f"  weight_tying: {config.weight_tying}")
    if hasattr(config, 'dropout'):
        print(f"  dropout: {config.dropout} (overridden to 0.0 for inference)")
    if hasattr(config, 'moe') and config.moe:
        print(f"  moe: True")
        if hasattr(config, 'moe_num_experts'):
            print(f"  moe_num_experts: {config.moe_num_experts}")
        if hasattr(config, 'moe_num_experts_per_tok'):
            print(f"  moe_num_experts_per_tok: {config.moe_num_experts_per_tok}")
    
    # Initialize model exactly as in main.py
    print(f"\nInitializing model...")
    model = get_model(config).to(device)
    print(f"\nModel:\n{model}")
    
    # Load checkpoint exactly as in main.py (using load_checkpoint from optim.utils)
    print(f"\nLoading weights from: {ckpt_file}")
    from optim.utils import load_checkpoint
    # Create dummy optimizer and scheduler for load_checkpoint
    dummy_opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    dummy_scheduler = None
    if hasattr(config, 'scheduler') and config.scheduler != "none":
        dummy_scheduler = torch.optim.lr_scheduler.LambdaLR(dummy_opt, lambda step: 1.0)
    
    try:
        curr_iter = load_checkpoint(model, dummy_opt, dummy_scheduler, ckpt_file, device)
        print(f"✓ Weights loaded successfully! (iteration: {curr_iter})")
    except Exception as e:
        print(f"Warning: load_checkpoint failed: {e}")
        # Fallback to manual loading
        ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
        if isinstance(ckpt, dict):
            if "model" in ckpt:
                state_dict = ckpt["model"]
            else:
                state_dict = ckpt
        else:
            state_dict = ckpt
        
        # Strip prefixes from DDP or torch.compile (can have both: _orig_mod.module.)
        if state_dict:
            first_key = next(iter(state_dict.keys()))
            # Handle double prefix: _orig_mod.module.
            if first_key.startswith("_orig_mod.module."):
                print("Stripping '_orig_mod.module.' prefix (from torch.compile + DDP)...")
                state_dict = {k.replace("_orig_mod.module.", "", 1): v for k, v in state_dict.items()}
            elif first_key.startswith("_orig_mod."):
                print("Stripping '_orig_mod.' prefix (from torch.compile)...")
                state_dict = {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}
            # After stripping _orig_mod, check if there's still module. prefix
            if state_dict:
                first_key = next(iter(state_dict.keys()))
                if first_key.startswith("module."):
                    print("Stripping 'module.' prefix (from DDP)...")
                    state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        
        try:
            model.load_state_dict(state_dict, strict=True)
            print("✓ Weights loaded successfully!")
        except Exception as e2:
            print(f"Warning: Could not load with strict=True: {e2}")
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                print(f"Missing keys: {missing_keys[:5]}..." if len(missing_keys) > 5 else f"Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys: {unexpected_keys[:5]}..." if len(unexpected_keys) > 5 else f"Unexpected keys: {unexpected_keys}")
    
    # Set model to eval mode
    model.eval()
    
    # Calculate and print model size (after loading weights)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Get actual dtype from model parameters
    param_dtype = next(p.dtype for p in model.parameters())
    dtype_size = param_dtype.itemsize
    
    total_size_mb = total_params * dtype_size / (1024 * 1024)
    buffer_size_mb = sum(b.numel() * b.element_size() for b in model.buffers()) / (1024 * 1024)
    
    print(f"\nModel size:")
    print(f"  Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Parameter dtype: {param_dtype}")
    print(f"  Model size (parameters): {total_size_mb:.2f} MB")
    if buffer_size_mb > 0:
        print(f"  Buffer size: {buffer_size_mb:.2f} MB")
        print(f"  Total model size: {total_size_mb + buffer_size_mb:.2f} MB")
    else:
        print(f"  Total model size: {total_size_mb:.2f} MB")
    
    # Compute validation loss exactly as in main.py
    print(f"\nComputing validation loss...")
    try:
        # Set up validation datareader exactly as in main.py
        data_srcs = get_dataset(config)
        val_reader = DataReader(
            data_src=data_srcs["val"],
            local_batch_size=getattr(config, 'batch_size', 50),
            sequence_length=config.sequence_length,
            seed=getattr(config, 'data_seed', 1337),
            with_replacement=False,
            auto_shard=False,
            keep_in_ram=True,
        )
        
        # Reset validation reader to start from beginning (as in eval_and_log)
        val_reader.set_step(0)
        
        # Set up autocast context exactly as in main.py
        if "cuda" in device:
            dtype_map = {
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }
            model_dtype = getattr(config, 'dtype', 'bfloat16')
            type_ctx = torch.amp.autocast(
                device_type="cuda",
                dtype=dtype_map.get(model_dtype, torch.bfloat16),
            )
        else:
            type_ctx = nullcontext()
        
        # Ensure model is in eval mode
        model.eval()
        
        # Run evaluation exactly as in main.py
        val_acc, val_loss, val_perplexity, val_aux_losses, router_logits = eval(
            model=model,
            reader=val_reader,
            device=device,
            max_num_batches=getattr(config, 'eval_batches', 24),  # Use config value if available
            ctx=type_ctx,
            moe=getattr(config, 'moe', False),
            get_router_logits=False,
            cfg=config,
        )
        
        print(f"\nValidation metrics:")
        print(f"  Validation loss: {val_loss:.4f}")
        print(f"  Validation perplexity: {val_perplexity:.4f}")
        print(f"  Validation accuracy: {val_acc:.4f}")
        if val_aux_losses:
            for k, v in val_aux_losses.items():
                print(f"  {k}: {v:.4f}")
    except Exception as e:
        print(f"Warning: Could not compute validation loss: {e}")
        import traceback
        traceback.print_exc()
    
    return model


@torch.no_grad()
def stream_generate(model, prompt, max_new_tokens, temperature=1.0, top_k=None):
    """Generate tokens and stream them as they're produced."""
    idx = (
        torch.tensor(
            model.tokenizer.encode(prompt, allowed_special={"<|endoftext|>"})
        )
        .view(1, -1)
        .to(next(model.parameters()).device)
    )
    generated = idx.clone()
    
    for _ in range(max_new_tokens):
        idx_cond = (
            generated 
            if generated.size(1) <= model.config.sequence_length 
            else generated[:, -model.config.sequence_length:]
        )
        logits = model(idx_cond, get_logits=True)["logits"]
        logits = logits[:, -1, :] / temperature
        
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")
        
        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        generated = torch.cat((generated, idx_next), dim=1)
        
        # Decode and print the new token
        token_text = model.tokenizer.decode([idx_next.item()])
        print(token_text, end="", flush=True)
    
    print()  # Newline after generation
    return generated


def main():
    parser = argparse.ArgumentParser(description="Generate text from a checkpoint")
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        required=True,
        help="Path to checkpoint directory (exp dir) or checkpoint file"
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Alias for --checkpoint"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda:0, cpu, etc.). Auto-detected if not specified."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (default: 0.8)"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=200,
        help="Top-k sampling (default: 200, 0 = no top-k)"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=100,
        help="Maximum number of tokens to generate (default: 100)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Prompt to use (if not provided, enters interactive mode)"
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream tokens as they are generated"
    )
    
    args = parser.parse_args()
    
    # Auto-detect device
    if args.device is None:
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {args.device}\n")
    
    # Resolve checkpoint path
    checkpoint_path = args.resume_from if args.resume_from is not None else args.checkpoint
    
    # Load model
    try:
        model = load_model(checkpoint_path, args.device)
    except Exception as e:
        print(f"\nError loading model: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Single prompt mode
    if args.prompt is not None:
        print(f"\n{'='*60}")
        print(f"Prompt: {args.prompt}")
        print(f"Generating {args.max_tokens} tokens (temperature={args.temperature}, top_k={args.top_k})...")
        print(f"{'='*60}\n")
        
        if args.stream:
            stream_generate(
                model,
                args.prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=None if args.top_k == 0 else args.top_k,
            )
        else:
            completion = model.generate_from_string(
                args.prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=None if args.top_k == 0 else args.top_k,
            )
            print(completion)
        return
    
    # Interactive mode
    print(f"\n{'='*60}")
    print("Model loaded! Enter a prompt to generate text.")
    print(f"Settings: temperature={args.temperature}, top_k={args.top_k}, max_tokens={args.max_tokens}")
    print("Type 'quit' or 'exit' to stop.")
    print(f"{'='*60}\n")
    
    while True:
        try:
            prompt = input("Enter prompt: ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not prompt:
                print("Please enter a prompt.")
                continue
            
            print(f"\nGenerating {args.max_tokens} tokens...\n")
            
            if args.stream:
                stream_generate(
                    model,
                    prompt,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_k=None if args.top_k == 0 else args.top_k,
                )
            else:
                completion = model.generate_from_string(
                    prompt,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_k=None if args.top_k == 0 else args.top_k,
                )
                print(completion)
            
            print()  # Blank line between generations
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"Error during generation: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
