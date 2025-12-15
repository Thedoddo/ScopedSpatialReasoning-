# Copyright 2025 Jackson Comer
# Licensed under the Apache License, Version 2.0

"""
SSR-2.5 Paper Training Script
=============================

Clean training script for reproducing paper results.

Usage:
    python train.py --model ssr25 --data wikitext103.txt
    python train.py --model transformer --data wikitext103.txt
    python train.py --model all --data wikitext103.txt
"""

import argparse
import json
import os
import time
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Import models
from models.ssr2 import CharTokenizer, TextDataset, WarmupCosineScheduler
from models.ssr2_5 import SSR25, SSR25NoSlots
from models.ssr2_5_variants import (
    SSR25A, SSR25B, SSR25C, SSR25D, SSR25E, SSR25F, SSR25G, SSR25H
)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================================
# TRANSFORMER BASELINE
# ============================================================================

class TransformerBlock(torch.nn.Module):
    """Standard pre-norm Transformer decoder block."""
    
    def __init__(self, d_model, n_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.ln1 = torch.nn.LayerNorm(d_model)
        self.attn = torch.nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = torch.nn.LayerNorm(d_model)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(d_model, ffn_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(ffn_dim, d_model),
            torch.nn.Dropout(dropout)
        )
        
    def forward(self, x, mask=None):
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerBaseline(torch.nn.Module):
    """GPT-style decoder-only Transformer for fair comparison."""
    
    def __init__(self, vocab_size, d_model, n_heads, n_layers, ffn_dim, max_seq_len=1024, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        self.tok_emb = torch.nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(max_seq_len, d_model)
        self.drop = torch.nn.Dropout(dropout)
        
        self.blocks = torch.nn.ModuleList([
            TransformerBlock(d_model, n_heads, ffn_dim, dropout)
            for _ in range(n_layers)
        ])
        
        self.ln_f = torch.nn.LayerNorm(d_model)
        self.head = torch.nn.Linear(d_model, vocab_size)
        self.register_buffer("causal_mask", None)
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, torch.nn.Embedding):
                torch.nn.init.normal_(m.weight, std=0.02)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()
    
    def _get_causal_mask(self, seq_len, device):
        if self.causal_mask is None or self.causal_mask.size(0) < seq_len:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
            self.causal_mask = mask
        return self.causal_mask[:seq_len, :seq_len]
        
    def forward(self, x):
        B, T = x.shape
        device = x.device
        pos = torch.arange(T, device=device).unsqueeze(0)
        h = self.drop(self.tok_emb(x) + self.pos_emb(pos))
        mask = self._get_causal_mask(T, device)
        for block in self.blocks:
            h = block(h, mask)
        h = self.ln_f(h)
        return self.head(h)


# ============================================================================
# MODEL REGISTRY
# ============================================================================

MODELS = {
    'ssr25': SSR25,
    'ssr25_noslots': SSR25NoSlots,
    'transformer': TransformerBaseline,
    'ssr25a': SSR25A,
    'ssr25b': SSR25B,
    'ssr25c': SSR25C,
    'ssr25d': SSR25D,
    'ssr25e': SSR25E,
    'ssr25f': SSR25F,
    'ssr25g': SSR25G,
    'ssr25h': SSR25H,
}


def create_model(name, vocab_size, config):
    """Create a model by name with given config."""
    d_model = config['d_model']
    hidden_mlp = config['hidden_mlp']
    num_layers = config['num_layers']
    num_slots = config['num_slots']
    dropout = config['dropout']
    
    if name == 'transformer':
        n_heads = max(1, d_model // 64)
        return TransformerBaseline(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=num_layers,
            ffn_dim=hidden_mlp,
            dropout=dropout,
        )
    elif name == 'ssr25_noslots':
        return SSR25NoSlots(
            vocab_size=vocab_size,
            d_model=d_model,
            hidden_mlp=hidden_mlp,
            num_layers=num_layers,
            dropout=dropout,
        )
    elif name == 'ssr25':
        return SSR25(
            vocab_size=vocab_size,
            d_model=d_model,
            num_slots=num_slots,
            hidden_mlp=hidden_mlp,
            num_layers=num_layers,
            slot_dropout=dropout,
        )
    elif name in MODELS:
        # SSR-2.5 variants A-H
        return MODELS[name](
            vocab_size=vocab_size,
            d_model=d_model,
            num_slots=num_slots,
            hidden_mlp=hidden_mlp,
            num_layers=num_layers,
        )
    else:
        raise ValueError(f"Unknown model: {name}")


# ============================================================================
# TRAINING
# ============================================================================

def evaluate(model, val_loader, device):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            inp, tgt = batch[:, :-1], batch[:, 1:]
            
            # Handle SSR models that return tuples
            out = model(inp)
            if isinstance(out, tuple):
                logits = out[0]
            else:
                logits = out
            
            loss = F.cross_entropy(
                logits.reshape(-1, model.vocab_size),
                tgt.reshape(-1),
                ignore_index=0
            )
            total_loss += loss.item()
            num_batches += 1
            
            if num_batches >= 100:  # Limit eval batches for speed
                break
    
    model.train()
    return total_loss / max(1, num_batches)


def train_model(model, train_loader, val_loader, config, device):
    """Train a model and return results."""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['peak_lr'],
        weight_decay=0.01,
        betas=(0.9, 0.98)
    )
    
    total_steps = config['total_steps']
    scheduler = WarmupCosineScheduler(
        optimizer,
        config['peak_lr'],
        config['min_lr'],
        config['warmup_steps'],
        total_steps
    )
    
    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params:,}")
    
    model.train()
    data_iter = iter(train_loader)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    start_time = time.time()
    
    for step in range(total_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
        
        batch = batch.to(device)
        inp, tgt = batch[:, :-1], batch[:, 1:]
        
        optimizer.zero_grad()
        
        # Handle SSR models that return tuples
        out = model(inp)
        if isinstance(out, tuple):
            logits = out[0]
        else:
            logits = out
        
        loss = F.cross_entropy(
            logits.reshape(-1, model.vocab_size),
            tgt.reshape(-1),
            ignore_index=0
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        train_losses.append(loss.item())
        
        # Evaluate every 500 steps
        if (step + 1) % 500 == 0:
            val_loss = evaluate(model, val_loader, device)
            val_losses.append(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            
            elapsed = time.time() - start_time
            print(f"  Step {step+1}/{total_steps}: train={loss.item():.4f}, val={val_loss:.4f}, best={best_val_loss:.4f} ({elapsed:.1f}s)")
    
    train_time = time.time() - start_time
    final_val_loss = evaluate(model, val_loader, device)
    
    return {
        'params': params,
        'final_val_loss': final_val_loss,
        'best_val_loss': best_val_loss,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_time': train_time,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train SSR-2.5 models')
    parser.add_argument('--model', type=str, default='ssr25',
                        choices=list(MODELS.keys()) + ['all'],
                        help='Model to train (or "all" for all models)')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to training data (text file)')
    parser.add_argument('--d_model', type=int, default=256,
                        help='Model dimension')
    parser.add_argument('--hidden', type=int, default=512,
                        help='Hidden/FFN dimension')
    parser.add_argument('--layers', type=int, default=2,
                        help='Number of layers')
    parser.add_argument('--slots', type=int, default=24,
                        help='Number of slots (for SSR models)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--seq_len', type=int, default=128,
                        help='Sequence length')
    parser.add_argument('--steps', type=int, default=10000,
                        help='Total training steps')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='Peak learning rate')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory for results')
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.data}...")
    with open(args.data, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    print(f"Loaded {len(text):,} characters")
    
    # Split train/val (90/10)
    split_idx = int(len(text) * 0.9)
    train_text = text[:split_idx]
    val_text = text[split_idx:]
    print(f"Split: train={len(train_text):,}, val={len(val_text):,}")
    
    # Tokenize
    tokenizer = CharTokenizer()
    tokenizer.fit(text)
    print(f"Vocab size: {tokenizer.vocab_size}")
    
    # Create datasets
    train_dataset = TextDataset(train_text, tokenizer, seq_len=args.seq_len)
    val_dataset = TextDataset(val_text, tokenizer, seq_len=args.seq_len)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=0,
    )
    
    # Config
    config = {
        'd_model': args.d_model,
        'hidden_mlp': args.hidden,
        'num_layers': args.layers,
        'num_slots': args.slots,
        'dropout': 0.1,
        'peak_lr': args.lr,
        'min_lr': args.lr / 10,
        'warmup_steps': 200,
        'total_steps': args.steps,
    }
    
    # Determine which models to train
    if args.model == 'all':
        model_names = list(MODELS.keys())
    else:
        model_names = [args.model]
    
    # Train each model
    results = {}
    for name in model_names:
        print(f"\n{'='*50}")
        print(f"Training: {name}")
        print(f"{'='*50}")
        
        model = create_model(name, tokenizer.vocab_size, config)
        model = model.to(DEVICE)
        
        result = train_model(model, train_loader, val_loader, config, DEVICE)
        results[name] = result
        
        print(f"\n{name} complete: val_loss={result['final_val_loss']:.4f}, best={result['best_val_loss']:.4f}")
        
        # Free memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Save results
    os.makedirs(args.output, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(args.output, f'results_{timestamp}.json')
    
    # Convert for JSON serialization
    save_results = {}
    for name, result in results.items():
        save_results[name] = {
            'params': result['params'],
            'final_val_loss': result['final_val_loss'],
            'best_val_loss': result['best_val_loss'],
            'train_time': result['train_time'],
        }
    
    with open(output_path, 'w') as f:
        json.dump({
            'config': config,
            'data': args.data,
            'vocab_size': tokenizer.vocab_size,
            'results': save_results,
        }, f, indent=2)
    
    print(f"\n{'='*50}")
    print("FINAL RESULTS")
    print(f"{'='*50}")
    print(f"{'Model':<20} {'Params':>10} {'Val Loss':>10} {'Best':>10}")
    print("-" * 52)
    for name, result in sorted(results.items(), key=lambda x: x[1]['best_val_loss']):
        print(f"{name:<20} {result['params']:>10,} {result['final_val_loss']:>10.4f} {result['best_val_loss']:>10.4f}")
    
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
