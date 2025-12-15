# Copyright 2025 Jackson Comer
# Licensed under the Apache License, Version 2.0
# See LICENSE file in the project root for full license text.

"""
SSR-2: Scoped Spatial Reasoning v2
==================================

A slot-memory language model with:
- GRU-based recurrent slot states (memory over time)
- Bidirectional token↔slot interaction (write AND read)
- Context-aware routing (token + memory summary)
- Transformer-style residual + LayerNorm + MLP blocks
- Optional: multi-head routing, slot dropout, cross-slot attention

Key differences from SSR-1/NSLM:
- Slots have RECURRENT state via GRU (not just accumulation)
- Tokens READ from slots (feedback), not just write
- Routing sees both token AND memory context
- Proper residual + LayerNorm makes depth work

Architecture per layer:
1. Routing: MLP(concat(token, slot_mean)) -> softmax -> alpha[B, S]
2. Write: GRU(alpha * token, slot_state) -> new slot states  
3. Read: weighted sum of slots by alpha -> slot_context
4. FFN: LayerNorm + MLP + residual

Shapes:
- B = batch size
- T = sequence length (e.g., 256)
- D = d_model (e.g., 256)
- S = num_slots (e.g., 24-32)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Optional
import math


# ============================================================================
# TOKENIZER (same as NSLM)
# ============================================================================

class CharTokenizer:
    """Character-level tokenizer with PAD token."""
    
    def __init__(self):
        self.char_to_id = {'<PAD>': 0}
        self.id_to_char = {0: '<PAD>'}
        self.vocab_size = 1
        
        # Aliases for save/load compatibility
        self.char2idx = self.char_to_id
        self.idx2char = self.id_to_char
        
    def fit(self, text: str):
        """Build vocabulary from text."""
        for char in sorted(set(text)):
            if char not in self.char_to_id:
                self.char_to_id[char] = self.vocab_size
                self.id_to_char[self.vocab_size] = char
                self.vocab_size += 1
        # Update aliases
        self.char2idx = self.char_to_id
        self.idx2char = self.id_to_char
        print(f"Vocabulary size: {self.vocab_size} characters")
        
    def encode(self, text: str) -> List[int]:
        return [self.char_to_id.get(c, 0) for c in text]
    
    def decode(self, ids: List[int]) -> str:
        return ''.join(self.id_to_char.get(i, '?') for i in ids)


# ============================================================================
# DATASET (same as NSLM)
# ============================================================================

class TextDataset(Dataset):
    """Fixed-length sequence dataset from text corpus."""
    
    def __init__(self, text: str, tokenizer: CharTokenizer, seq_len: int):
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        self.num_sequences = (len(self.data) - 1) // seq_len
        print(f"Dataset: {len(self.data)} tokens, {self.num_sequences} sequences")
        
    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, idx):
        start = idx * self.seq_len
        return self.data[start:start + self.seq_len + 1]


# ============================================================================
# SSR-2 BLOCK
# ============================================================================

class SSR2Block(nn.Module):
    """
    Single SSR-2 processing block with:
    - Context-aware routing (token + memory)
    - GRU-based recurrent slot update
    - Slot→token feedback (read)
    - Transformer-style FFN + residual + LayerNorm
    
    This is the heart of SSR-2.
    """
    
    def __init__(
        self, 
        d_model: int, 
        num_slots: int, 
        hidden_mlp: int,
        slot_dropout: float = 0.0,
        tau: float = 1.0
    ):
        super().__init__()
        self.d_model = d_model
        self.num_slots = num_slots
        self.hidden_mlp = hidden_mlp
        self.slot_dropout = slot_dropout
        self.tau = tau  # Routing temperature (can be annealed)
        
        # ===== Routing MLP: concat(token, slot_mean) -> alpha =====
        # Input: [B, 2*D], Output: [B, S]
        self.route_mlp = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_slots)
        )
        
        # ===== GRU for slot recurrence =====
        # Shared across all slots (efficient)
        # Input: alpha_i * token (routed input)
        # Hidden: slot state
        self.slot_gru = nn.GRUCell(d_model, d_model)
        
        # ===== Slot→Token projection (for feedback) =====
        self.slot_proj = nn.Linear(d_model, d_model)
        
        # ===== FFN block (Transformer-style) =====
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_mlp),
            nn.GELU(),
            nn.Linear(hidden_mlp, d_model)
        )
        self.ln2 = nn.LayerNorm(d_model)
        
        # ===== Optional: Cross-slot attention =====
        # (disabled by default, enable after MVP works)
        self.use_cross_slot = False
        if self.use_cross_slot:
            self.cross_slot_attn = nn.MultiheadAttention(
                d_model, num_heads=2, batch_first=True, dropout=0.1
            )
        
    def forward(
        self, 
        x: torch.Tensor, 
        S: torch.Tensor,
        training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process one token through this SSR-2 block.
        
        Args:
            x: [B, D] token representation (from embedding or prev layer)
            S: [B, S, D] current slot states for this layer
            training: whether in training mode (for slot dropout)
            
        Returns:
            x_out: [B, D] output token representation
            S_new: [B, S, D] updated slot states
            alpha: [B, S] routing weights (for visualization)
        """
        B, D = x.shape
        num_slots = S.shape[1]
        
        # ===== 2.1 Context-aware routing =====
        # Combine token + memory summary for routing decision
        slot_mean = S.mean(dim=1)  # [B, D] - simple memory summary
        ctx = torch.cat([x, slot_mean], dim=-1)  # [B, 2D]
        
        r_logits = self.route_mlp(ctx)  # [B, S]
        alpha = F.softmax(r_logits / self.tau, dim=-1)  # [B, S]
        
        # ===== 2.3 GRU slot update (write) =====
        # Each slot gets routed input: alpha_i * token
        x_expanded = x.unsqueeze(1).expand(-1, num_slots, -1)  # [B, S, D]
        U = alpha.unsqueeze(-1) * x_expanded  # [B, S, D] - routed inputs
        
        # Flatten for GRU: treat each slot as separate sequence
        U_flat = U.reshape(B * num_slots, D)  # [B*S, D]
        S_flat = S.reshape(B * num_slots, D)  # [B*S, D]
        
        S_new_flat = self.slot_gru(U_flat, S_flat)  # [B*S, D]
        S_new = S_new_flat.reshape(B, num_slots, D)  # [B, S, D]
        
        # ===== 2.4 Optional cross-slot communication =====
        if self.use_cross_slot:
            S_comm, _ = self.cross_slot_attn(S_new, S_new, S_new)
            S_new = S_new + 0.1 * S_comm  # Small residual mix
        
        # ===== 2.5 Slot dropout (regularization) =====
        if training and self.slot_dropout > 0:
            mask = torch.bernoulli(
                torch.ones(B, num_slots, 1, device=S.device) * (1 - self.slot_dropout)
            )
            S_masked = S_new * mask
        else:
            S_masked = S_new
        
        # ===== 2.6 Read from slots (feedback) =====
        # Reuse alpha for reading (could learn separate beta)
        slot_context = (alpha.unsqueeze(-1) * S_masked).sum(dim=1)  # [B, D]
        slot_update = self.slot_proj(slot_context)  # [B, D]
        
        # Residual: token + memory feedback
        x_mid = x + slot_update  # [B, D]
        
        # ===== 2.7 FFN block with residual + LayerNorm =====
        h = self.ln1(x_mid)
        ff = self.ffn(h)
        x_out = self.ln2(x_mid + ff)  # [B, D]
        
        return x_out, S_new, alpha


# ============================================================================
# SSR-2 MODEL
# ============================================================================

class SSR2(nn.Module):
    """
    SSR-2: Scoped Spatial Reasoning v2
    
    A slot-memory language model where:
    - Slots are GRU cells with persistent state over time
    - Tokens write to AND read from slots
    - Routing is context-aware (sees token + memory)
    - Depth works via proper residual + LayerNorm
    
    Target: Beat 3.3 loss wall on Tiny Shakespeare!
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_slots: int = 24,
        hidden_mlp: int = 512,
        num_layers: int = 2,
        slot_dropout: float = 0.0,
        tau_init: float = 1.0
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_slots = num_slots
        self.hidden_mlp = hidden_mlp
        self.num_layers = num_layers
        self.tau = tau_init  # Can be annealed during training
        
        # ===== Token embedding + positional =====
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(1024, d_model)  # Max seq len 1024
        
        # ===== Stack of SSR-2 blocks =====
        self.blocks = nn.ModuleList([
            SSR2Block(
                d_model=d_model,
                num_slots=num_slots,
                hidden_mlp=hidden_mlp,
                slot_dropout=slot_dropout,
                tau=tau_init
            )
            for _ in range(num_layers)
        ])
        
        # ===== Output head =====
        self.ln_out = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with small values for stability."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.GRUCell):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param, gain=0.5)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
                        
    def set_tau(self, tau: float):
        """Set routing temperature for all blocks (for annealing)."""
        self.tau = tau
        for block in self.blocks:
            block.tau = tau
            
    def step(
        self, 
        x_t: torch.Tensor, 
        pos: int,
        S_list: List[torch.Tensor],
        training: bool = True
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """
        Process a single time step through all layers.
        
        Args:
            x_t: [B] token IDs at time t
            pos: position index (for pos embedding)
            S_list: list of [B, S, D] slot states per layer
            training: whether in training mode
            
        Returns:
            logits: [B, vocab_size]
            S_list_new: updated slot states
            alpha: [B, S] routing weights from last layer (for viz)
        """
        B = x_t.shape[0]
        device = x_t.device
        
        # Embed token + position
        e_t = self.embedding(x_t)  # [B, D]
        p_t = self.pos_embedding(torch.tensor([pos], device=device))  # [1, D]
        x = e_t + p_t  # [B, D]
        
        # Process through each layer
        S_list_new = []
        alpha_last = None
        
        for i, block in enumerate(self.blocks):
            x, S_new, alpha = block(x, S_list[i], training=training)
            S_list_new.append(S_new)
            alpha_last = alpha
        
        # Output head
        logits = self.head(self.ln_out(x))  # [B, vocab_size]
        
        return logits, S_list_new, alpha_last
    
    def forward(
        self, 
        x: torch.Tensor, 
        return_slots: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        """
        Full forward pass over a sequence.
        
        Args:
            x: [B, T] input token IDs
            return_slots: if True, also return final slots and alphas
            
        Returns:
            logits: [B, T, vocab_size]
            (optional) slots: [B, S, D] from last layer
            (optional) alphas: [B, T, S] routing weights
        """
        B, T = x.shape
        device = x.device
        
        # Initialize slots for each layer
        S_list = [
            torch.zeros(B, self.num_slots, self.d_model, device=device)
            for _ in range(self.num_layers)
        ]
        
        logits_list = []
        alpha_list = []
        
        # Process sequence step by step
        for t in range(T):
            logits_t, S_list, alpha = self.step(
                x[:, t], t, S_list, training=self.training
            )
            logits_list.append(logits_t)
            alpha_list.append(alpha)
            
        logits = torch.stack(logits_list, dim=1)  # [B, T, vocab_size]
        
        if return_slots:
            alphas = torch.stack(alpha_list, dim=1)  # [B, T, S]
            return logits, S_list[-1], alphas
        return logits
    
    @torch.no_grad()
    def generate(
        self,
        tokenizer: CharTokenizer,
        start_text: str = "",
        max_len: int = 200,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        device: str = 'cpu'
    ) -> str:
        """Generate text autoregressively."""
        self.eval()
        
        # Encode start text
        if start_text:
            tokens = tokenizer.encode(start_text)
        else:
            tokens = [tokenizer.char_to_id.get(' ', 1)]
            
        # Initialize slots for each layer
        S_list = [
            torch.zeros(1, self.num_slots, self.d_model, device=device)
            for _ in range(self.num_layers)
        ]
        
        # Feed start tokens through model
        for pos, tok in enumerate(tokens[:-1]):
            x_t = torch.tensor([tok], device=device)
            _, S_list, _ = self.step(x_t, pos, S_list, training=False)
            
        # Generate new tokens
        generated = list(tokens)
        pos = len(tokens) - 1
        
        for _ in range(max_len):
            x_t = torch.tensor([generated[-1]], device=device)
            logits_t, S_list, _ = self.step(x_t, pos, S_list, training=False)
            pos += 1
            
            # Apply temperature
            logits_t = logits_t / temperature
            
            # Optional top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits_t, min(top_k, logits_t.size(-1)))
                logits_t[logits_t < v[:, [-1]]] = -float('inf')
                
            # Sample
            probs = F.softmax(logits_t, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_token)
            
            # Stop on newline after generating enough
            if len(generated) > len(tokens) + 50:
                char = tokenizer.id_to_char.get(next_token, '')
                if char == '\n':
                    break
                    
        return tokenizer.decode(generated)


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

class WarmupCosineScheduler:
    """
    Linear warmup + Cosine decay scheduler.
    
    - Linear warmup from 0 to peak_lr over warmup_steps
    - Cosine decay from peak_lr to min_lr over remaining steps
    """
    
    def __init__(self, optimizer, peak_lr, min_lr, warmup_steps, total_steps):
        self.optimizer = optimizer
        self.peak_lr = peak_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.current_step = 0
        self._update_lr()
        
    def _get_lr(self):
        if self.current_step < self.warmup_steps:
            return self.peak_lr * (self.current_step / max(1, self.warmup_steps))
        else:
            decay_steps = self.total_steps - self.warmup_steps
            progress = (self.current_step - self.warmup_steps) / max(1, decay_steps)
            progress = min(1.0, progress)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return self.min_lr + (self.peak_lr - self.min_lr) * cosine_decay
    
    def _update_lr(self):
        lr = self._get_lr()
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
            
    def step(self):
        self.current_step += 1
        self._update_lr()
        
    def get_last_lr(self):
        return [self._get_lr()]


class TemperatureScheduler:
    """
    Anneal routing temperature from tau_init to tau_final.
    
    - Start with tau=1.0 (fuzzy exploration)
    - End with tau=0.5-0.7 (sharper specialization)
    """
    
    def __init__(self, model: SSR2, tau_init: float = 1.0, tau_final: float = 0.5, total_epochs: int = 50):
        self.model = model
        self.tau_init = tau_init
        self.tau_final = tau_final
        self.total_epochs = total_epochs
        
    def step(self, epoch: int):
        """Update tau based on current epoch."""
        progress = epoch / max(1, self.total_epochs)
        tau = self.tau_init - (self.tau_init - self.tau_final) * progress
        tau = max(self.tau_final, tau)
        self.model.set_tau(tau)
        return tau


# ============================================================================
# QUICK TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SSR-2 Architecture Test")
    print("=" * 60)
    
    # Create model
    model = SSR2(
        vocab_size=66,  # Tiny Shakespeare
        d_model=256,
        num_slots=24,
        hidden_mlp=512,
        num_layers=2,
        slot_dropout=0.1
    )
    
    # Count parameters
    params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {params:,}")
    
    # Test shapes
    B, T = 4, 32
    x = torch.randint(0, 66, (B, T))
    
    print(f"\nInput shape: {x.shape}")
    
    # Forward pass
    logits = model(x)
    print(f"Output shape: {logits.shape}")
    
    # With slots
    logits, slots, alphas = model(x, return_slots=True)
    print(f"Slots shape: {slots.shape}")
    print(f"Alphas shape: {alphas.shape}")
    
    # Check routing distribution
    print(f"\nRouting alpha stats:")
    print(f"  Mean: {alphas.mean():.4f}")
    print(f"  Std: {alphas.std():.4f}")
    print(f"  Max: {alphas.max():.4f}")
    print(f"  Min: {alphas.min():.4f}")
    
    print("\n✓ SSR-2 architecture test passed!")
