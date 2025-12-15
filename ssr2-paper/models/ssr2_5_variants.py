# Copyright 2025 Jackson Comer
# Licensed under the Apache License, Version 2.0

"""
SSR-2.5 Variants (A-H): Systematic Architecture Ablations
=========================================================

Each variant tests ONE architectural change from base SSR-2.5:

A - Strong Residuals: Proper x + block(x) residual connections
B - Pre-LayerNorm: LayerNorm before block (GPT-2 style)  
C - Learned Slot Init: Better initialization for slot memory
D - Gate Bias: Initialize gate to favor slots (0.7)
E - Wide Slots: 2x slot dimension with projection
F - Multi-Head Slots: Slots split into heads (like attention)
G - Slot Communication: Mini self-attention among slots
H - QK Routing: Query-key style routing instead of MLP

All variants maintain same param budget for fair comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional

# Import base components
try:
    from .ssr2_5 import DensePath, SlotPath
    from .ssr2 import CharTokenizer, TextDataset, WarmupCosineScheduler
except ImportError:
    from ssr2_5 import DensePath, SlotPath
    from ssr2 import CharTokenizer, TextDataset, WarmupCosineScheduler


# ============================================================================
# VARIANT A: Strong Residuals
# ============================================================================

class SSR25A_Block(nn.Module):
    """SSR-2.5A: Proper residual connections like Transformer."""
    
    def __init__(self, d_model: int, num_slots: int, hidden_mlp: int, 
                 top_k: int = 8, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dense = DensePath(d_model, hidden_mlp, dropout)
        self.slot = SlotPath(d_model, num_slots, hidden_mlp, top_k)
        
        # Learned gate
        self.gate_proj = nn.Linear(d_model, 1)
        nn.init.zeros_(self.gate_proj.bias)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        B, T, D = x.shape
        
        # First residual: slot path
        normed = self.norm1(x)
        slot_out = self.slot.forward_parallel(normed, training)
        x = x + self.dropout(slot_out)  # RESIDUAL
        
        # Second residual: dense path with gating
        normed = self.norm2(x)
        dense_out = self.dense(normed.reshape(B*T, D)).reshape(B, T, D)
        gate = torch.sigmoid(self.gate_proj(normed))
        
        # Gated combination with residual
        x = x + self.dropout(gate * slot_out + (1 - gate) * dense_out)  # RESIDUAL
        
        return x


class SSR25A(nn.Module):
    """SSR-2.5A: Strong Residual Connections"""
    
    def __init__(self, vocab_size: int, d_model: int = 256, num_slots: int = 24,
                 hidden_mlp: int = 512, num_layers: int = 2, top_k: int = 8,
                 dropout: float = 0.1, **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(2048, d_model)
        self.drop = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList([
            SSR25A_Block(d_model, num_slots, hidden_mlp, top_k, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.embed.weight  # Weight tying
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
    
    def forward(self, x: torch.Tensor, targets: torch.Tensor = None):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        
        h = self.drop(self.embed(x) + self.pos_embed(pos))
        
        for block in self.blocks:
            h = block(h, self.training)
        
        h = self.norm_f(h)
        logits = self.head(h)
        
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
        return logits


# ============================================================================
# VARIANT B: Pre-LayerNorm (GPT-2 Style)
# ============================================================================

class SSR25B_Block(nn.Module):
    """SSR-2.5B: Pre-LayerNorm like GPT-2 for stable deep training."""
    
    def __init__(self, d_model: int, num_slots: int, hidden_mlp: int,
                 top_k: int = 8, dropout: float = 0.1):
        super().__init__()
        # Pre-norm: normalize BEFORE the block
        self.norm_slot = nn.LayerNorm(d_model)
        self.norm_dense = nn.LayerNorm(d_model)
        
        self.dense = DensePath(d_model, hidden_mlp, dropout)
        self.slot = SlotPath(d_model, num_slots, hidden_mlp, top_k)
        
        self.gate_proj = nn.Linear(d_model, 1)
        nn.init.zeros_(self.gate_proj.bias)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        B, T, D = x.shape
        
        # Pre-norm slot path with residual
        slot_out = self.slot.forward_parallel(self.norm_slot(x), training)
        x = x + self.dropout(slot_out)
        
        # Pre-norm dense path with gating and residual
        normed = self.norm_dense(x)
        dense_out = self.dense(normed.reshape(B*T, D)).reshape(B, T, D)
        gate = torch.sigmoid(self.gate_proj(normed))
        
        gated = gate * slot_out + (1 - gate) * dense_out
        x = x + self.dropout(gated)
        
        return x


class SSR25B(nn.Module):
    """SSR-2.5B: Pre-LayerNorm (GPT-2 Style)"""
    
    def __init__(self, vocab_size: int, d_model: int = 256, num_slots: int = 24,
                 hidden_mlp: int = 512, num_layers: int = 2, top_k: int = 8,
                 dropout: float = 0.1, **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(2048, d_model)
        self.drop = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList([
            SSR25B_Block(d_model, num_slots, hidden_mlp, top_k, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.embed.weight
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
    
    def forward(self, x: torch.Tensor, targets: torch.Tensor = None):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        
        h = self.drop(self.embed(x) + self.pos_embed(pos))
        
        for block in self.blocks:
            h = block(h, self.training)
        
        h = self.norm_f(h)
        logits = self.head(h)
        
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
        return logits


# ============================================================================
# VARIANT C: Learned Slot Initialization  
# ============================================================================

class SlotPathC(nn.Module):
    """SlotPath with better learned initialization."""
    
    def __init__(self, d_model: int, num_slots: int, hidden_mlp: int,
                 top_k: int = 8, slot_dropout: float = 0.0):
        super().__init__()
        
        self.d_model = d_model
        self.num_slots = num_slots
        self.top_k = min(top_k, num_slots)
        
        # IMPROVED: Orthogonal initialization for slots
        slot_init = torch.zeros(num_slots, d_model)
        nn.init.orthogonal_(slot_init)
        self.slot_init = nn.Parameter(slot_init * 0.1)
        
        # Learnable scale per slot
        self.slot_scale = nn.Parameter(torch.ones(num_slots, 1))
        
        self.route_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, num_slots)
        )
        
        self.slot_gru = nn.GRUCell(d_model, d_model)
        self.slot_value_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.tau = nn.Parameter(torch.ones(1))
        self.slot_dropout = slot_dropout
    
    def forward_parallel(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        B, T, D = x.shape
        device = x.device
        
        # Use learned scale
        S = (self.slot_init * self.slot_scale).unsqueeze(0).expand(B, -1, -1).clone().to(x.dtype)
        
        slot_mean = S.mean(dim=1, keepdim=True).expand(-1, T, -1)
        ctx = torch.cat([x, slot_mean], dim=-1)
        
        route_logits = self.route_mlp(ctx) / (self.tau.abs() + 0.1)
        topk_values, topk_indices = torch.topk(route_logits, self.top_k, dim=-1)
        topk_weights = F.softmax(topk_values, dim=-1)
        
        alpha = torch.zeros(B, T, self.num_slots, device=device, dtype=x.dtype)
        alpha.scatter_(2, topk_indices, topk_weights)
        
        slot_input = torch.einsum('bts,btd->bsd', alpha, x)
        slot_weights_sum = alpha.sum(dim=1) + 1e-8
        slot_input_norm = slot_input / slot_weights_sum.unsqueeze(-1)
        
        slot_input_flat = slot_input_norm.reshape(B * self.num_slots, D)
        S_flat = S.reshape(B * self.num_slots, D)
        S_new_flat = self.slot_gru(slot_input_flat, S_flat)
        S_new = S_new_flat.reshape(B, self.num_slots, D)
        
        slot_values = self.slot_value_proj(S_new)
        output = torch.einsum('bts,bsd->btd', alpha, slot_values)
        output = self.out_proj(output)
        
        if self.slot_dropout > 0 and training:
            output = F.dropout(output, p=self.slot_dropout, training=training)
        
        return output


class SSR25C_Block(nn.Module):
    """SSR-2.5C: Better slot initialization."""
    
    def __init__(self, d_model: int, num_slots: int, hidden_mlp: int,
                 top_k: int = 8, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dense = DensePath(d_model, hidden_mlp, dropout)
        self.slot = SlotPathC(d_model, num_slots, hidden_mlp, top_k)
        
        self.gate_proj = nn.Linear(d_model, 1)
        nn.init.zeros_(self.gate_proj.bias)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        B, T, D = x.shape
        normed = self.norm(x)
        
        dense_out = self.dense(normed.reshape(B*T, D)).reshape(B, T, D)
        slot_out = self.slot.forward_parallel(normed, training)
        
        gate = torch.sigmoid(self.gate_proj(normed))
        combined = gate * slot_out + (1 - gate) * dense_out
        
        return x + self.dropout(combined)


class SSR25C(nn.Module):
    """SSR-2.5C: Learned Slot Initialization"""
    
    def __init__(self, vocab_size: int, d_model: int = 256, num_slots: int = 24,
                 hidden_mlp: int = 512, num_layers: int = 2, top_k: int = 8,
                 dropout: float = 0.1, **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(2048, d_model)
        self.drop = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList([
            SSR25C_Block(d_model, num_slots, hidden_mlp, top_k, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.embed.weight
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
    
    def forward(self, x: torch.Tensor, targets: torch.Tensor = None):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        
        h = self.drop(self.embed(x) + self.pos_embed(pos))
        
        for block in self.blocks:
            h = block(h, self.training)
        
        h = self.norm_f(h)
        logits = self.head(h)
        
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
        return logits


# ============================================================================
# VARIANT D: Gate Bias (Favor Slots)
# ============================================================================

class SSR25D_Block(nn.Module):
    """SSR-2.5D: Gate initialized to favor slots (0.7)."""
    
    def __init__(self, d_model: int, num_slots: int, hidden_mlp: int,
                 top_k: int = 8, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dense = DensePath(d_model, hidden_mlp, dropout)
        self.slot = SlotPath(d_model, num_slots, hidden_mlp, top_k)
        
        self.gate_proj = nn.Linear(d_model, 1)
        # BIAS: Initialize to favor slots (sigmoid(0.85) ≈ 0.7)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, 0.85)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        B, T, D = x.shape
        normed = self.norm(x)
        
        dense_out = self.dense(normed.reshape(B*T, D)).reshape(B, T, D)
        slot_out = self.slot.forward_parallel(normed, training)
        
        gate = torch.sigmoid(self.gate_proj(normed))
        combined = gate * slot_out + (1 - gate) * dense_out
        
        return x + self.dropout(combined)


class SSR25D(nn.Module):
    """SSR-2.5D: Gate Bias (Favor Slots)"""
    
    def __init__(self, vocab_size: int, d_model: int = 256, num_slots: int = 24,
                 hidden_mlp: int = 512, num_layers: int = 2, top_k: int = 8,
                 dropout: float = 0.1, **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(2048, d_model)
        self.drop = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList([
            SSR25D_Block(d_model, num_slots, hidden_mlp, top_k, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.embed.weight
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m != self.blocks[0].gate_proj:  # Don't overwrite gate init
                    nn.init.normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
    
    def forward(self, x: torch.Tensor, targets: torch.Tensor = None):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        
        h = self.drop(self.embed(x) + self.pos_embed(pos))
        
        for block in self.blocks:
            h = block(h, self.training)
        
        h = self.norm_f(h)
        logits = self.head(h)
        
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
        return logits


# ============================================================================
# VARIANT E: Wide Slots (2x dimension)
# ============================================================================

class SlotPathE(nn.Module):
    """SlotPath with wider slots (1.25x dim) and projection - balanced for similar params."""
    
    def __init__(self, d_model: int, num_slots: int, hidden_mlp: int,
                 top_k: int = 8, slot_dropout: float = 0.0):
        super().__init__()
        
        self.d_model = d_model
        self.slot_dim = int(d_model * 1.125)  # Slightly wider slots
        self.num_slots = num_slots
        self.top_k = min(top_k, num_slots)
        
        self.slot_init = nn.Parameter(torch.randn(num_slots, self.slot_dim) * 0.02)
        
        # Project input to slot dim
        self.input_proj = nn.Linear(d_model, self.slot_dim)
        
        self.route_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model // 2),  # Use d_model*2 for routing context
            nn.GELU(),
            nn.Linear(d_model // 2, num_slots)
        )
        
        self.slot_gru = nn.GRUCell(self.slot_dim, self.slot_dim)
        self.slot_value_proj = nn.Linear(self.slot_dim, d_model)  # Project back
        self.out_proj = nn.Linear(d_model, d_model)
        self.tau = nn.Parameter(torch.ones(1))
        self.slot_dropout = slot_dropout
    
    def forward_parallel(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        B, T, D = x.shape
        device = x.device
        
        S = self.slot_init.unsqueeze(0).expand(B, -1, -1).clone().to(x.dtype)
        
        # Project slot mean to d_model for routing context
        slot_mean_full = S.mean(dim=1, keepdim=True)  # [B, 1, slot_dim]
        slot_mean = slot_mean_full[:, :, :D].expand(-1, T, -1)  # Use first D dims
        ctx = torch.cat([x, slot_mean], dim=-1)  # [B, T, d_model*2]
        
        route_logits = self.route_mlp(ctx) / (self.tau.abs() + 0.1)
        topk_values, topk_indices = torch.topk(route_logits, self.top_k, dim=-1)
        topk_weights = F.softmax(topk_values, dim=-1)
        
        alpha = torch.zeros(B, T, self.num_slots, device=device, dtype=x.dtype)
        alpha.scatter_(2, topk_indices, topk_weights)
        
        # Project input to slot dim
        x_proj = self.input_proj(x)  # [B, T, slot_dim]
        
        slot_input = torch.einsum('bts,btd->bsd', alpha, x_proj)
        slot_weights_sum = alpha.sum(dim=1) + 1e-8
        slot_input_norm = slot_input / slot_weights_sum.unsqueeze(-1)
        
        slot_input_flat = slot_input_norm.reshape(B * self.num_slots, self.slot_dim)
        S_flat = S.reshape(B * self.num_slots, self.slot_dim)
        S_new_flat = self.slot_gru(slot_input_flat, S_flat)
        S_new = S_new_flat.reshape(B, self.num_slots, self.slot_dim)
        
        slot_values = self.slot_value_proj(S_new)  # Back to d_model
        output = torch.einsum('bts,bsd->btd', alpha, slot_values)
        output = self.out_proj(output)
        
        if self.slot_dropout > 0 and training:
            output = F.dropout(output, p=self.slot_dropout, training=training)
        
        return output


class SSR25E_Block(nn.Module):
    """SSR-2.5E: Wide slots."""
    
    def __init__(self, d_model: int, num_slots: int, hidden_mlp: int,
                 top_k: int = 8, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dense = DensePath(d_model, hidden_mlp, dropout)
        self.slot = SlotPathE(d_model, num_slots, hidden_mlp, top_k)
        
        self.gate_proj = nn.Linear(d_model, 1)
        nn.init.zeros_(self.gate_proj.bias)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        B, T, D = x.shape
        normed = self.norm(x)
        
        dense_out = self.dense(normed.reshape(B*T, D)).reshape(B, T, D)
        slot_out = self.slot.forward_parallel(normed, training)
        
        gate = torch.sigmoid(self.gate_proj(normed))
        combined = gate * slot_out + (1 - gate) * dense_out
        
        return x + self.dropout(combined)


class SSR25E(nn.Module):
    """SSR-2.5E: Wide Slots (2x dimension)"""
    
    def __init__(self, vocab_size: int, d_model: int = 256, num_slots: int = 24,
                 hidden_mlp: int = 512, num_layers: int = 2, top_k: int = 8,
                 dropout: float = 0.1, **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(2048, d_model)
        self.drop = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList([
            SSR25E_Block(d_model, num_slots, hidden_mlp, top_k, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.embed.weight
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
    
    def forward(self, x: torch.Tensor, targets: torch.Tensor = None):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        
        h = self.drop(self.embed(x) + self.pos_embed(pos))
        
        for block in self.blocks:
            h = block(h, self.training)
        
        h = self.norm_f(h)
        logits = self.head(h)
        
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
        return logits


# ============================================================================
# VARIANT F: Multi-Head Slots
# ============================================================================

class SlotPathF(nn.Module):
    """SlotPath with multi-head slots (like multi-head attention) - with full routing."""
    
    def __init__(self, d_model: int, num_slots: int, hidden_mlp: int,
                 top_k: int = 8, n_heads: int = 4, slot_dropout: float = 0.0):
        super().__init__()
        
        self.d_model = d_model
        self.num_slots = num_slots
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.slots_per_head = num_slots // n_heads
        self.top_k = min(max(2, top_k // n_heads), self.slots_per_head)
        
        # Separate slot init per head
        self.slot_init = nn.Parameter(torch.randn(n_heads, self.slots_per_head, self.head_dim) * 0.02)
        
        # FULL routing MLP (larger to match base SSR-2.5 params)
        self.route_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model),  # Bigger hidden
            nn.GELU(),
            nn.Linear(d_model, num_slots)  # Route to all slots, then split
        )
        
        # Per-head GRU  
        self.slot_gru = nn.GRUCell(self.head_dim, self.head_dim)
        # Add hidden projection per head to increase params (like FFN)
        self.slot_hidden_proj = nn.Linear(self.head_dim, self.head_dim * 4)  # Bigger expansion
        self.slot_out_head = nn.Linear(self.head_dim * 4, self.head_dim)
        # Value projection to add params
        self.slot_value_proj = nn.Linear(d_model, hidden_mlp)  # Expand to hidden_mlp
        self.slot_value_out = nn.Linear(hidden_mlp, d_model)   # Back down
        self.out_proj = nn.Linear(d_model, d_model)
        self.tau = nn.Parameter(torch.ones(1))
        self.slot_dropout = slot_dropout
    
    def forward_parallel(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        B, T, D = x.shape
        device = x.device
        
        # Full slot state [B, num_slots, head_dim] reshaped from [n_heads, slots_per_head, head_dim]
        S_full = self.slot_init.reshape(self.num_slots, self.head_dim)
        S = S_full.unsqueeze(0).expand(B, -1, -1).clone().to(x.dtype)  # [B, num_slots, head_dim]
        
        # Full routing using all slots
        slot_mean = S.mean(dim=1, keepdim=True)  # [B, 1, head_dim]
        # Expand to d_model by repeating
        slot_mean_full = slot_mean.repeat(1, 1, self.n_heads)  # [B, 1, d_model]
        slot_mean_exp = slot_mean_full.expand(-1, T, -1)  # [B, T, d_model]
        ctx = torch.cat([x, slot_mean_exp], dim=-1)  # [B, T, d_model*2]
        
        route_logits = self.route_mlp(ctx) / (self.tau.abs() + 0.1)  # [B, T, num_slots]
        topk_values, topk_indices = torch.topk(route_logits, self.top_k * self.n_heads, dim=-1)
        topk_weights = F.softmax(topk_values, dim=-1)
        
        alpha = torch.zeros(B, T, self.num_slots, device=device, dtype=x.dtype)
        alpha.scatter_(2, topk_indices, topk_weights)
        
        # Reshape x to heads for slot update
        x_heads = x.reshape(B, T, self.n_heads, self.head_dim)
        
        outputs = []
        for h in range(self.n_heads):
            # Get alpha for this head's slots
            start_slot = h * self.slots_per_head
            end_slot = start_slot + self.slots_per_head
            alpha_h = alpha[:, :, start_slot:end_slot]  # [B, T, slots_per_head]
            
            x_h = x_heads[:, :, h, :]  # [B, T, head_dim]
            S_h = S[:, start_slot:end_slot, :]  # [B, slots_per_head, head_dim]
            
            slot_input = torch.einsum('bts,btd->bsd', alpha_h, x_h)
            slot_weights_sum = alpha_h.sum(dim=1) + 1e-8
            slot_input_norm = slot_input / slot_weights_sum.unsqueeze(-1)
            
            slot_input_flat = slot_input_norm.reshape(B * self.slots_per_head, self.head_dim)
            S_flat = S_h.reshape(B * self.slots_per_head, self.head_dim)
            S_new_flat = self.slot_gru(slot_input_flat, S_flat)
            
            # Add hidden projection (like FFN) for more expressiveness
            S_new_flat = self.slot_out_head(F.gelu(self.slot_hidden_proj(S_new_flat)))
            S_new = S_new_flat.reshape(B, self.slots_per_head, self.head_dim)
            
            output_h = torch.einsum('bts,bsd->btd', alpha_h, S_new)
            outputs.append(output_h)
        
        # Concat heads
        output = torch.cat(outputs, dim=-1)  # [B, T, d_model]
        output = self.slot_value_out(F.gelu(self.slot_value_proj(output)))
        output = self.out_proj(output)
        
        if self.slot_dropout > 0 and training:
            output = F.dropout(output, p=self.slot_dropout, training=training)
        
        return output


class SSR25F_Block(nn.Module):
    """SSR-2.5F: Multi-head slots."""
    
    def __init__(self, d_model: int, num_slots: int, hidden_mlp: int,
                 top_k: int = 8, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dense = DensePath(d_model, hidden_mlp, dropout)
        self.slot = SlotPathF(d_model, num_slots, hidden_mlp, top_k, n_heads=4)
        
        self.gate_proj = nn.Linear(d_model, 1)
        nn.init.zeros_(self.gate_proj.bias)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        B, T, D = x.shape
        normed = self.norm(x)
        
        dense_out = self.dense(normed.reshape(B*T, D)).reshape(B, T, D)
        slot_out = self.slot.forward_parallel(normed, training)
        
        gate = torch.sigmoid(self.gate_proj(normed))
        combined = gate * slot_out + (1 - gate) * dense_out
        
        return x + self.dropout(combined)


class SSR25F(nn.Module):
    """SSR-2.5F: Multi-Head Slots"""
    
    def __init__(self, vocab_size: int, d_model: int = 256, num_slots: int = 24,
                 hidden_mlp: int = 512, num_layers: int = 2, top_k: int = 8,
                 dropout: float = 0.1, **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(2048, d_model)
        self.drop = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList([
            SSR25F_Block(d_model, num_slots, hidden_mlp, top_k, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.embed.weight
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
    
    def forward(self, x: torch.Tensor, targets: torch.Tensor = None):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        
        h = self.drop(self.embed(x) + self.pos_embed(pos))
        
        for block in self.blocks:
            h = block(h, self.training)
        
        h = self.norm_f(h)
        logits = self.head(h)
        
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
        return logits


# ============================================================================
# VARIANT G: Slot Communication (Mini Self-Attention)
# ============================================================================

class SlotPathG(nn.Module):
    """SlotPath with slot-to-slot communication."""
    
    def __init__(self, d_model: int, num_slots: int, hidden_mlp: int,
                 top_k: int = 8, slot_dropout: float = 0.0):
        super().__init__()
        
        self.d_model = d_model
        self.num_slots = num_slots
        self.top_k = min(top_k, num_slots)
        
        self.slot_init = nn.Parameter(torch.randn(num_slots, d_model) * 0.02)
        
        self.route_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, num_slots)
        )
        
        self.slot_gru = nn.GRUCell(d_model, d_model)
        
        # SLOT COMMUNICATION: lightweight linear mixing instead of full attention
        # This adds ~2*d_model*d_model params instead of 4*d_model*d_model for MHA
        self.slot_mix_q = nn.Linear(d_model, d_model // 4)
        self.slot_mix_k = nn.Linear(d_model, d_model // 4)
        self.slot_mix_v = nn.Linear(d_model, d_model)
        self.slot_norm = nn.LayerNorm(d_model)
        
        self.slot_value_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.tau = nn.Parameter(torch.ones(1))
        self.slot_dropout = slot_dropout
    
    def forward_parallel(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        B, T, D = x.shape
        device = x.device
        
        S = self.slot_init.unsqueeze(0).expand(B, -1, -1).clone().to(x.dtype)
        
        slot_mean = S.mean(dim=1, keepdim=True).expand(-1, T, -1)
        ctx = torch.cat([x, slot_mean], dim=-1)
        
        route_logits = self.route_mlp(ctx) / (self.tau.abs() + 0.1)
        topk_values, topk_indices = torch.topk(route_logits, self.top_k, dim=-1)
        topk_weights = F.softmax(topk_values, dim=-1)
        
        alpha = torch.zeros(B, T, self.num_slots, device=device, dtype=x.dtype)
        alpha.scatter_(2, topk_indices, topk_weights)
        
        slot_input = torch.einsum('bts,btd->bsd', alpha, x)
        slot_weights_sum = alpha.sum(dim=1) + 1e-8
        slot_input_norm = slot_input / slot_weights_sum.unsqueeze(-1)
        
        slot_input_flat = slot_input_norm.reshape(B * self.num_slots, D)
        S_flat = S.reshape(B * self.num_slots, D)
        S_new_flat = self.slot_gru(slot_input_flat, S_flat)
        S_new = S_new_flat.reshape(B, self.num_slots, D)
        
        # SLOT COMMUNICATION: lightweight attention-like mixing
        q = self.slot_mix_q(S_new)  # [B, num_slots, d_model//4]
        k = self.slot_mix_k(S_new)  # [B, num_slots, d_model//4]
        v = self.slot_mix_v(S_new)  # [B, num_slots, d_model]
        
        # Scaled dot-product attention
        scale = (D // 4) ** -0.5
        attn = torch.softmax(torch.bmm(q, k.transpose(-2, -1)) * scale, dim=-1)
        S_comm = torch.bmm(attn, v)
        S_new = self.slot_norm(S_new + S_comm)
        
        slot_values = self.slot_value_proj(S_new)
        output = torch.einsum('bts,bsd->btd', alpha, slot_values)
        output = self.out_proj(output)
        
        if self.slot_dropout > 0 and training:
            output = F.dropout(output, p=self.slot_dropout, training=training)
        
        return output


class SSR25G_Block(nn.Module):
    """SSR-2.5G: Slot communication."""
    
    def __init__(self, d_model: int, num_slots: int, hidden_mlp: int,
                 top_k: int = 8, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dense = DensePath(d_model, hidden_mlp, dropout)
        self.slot = SlotPathG(d_model, num_slots, hidden_mlp, top_k)
        
        self.gate_proj = nn.Linear(d_model, 1)
        nn.init.zeros_(self.gate_proj.bias)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        B, T, D = x.shape
        normed = self.norm(x)
        
        dense_out = self.dense(normed.reshape(B*T, D)).reshape(B, T, D)
        slot_out = self.slot.forward_parallel(normed, training)
        
        gate = torch.sigmoid(self.gate_proj(normed))
        combined = gate * slot_out + (1 - gate) * dense_out
        
        return x + self.dropout(combined)


class SSR25G(nn.Module):
    """SSR-2.5G: Slot Communication"""
    
    def __init__(self, vocab_size: int, d_model: int = 256, num_slots: int = 24,
                 hidden_mlp: int = 512, num_layers: int = 2, top_k: int = 8,
                 dropout: float = 0.1, **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(2048, d_model)
        self.drop = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList([
            SSR25G_Block(d_model, num_slots, hidden_mlp, top_k, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.embed.weight
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
    
    def forward(self, x: torch.Tensor, targets: torch.Tensor = None):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        
        h = self.drop(self.embed(x) + self.pos_embed(pos))
        
        for block in self.blocks:
            h = block(h, self.training)
        
        h = self.norm_f(h)
        logits = self.head(h)
        
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
        return logits


# ============================================================================
# VARIANT H: QK-Style Routing
# ============================================================================

class SlotPathH(nn.Module):
    """SlotPath with query-key style routing (like attention)."""
    
    def __init__(self, d_model: int, num_slots: int, hidden_mlp: int,
                 top_k: int = 8, slot_dropout: float = 0.0):
        super().__init__()
        
        self.d_model = d_model
        self.num_slots = num_slots
        self.top_k = min(top_k, num_slots)
        
        self.slot_init = nn.Parameter(torch.randn(num_slots, d_model) * 0.02)
        
        # QK-style routing: token is query, slots are keys
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.scale = d_model ** -0.5
        
        self.slot_gru = nn.GRUCell(d_model, d_model)
        self.slot_value_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.slot_dropout = slot_dropout
    
    def forward_parallel(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        B, T, D = x.shape
        device = x.device
        
        S = self.slot_init.unsqueeze(0).expand(B, -1, -1).clone().to(x.dtype)
        
        # QK routing: q from tokens, k from slots
        q = self.query_proj(x)  # [B, T, D]
        k = self.key_proj(S)  # [B, num_slots, D]
        
        # Attention scores
        route_logits = torch.einsum('btd,bsd->bts', q, k) * self.scale  # [B, T, num_slots]
        
        # Top-k selection
        topk_values, topk_indices = torch.topk(route_logits, self.top_k, dim=-1)
        topk_weights = F.softmax(topk_values, dim=-1)
        
        alpha = torch.zeros(B, T, self.num_slots, device=device, dtype=x.dtype)
        alpha.scatter_(2, topk_indices, topk_weights)
        
        slot_input = torch.einsum('bts,btd->bsd', alpha, x)
        slot_weights_sum = alpha.sum(dim=1) + 1e-8
        slot_input_norm = slot_input / slot_weights_sum.unsqueeze(-1)
        
        slot_input_flat = slot_input_norm.reshape(B * self.num_slots, D)
        S_flat = S.reshape(B * self.num_slots, D)
        S_new_flat = self.slot_gru(slot_input_flat, S_flat)
        S_new = S_new_flat.reshape(B, self.num_slots, D)
        
        slot_values = self.slot_value_proj(S_new)
        output = torch.einsum('bts,bsd->btd', alpha, slot_values)
        output = self.out_proj(output)
        
        if self.slot_dropout > 0 and training:
            output = F.dropout(output, p=self.slot_dropout, training=training)
        
        return output


class SSR25H_Block(nn.Module):
    """SSR-2.5H: QK-style routing."""
    
    def __init__(self, d_model: int, num_slots: int, hidden_mlp: int,
                 top_k: int = 8, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dense = DensePath(d_model, hidden_mlp, dropout)
        self.slot = SlotPathH(d_model, num_slots, hidden_mlp, top_k)
        
        self.gate_proj = nn.Linear(d_model, 1)
        nn.init.zeros_(self.gate_proj.bias)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        B, T, D = x.shape
        normed = self.norm(x)
        
        dense_out = self.dense(normed.reshape(B*T, D)).reshape(B, T, D)
        slot_out = self.slot.forward_parallel(normed, training)
        
        gate = torch.sigmoid(self.gate_proj(normed))
        combined = gate * slot_out + (1 - gate) * dense_out
        
        return x + self.dropout(combined)


class SSR25H(nn.Module):
    """SSR-2.5H: QK-Style Routing"""
    
    def __init__(self, vocab_size: int, d_model: int = 256, num_slots: int = 24,
                 hidden_mlp: int = 512, num_layers: int = 2, top_k: int = 8,
                 dropout: float = 0.1, **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(2048, d_model)
        self.drop = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList([
            SSR25H_Block(d_model, num_slots, hidden_mlp, top_k, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.embed.weight
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
    
    def forward(self, x: torch.Tensor, targets: torch.Tensor = None):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        
        h = self.drop(self.embed(x) + self.pos_embed(pos))
        
        for block in self.blocks:
            h = block(h, self.training)
        
        h = self.norm_f(h)
        logits = self.head(h)
        
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
        return logits


# ============================================================================
# EXPORT ALL VARIANTS
# ============================================================================

VARIANTS = {
    'SSR-2.5A': SSR25A,  # Strong Residuals
    'SSR-2.5B': SSR25B,  # Pre-LayerNorm
    'SSR-2.5C': SSR25C,  # Learned Slot Init
    'SSR-2.5D': SSR25D,  # Gate Bias (favor slots)
    'SSR-2.5E': SSR25E,  # Wide Slots
    'SSR-2.5F': SSR25F,  # Multi-Head Slots
    'SSR-2.5G': SSR25G,  # Slot Communication
    'SSR-2.5H': SSR25H,  # QK-Style Routing
}

__all__ = ['SSR25A', 'SSR25B', 'SSR25C', 'SSR25D', 'SSR25E', 'SSR25F', 'SSR25G', 'SSR25H', 'VARIANTS']
