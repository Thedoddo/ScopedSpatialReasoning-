# SSR-2.5: Selective Slot Routing with Hybrid Gating

Official implementation for the paper: *"SSR-2.5: Hybrid Dense-Slot Architecture for Efficient Language Modeling"*

## Overview

SSR-2.5 introduces a hybrid architecture that combines:
- **Dense path**: Fast FFN processing for "easy" tokens
- **Slot path**: Structured memory routing for complex patterns
- **Learned gate**: Per-token routing between paths

## Installation

```bash
pip install -r requirements.txt
```

Requires: PyTorch >= 2.0, numpy

## Quick Start

```bash
# Train SSR-2.5 on WikiText-103
python train.py --model ssr25 --data path/to/wikitext103.txt --epochs 10

# Train Transformer baseline
python train.py --model transformer --data path/to/wikitext103.txt --epochs 10

# Run all ablations
python train.py --model all --data path/to/wikitext103.txt --epochs 10
```

## Models

| Model | Description | Params (~2M config) |
|-------|-------------|---------------------|
| `ssr25` | SSR-2.5 Hybrid Gate | 2.31M |
| `ssr25_noslots` | SSR-2.5 No-Slots (ablation) | 1.08M |
| `transformer` | Transformer baseline | 2.28M |
| `ssr25a` | Variant A: Strong Residuals | 2.28M |
| `ssr25b` | Variant B: Pre-LayerNorm | 2.28M |
| `ssr25c` | Variant C: Learned Slot Init | 2.28M |
| `ssr25d` | Variant D: Gate Bias | 2.28M |
| `ssr25e` | Variant E: Wide Slots | 2.66M |
| `ssr25f` | Variant F: Multi-Head Slots | 2.13M |
| `ssr25g` | Variant G: Slot Communication | 2.48M |
| `ssr25h` | Variant H: QK-Style Routing | 2.41M |

## Results

See `results/` for training logs and final metrics.

## Citation

```bibtex
@article{comer2025ssr25,
  title={SSR-2.5: Hybrid Dense-Slot Architecture for Efficient Language Modeling},
  author={Comer, Jackson},
  year={2025}
}
```

## License

Apache 2.0 - see LICENSE
