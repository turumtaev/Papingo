# Papingo

Papingo is an experimental `nanoGPT` fork that tests whether next-token prediction can be distributed across layers instead of being done only by the final layer.

This repo was forked from `karpathy/nanoGPT` at commit `3adf61e`, dated 2025-11-12. The Papingo work happened in 20 commits from 2026-01-28 to 2026-01-30, ending at `9b8e900` (`Add papingo table script and docs`).

## What the idea was

The intended hypothesis was:

- every Transformer layer should have its own token embedding path and prediction head
- shallow layers should be allowed to handle "easy" tokens
- deeper layers should focus on harder tokens instead of reprocessing everything

The implementation approximates that design with confidence-aware masking
during training and evaluation, plus metrics that estimate where each token
would have exited. It does not yet realize the full compute-saving behavior.

## What was actually implemented

- One token embedding table per layer in `model.py`.
- One logits head per layer in `model.py`.
- Per-layer confidence using either:
  - `confidence_mode=max`
  - `confidence_mode=gold`
- Confidence-based context masking:
  - if layer `L` is confident on a token, layer `L+1` can be prevented from attending to that token
- Configurable supervision:
  - `layer_supervision=all`
  - `layer_supervision=skip_easy`
- Optional `detach_between_layers`
- Optional `layer_widths` for smaller early layers
- Optional `gate_mode=learned` to blend the hidden state with the current layer's token embedding
- Optional `layer_contexts` for per-layer attention windows
- Training and eval logging for:
  - per-layer losses
  - inference-simulated loss
  - exit rates
  - exit-correct rates
  - overall accuracy
- `scripts/papingo_table.py` to visualize which layer would have produced each token

## What is not finished

- Real compute-saving early exit in generation is not implemented.
  - `sample.py` and `model.generate()` still run all layers.
  - The repo measures simulated exit behavior, but it does not yet skip deeper-layer compute at sampling time.
- Papingo-specific benchmark evaluation is not finished.
  - `hellaswag.py` was added as a stub, but it is not wired up as a Papingo eval.
- The masking behavior does not have dedicated tests.
- The work was only pushed through a `shakespeare_char` MVP, not a broader benchmark suite.

## Best recorded metrics

These numbers come from the Papingo notes committed into the repo.

### CPU MVP

Config:

- `block_size=64`
- `n_layer=4`
- `n_head=4`
- `n_embd=128`
- `confidence_mode=max`
- `confidence_threshold=0.9`
- `layer_supervision=all`

Metrics:

- `val loss: 1.8779`
- `val infer: 1.8633`
- `val acc: 0.447`
- `val exit rates: [0:0.056, 1:0.023, 2:0.008, 3:0.913]`

### A100 run

Config:

- `block_size=256`
- `n_layer=6`
- `n_head=6`
- `n_embd=384`
- `dropout=0.2`
- `confidence_mode=max`
- `confidence_threshold=0.9`
- `layer_supervision=all`

Metrics:

- `val loss: 1.5197`
- `val infer: 1.4947`
- `val acc: 0.571`
- `val exit rates: [0:0.195, 1:0.105, 2:0.038, 3:0.017, 4:0.006, 5:0.640]`

## Reported nanoGPT baselines

No vanilla baseline was rerun as part of the Papingo experiment. The numbers
below are copied from the
[`karpathy/nanoGPT` README](https://github.com/karpathy/nanoGPT#quick-start),
which reports results for the same Shakespeare dataset and model dimensions.

| Configuration | Original nanoGPT val loss | Papingo val loss | Papingo inference-simulated loss |
| --- | ---: | ---: | ---: |
| CPU: block 64, 4 layers, 4 heads, embedding size 128, 2,000 iterations | ~1.88 | 1.8779 | 1.8633 |
| A100: block 256, 6 layers, 6 heads, embedding size 384, dropout 0.2 | 1.4697 | 1.5197 | 1.4947 |

These are reference comparisons, not a controlled rerun. In particular, the
CPU baseline is published only to two decimal places. The A100 Papingo final
loss is `0.0500` higher than the reported original result; its simulated-exit
loss is `0.0250` higher.

## Commit sequence

The Papingo work in this repo followed this path:

1. `ea0a3b7` Add papingo MVP todo
2. `e411378` Add papingo config knobs
3. `a697738` Use per-layer embeddings and heads
4. `8d55eee` Add confidence masking in attention
5. `74f2dac` Log inference-sim loss and per-layer losses
6. `0dc5b39` Add exit-rate metrics and papingo README
7. `a4d4ec8` Pass papingo config into model
8. `97a6e2d` Add detach_between_layers option
9. `dccdac4` Add layer_widths for variable per-layer dims
10. `9f2f533` Add hellaswag eval stub and update todo
11. `2dd4bdf` Log overall accuracy
12. `47717aa` Add gated per-layer embedding injection
13. `13cd2ae` Add per-layer context windows
14. `fc6d2dd` Skip window mask when layer_contexts unset
15. `99a5cf0` Fix SDPA mask with causal attention
16. `9f24356` Fix SDPA mask semantics
17. `47ea4d9` Update papingo README with best metrics
18. `f120dad` Adjust confidence masking and update metrics
19. `57c0046` Update Papingo 0.9_max metrics
20. `9b8e900` Add papingo table script and docs

## Files that matter

- `model.py` contains the Papingo architecture changes
- `train.py` contains the training loop, logging, and config knobs
- `PAPINGO_TODO.md` is the original implementation checklist
- `README_PAPINGO.md` is the longer experiment note
- `scripts/papingo_table.py` renders per-token exit tables
- `papingo_table_example.md` is a sample visualization

## Quick run

Prepare the Shakespeare character dataset:

```sh
python data/shakespeare_char/prepare.py
```

Run the small CPU configuration used for the MVP:

```sh
python train.py \
  --dataset=shakespeare_char \
  --device=cpu \
  --compile=False \
  --eval_interval=250 \
  --eval_iters=20 \
  --log_interval=1 \
  --batch_size=12 \
  --block_size=64 \
  --n_layer=4 \
  --n_head=4 \
  --n_embd=128 \
  --max_iters=2000 \
  --lr_decay_iters=2000 \
  --dropout=0.0 \
  --bias=False \
  --confidence_threshold=0.9 \
  --confidence_mode=max \
  --layer_supervision=all \
  --detach_between_layers=False \
  --gate_mode=none
```

## Status

This repository is the last version of the Papingo idea that was actually implemented. It is best treated as an experiment archive, not as a maintained training stack.
