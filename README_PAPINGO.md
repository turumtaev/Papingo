# Papingo Notes

This file is the longer note for the Papingo experiment. The main project summary now lives in `README.md`.

## Short version

Papingo is a `nanoGPT` experiment where each layer gets:

- its own token embedding table
- its own prediction head
- a confidence score that can influence what deeper layers are allowed to attend to

The repo records simulated early exits and per-layer behavior, but it does not implement true generation-time compute skipping yet.

## Implemented pieces

- Per-layer token embeddings
- Per-layer logits heads
- `confidence_mode=max|gold`
- `layer_supervision=all|skip_easy`
- Confidence-based masking of deeper-layer context
- `detach_between_layers`
- `layer_widths`
- `gate_mode=learned`
- `layer_contexts`
- Inference-simulated loss
- Exit-rate and exit-correct logging
- Overall accuracy logging
- `scripts/papingo_table.py` for token-by-token exit inspection

## Important limitation

The idea was to make shallow layers handle easy tokens and save deeper-layer compute. The current code only measures that behavior indirectly:

- training and eval compute per-layer confidences
- metrics estimate where each token would have exited
- generation still runs all layers

So the experiment answered "do shallow exits appear?" more than "does this already save wall-clock inference cost?"

## Fork point and timeline

- Fork base: `karpathy/nanoGPT` commit `3adf61e`
- Upstream commit date: 2025-11-12
- Papingo implementation window: 2026-01-28 through 2026-01-30
- Last Papingo commit: `9b8e900`

## Metrics kept in the repo

No vanilla baseline was run during these experiments. For reference, the
original nanoGPT README reports approximately `1.88` validation loss for the
matching small CPU setup and `1.4697` for the matching A100 Shakespeare setup.

### CPU

- `val loss: 1.8779`
- `val infer: 1.8633`
- `val acc: 0.447`
- `val exit rates: [0:0.056, 1:0.023, 2:0.008, 3:0.913]`

### A100

- `val loss: 1.5197`
- `val infer: 1.4947`
- `val acc: 0.571`
- `val exit rates: [0:0.195, 1:0.105, 2:0.038, 3:0.017, 4:0.006, 5:0.640]`

## Files

- `model.py`: architecture and masking logic
- `train.py`: config surface, logging, eval summaries
- `PAPINGO_TODO.md`: original checklist
- `scripts/papingo_table.py`: visualization helper
- `papingo_table_example.md`: example output

## Unfinished work

- No real early-exit sampling path
- No Papingo-aware HellaSwag eval
- No dedicated masking tests
- No locally reproduced comparison suite against a vanilla baseline
