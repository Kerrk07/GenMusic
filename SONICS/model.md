# SONICS SpecTTTra-Gamma-120s Model Configuration

## Model Overview

| Property | Value |
|----------|-------|
| **Architecture** | SpecTTTra-γ (Spectro-Temporal Tokens Transformer) |
| **Task** | Fake Song Detection (binary classification) |
| **Total Parameters** | 24M (20M active) |
| **License** | MIT |
| **Paper** | ICLR 2025 Poster |

## Key Config

### Audio Input

| Parameter | Value |
|-----------|-------|
| `sample_rate` | 16,000 Hz |
| `max_time` | 120 seconds |
| `max_len` | 1,920,000 samples |
| `normalize` | mean-std normalization |

### Mel Spectrogram

| Parameter | Value |
|-----------|-------|
| `n_fft` | 2048 |
| `hop_length` | 512 |
| `n_mels` | 128 |
| `f_min` / `f_max` | 20 Hz – 8,000 Hz |

### Transformer Architecture

| Parameter | Value |
|-----------|-------|
| `embed_dim` | 384 |
| `num_heads` | 6 |
| `num_layers` | 12 |
| `mlp_ratio` | 2.67 |
| `input_shape` | [128, 3744] |
| `f_clip` / `t_clip` | 5 / 7 |

### Training

| Parameter | Value |
|-----------|-------|
| `num_classes` | 1 (BCEWithLogitsLoss) |
| `optimizer` | AdamW (lr=0.0005, weight_decay=0.05) |
| `scheduler` | Cosine with 5-epoch warmup |
| `batch_size` | 128 |
| `epochs` | 50 |
| `label_smoothing` | 0.02 |

### Performance (120s variant)

| Metric | Value |
|--------|-------|
| F1 | 0.88 |
| Sensitivity | 0.79 |
| Specificity | 0.99 |
| FLOPs | 10.1G |
| Memory | 1.6GB |
