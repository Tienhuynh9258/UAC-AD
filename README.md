# UAC-AD

Source code for the paper **"UAC-AD: Unsupervised Adversarial Contrastive Learning for Anomaly Detection on Multi-source Data"**.

UAC-AD detects anomalies in cloud/microservice systems by jointly learning from three data modalities — **KPI metrics**, **logs**, and **traces** — without requiring labeled training data.

---

## Architecture Overview

![UAC-AD Overview](./result21/overview.png)

The model encodes each modality separately, fuses them via multi-modal self-attention, and reconstructs the input using an adversarial autoencoder. Windows with high reconstruction error are flagged as anomalies.

- **KPI Encoder**: Conv1d token embedding + Positional encoding
- **Log Encoder**: 4-layer Transformer
- **Trace Encoder** *(optional)*: 2-layer Graph Attention Network (GAT) on the service call graph
- **Fusion**: Multi-modal self-attention over log + KPI, with trace guiding the decoder
- **Training**: GAN adversarial loss + contrastive loss on mismatched modality pairs

---

## Requirements

- Python >= 3.7
- PyTorch 1.11.0
- CUDA-capable GPU recommended

```bash
pip install -r requirements.txt
```

**Key dependencies:** `torch==1.11.0`, `gensim==4.2.0`, `drain3>=0.9.11`, `scikit-learn==1.1.1`, `pandas==1.4.2`

---

## Quick Start

```bash
cd codes
python run.py
```

This runs on **Dataset A** (`data/chunk_10`) with default settings (KPI + log fusion, no trace branch).

---

## Datasets

| Dataset | Modalities | Source |
|---------|-----------|--------|
| **A** (default) | KPI, Logs | Spark runtime — [Zenodo 7609780](https://doi.org/10.5281/zenodo.7609780) |
| **B** (MicroSS) | KPI, Logs, Traces | QR-code login simulation — [GAIA-DataSet/MicroSS](https://github.com/CloudWise-OpenSource/GAIA-DataSet/tree/main/MicroSS) |
| **C** (SocialNetwork) | KPI, Logs, Traces | 12-service microservice app — confidential |

**Dataset A** includes CPU, memory, IO, and network metrics + Spark runtime logs.

**Dataset B (MicroSS)** covers a QR-code login workflow with 4 services, ~85 KPI dimensions, and 20 log templates (Drain3-parsed).

For preprocessing instructions, see [`docs/preprocess_micross_en.md`](docs/preprocess_micross_en.md) and [`docs/preprocess_sn.md`](docs/preprocess_sn.md).

---

## Running Experiments

### Dataset A (default)

```bash
cd codes
python run.py --data_type fuse --dataset original --data ../data/chunk_10
```

### Dataset B (MicroSS, with traces)

```bash
cd codes
python run.py \
  --data_type fuse \
  --dataset micross \
  --data ../data/micross \
  --open_trace True \
  --window_size 50
```

### Dataset C (SocialNetwork, per-scenario evaluation)

```bash
cd codes
python run.py \
  --data_type fuse \
  --dataset sn \
  --data ../data/sn \
  --open_trace True \
  --window_size 30 \
  --val_percentile 95
```

### Run multiple times (different random seeds)

```bash
python run.py --run_start 0 --run_end 5
```

---

## Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data` | `../data/chunk_10` | Path to dataset directory |
| `--dataset` | `original` | Dataset type: `original`, `micross`, `sn` |
| `--data_type` | `kpi` | Modalities to use: `fuse` (log+KPI), `log`, `kpi` |
| `--open_trace` | `False` | Enable trace branch (GAT; requires trace data) |
| `--window_size` | `5` | Sliding window size |
| `--hidden_size` | `32` | Common embedding dimension |
| `--epoches` | `50 50` | Epochs for phase 1 and phase 2 |
| `--batch_size` | `128` | Training batch size |
| `--learning_rate` | `0.001` | Optimizer learning rate |
| `--open_gan` | `True` | Enable GAN adversarial training |
| `--open_unmatch_zoomout` | `True` | Enable contrastive loss on mismatched pairs |
| `--fuse_type` | `multi_modal_self_attn` | Fusion strategy: `multi_modal_self_attn`, `concat`, `cross_attn`, `sep_attn` |
| `--criterion` | `l1` | Reconstruction loss: `l1` or `mse` |
| `--val_percentile` | `None` | If set, use this percentile of normal losses as threshold (e.g. `95`) |
| `--result_dir` | `../result21/` | Output directory for results and checkpoints |

---

## Results

![Main Results](./result21/main_result.png)

Detailed per-dataset experiment results:
- [`docs/experiment_results_micross_trace_vs_baseline_en.md`](docs/experiment_results_micross_trace_vs_baseline_en.md)
- [`docs/experiment_results_sn_trace_vs_baseline.md`](docs/experiment_results_sn_trace_vs_baseline.md)

Each experiment run saves outputs to `result21/<run_hash>/`:

```
result21/
└── <run_hash>/
    ├── params.json       # All hyperparameters
    ├── info_score.txt    # Final F1, Recall, Precision
    ├── running.log       # Training log
    └── model.ckpt        # Saved model weights
```

---

## Project Structure

```
UAC-AD/
├── codes/
│   ├── run.py                        # Main entry point
│   ├── run_sequential.py             # Memory-efficient sequential variant
│   ├── gpu0.sh / gpu1.sh             # Pre-configured experiment scripts
│   ├── data_analysis.py              # Data exploration utilities
│   ├── common/
│   │   ├── data_loads.py             # Data loading & windowing
│   │   ├── data_processing.py        # Dataset-specific preprocessing
│   │   ├── data_processing_utils.py  # Feature normalization & visualization
│   │   ├── semantics.py              # Log feature extraction (Word2Vec, Drain3)
│   │   ├── preprocess_micross.py     # MicroSS preprocessing script
│   │   ├── preprocess_sn.py          # SocialNetwork preprocessing script
│   │   ├── eval_per_scenario_sn.py   # Per-scenario evaluation for SN
│   │   └── utils.py                  # General utilities
│   └── models/
│       ├── basev3.py                 # Train/eval loop & BaseModel
│       ├── fuse_v3.py                # Multimodal fusion model
│       ├── kpi_model_v3.py           # KPI encoder/decoder
│       ├── log_model_v3.py           # Log encoder/decoder
│       ├── trace_model_v3.py         # Trace encoder (GAT)
│       └── utils.py                  # Shared modules (attention, embedders)
├── data/
│   ├── chunk_10/                     # Dataset A (train/test/unlabel .pkl)
│   ├── micross/                      # Dataset B (after preprocessing)
│   └── sn/                           # Dataset C (after preprocessing)
├── docs/                             # Architecture docs & experiment results
├── result21/                         # Output directory
└── requirements.txt
```

---

## Documentation

- [Model Architecture & Data Flow](docs/model_architecture_flow_en.md)
- [MicroSS Preprocessing Guide](docs/preprocess_micross_en.md)
- [SocialNetwork Preprocessing Guide](docs/preprocess_sn.md)
