# SynEnergy

**Anomaly Semantic-Guided Diffusion for Synthetic Energy Data Generation**  
(KDD 2027 AI4S)

SynEnergy is a two-stage diffusion framework for **anomaly-preserving synthetic energy data generation**.

- **HG-ASL** learns region-specific anomaly semantics from residual consumption patterns and cross-region dependencies.
- **AS-Diff** injects the learned anomaly semantics into a diffusion Backbone Denoiser through **AS-Control**.

## Framework

![SynEnergy framework](SynEnergy/Figures/Framework.png)

## Method Summary

![Method table](SynEnergy/Figures/Table.png)

| Stage | Components | Role |
|---|---|---|
| **HG-ASL** | SR-Encoder, AS-Pooling, RAPA-Graph | Learns region-specific anomaly semantics |
| **AS-Diff** | Backbone Denoiser, AS-Control | Generates anomaly-preserving synthetic consumption data |

**Configuration:** `SynEnergy/Config/Extreme/power_FL.yaml`

**Bundled data:** `SynEnergy/Data/Power/Tallahassee_seq_180_sample_10000.npy`

---

## Setup

```bash
cd SynEnergy
pip install -U pip
pip install -r requirements.txt
```

Python 3.10 or later is recommended. A CUDA-enabled GPU is recommended.

---

## Quick Start

From the repository root:

```bash
bash Bash/run_power_FL1.sh
```

The script runs:

1. Backbone Denoiser training
2. HG-ASL and AS-Control training
3. Synthetic data generation

All stages use:

```text
--name power_FL1
--config_file Config/Extreme/power_FL.yaml
```

---

## Manual Commands

Run the commands from `SynEnergy/`:

```bash
cd SynEnergy

CONFIG=Config/Extreme/power_FL.yaml
NAME=power_FL1
```

### Step 1: Train the Backbone Denoiser

```bash
python main.py \
  --name "${NAME}" \
  --config_file "${CONFIG}" \
  --train \
  --condition
```

### Step 2: Train HG-ASL and AS-Control

```bash
python main.py \
  --name "${NAME}" \
  --config_file "${CONFIG}" \
  --train \
  --controlnet \
  --condition
```

### Step 3: Generate Synthetic Data

```bash
python main.py \
  --name "${NAME}" \
  --config_file "${CONFIG}" \
  --sample \
  --controlnet \
  --condition
```

Optional arguments:

```text
--seed 42
--sample_seed 42
```

---

## Outputs

| Path | Description |
|---|---|
| `SynEnergy/OUTPUT/power_FL1/` | Logs, configuration snapshots, and generated data |
| `SynEnergy/OUTPUT/power_FL1/intermediate/` | Intermediate anomaly representations and HG-ASL modules |
| `SynEnergy/Checkpoint/power_FL1_180/` | Backbone Denoiser and AS-Control checkpoints |

Key generated files:

- `generate_data_norm_power_FL1.npy`: normalized synthetic consumption sequences
- `generate_cond_power_FL1.npy`: region condition for each generated sequence

---

## Evaluation

Evaluation scripts are provided in:

```text
SynEnergy/Experiment/Result_Analysis/Metric_Check.ipynb
```

Set:

```python
RUN_NAME = "power_FL1"
```

The evaluation includes:

- **Overall Generation Fidelity:** T-Wass., D-Wass., S-Wass., MMD
- **Anomaly Preservation Fidelity:** A-Rate, A-Count, A-Energy, A-Tail
- **Downstream Quality:** Det-PRAUC, Pred-PRAUC

---

## Repository Layout

```text
SynEnergy/
├── main.py
├── Config/
│   └── Extreme/
│       └── power_FL.yaml
├── Data/
│   └── Power/
├── controlnet/
├── Models/
├── Figures/
│   ├── Framework.png
│   └── Table.png
├── Bash/
│   └── run_power_FL1.sh
└── Experiment/
    └── Result_Analysis/
        └── Metric_Check.ipynb
```

## Citation

Citation information will be added after publication.