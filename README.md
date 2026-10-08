# TAHA

Official implementation of **TAHA: Task-Adaptive Heterogeneous Attention for
Multi-Task Electric Vehicle Routing Problems (MTEVRP)**.

TAHA disentangles static geometry from dynamic feasibility through a stratified
processing pipeline built from four synergistic components:

| Component | Location in this repository | Role |
| --- | --- | --- |
| **Constraint-Aware Heterogeneous Attention (CAHA)** | `rl4co/models/nn/graph/attnnet.py`, `rl4co/models/nn/attention.py` | Foundational structural disentanglement: explicitly separates invariant routing semantics from feasibility-driven interactions dictated by EVRP-specific constraints (depot / station / customer heterogeneity). |
| **Decoupled-Synergistic MoE (DS MoE)** | `rl4co/models/nn/moe.py` | Decomposes representation learning into a shared backbone for universal geometric invariances and a sparse bank of low-rank (LoRA) experts for task-specific policy adaptations; a loss-free stochastic switch router performs Top-1 expert selection. |
| **State-Aware Adapter (SAA)** | `rl4co/models/zoo/am/adapter.py` | Dynamic interface between encoder and decoder: harmonizes static problem representations with real-time execution states by injecting feasibility signals (energy margins, constraint attributes) into the decoding process. |
| **Constraint-Modulated Decoder** | `rl4co/models/zoo/am/decoder.py` | Harmonizes node desirability with operational viability in logit space: the SAA energy score is injected as a differentiable logit-level bias (soft modulation) on top of the environment feasibility mask. |

## Training regime

TAHA is trained with a unified multi-task regime that combines:

- **Stochastic task composition** — each instance activates a random combination
  of operational constraints (energy, time windows, backhaul, nonlinear charging,
  partial charging) via independent Bernoulli(0.5) sampling
  (`rl4co/envs/routing/unified_vrp/`).
- **Multi-start policy optimization** with 8 dihedral augmentations at
  validation / test time and a shared top-k baseline (POMO-style;
  `rl4co/models/zoo/pomo/`).
- **Task-stratified reward normalization** (multi-task norm) to stabilize policy
  gradients across variants with different reward scales
  (`configs/experiment/routing/taha.yaml`).

Evaluated on nine MTEVRP variants: `CEVRP`, `EVRPTW`, `EVRPTWNC`, `EVRPTWPC`,
`EVRPTWNCPC`, `EVRPBTW`, `EVRPBTWNC`, `EVRPBTWPC`, `EVRPBTWNCPC`.

## Repository layout

```
TAHA/
├── run.py                      # entry point: training / testing
├── configs/
│   ├── main.yaml               # global defaults (Hydra)
│   ├── experiment/routing/taha.yaml   # TAHA training configuration
│   ├── model/pomo.yaml         # backbone model hyperparameters
│   └── env/unified_vrp.yaml    # MTEVRP environment
└── rl4co/                      # model & environment library
    ├── envs/routing/unified_vrp/     # MTEVRP env + generator (stochastic task composition)
    ├── envs/routing/evrp|evrptw/     # base EVRP / EVRPTW environments
    ├── models/nn/graph/attnnet.py    # CAHA encoder network
    ├── models/nn/moe.py              # DS MoE (LoRA experts + stochastic switch router)
    ├── models/zoo/am/adapter.py      # SAA
    ├── models/zoo/am/decoder.py      # constraint-modulated decoder
    ├── models/zoo/pomo/              # POMO-style multi-start training loop
    ├── data/                         # instance generation and augmentation utilities
    ├── tasks/train.py                # Hydra training / testing task
    └── utils/                        # trainer, callbacks, decoding helpers
```

## Installation

Tested with Python 3.10, PyTorch 2.11 (CUDA 13.0), Lightning 2.6.

```bash
conda create -n taha python=3.10 -y
conda activate taha

# install the PyTorch build matching your CUDA version first, see https://pytorch.org
pip install torch

pip install -r requirements.txt
```

No dataset download is required: training / evaluation instances are generated
on the fly from the `generator_params` in the config.

## Training

Full training (default: 300 epochs, batch size 256, 15 customers + 8 stations):

```bash
python run.py
```

Quick smoke test on a single GPU (~1 minute):

```bash
python run.py model.train_data_size=256 model.val_data_size=64 model.batch_size=64 \
    trainer.max_epochs=1 +trainer.limit_train_batches=2
```

Checkpoints and logs are written to `logs/train/runs/<timestamp>/`.

### Ablation switches

```bash
# w/o DS MoE: disable the experts (falls back to a dense feed-forward layer)
python run.py model.policy_kwargs.moe_kwargs.encoder=null

# w/o SAA: disable the state-aware adapter
python run.py model.policy_kwargs.use_adapter=False

# w/o CAHA: switch the heterogeneous attention to plain multi-head attention
# (set attention_type="mha" in rl4co/models/zoo/am/encoder.py)
```

## Evaluation

```bash
# evaluate a trained checkpoint on generated test instances
python run.py train=False test=True ckpt_path=/path/to/epoch_XXX.ckpt

# evaluate on custom instance files (.txt / .npz / .evrp / .csv)
python run.py train=False test=True ckpt_path=/path/to/epoch_XXX.ckpt \
    env.test_file=/path/to/instances.txt
```

During validation / testing the best solution path (a tensor of visited node
indices) is logged, which can be parsed from stdout for downstream analysis.

## Acknowledgment

This project is built upon [RL4CO](https://github.com/ai4co/rl4co) (MIT License);
the `rl4co/` package in this repository is a trimmed and modified version of it.

## License

[MIT](LICENSE)
