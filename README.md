# TAHA

Official implementation of **TAHA**, a unified neural combinatorial optimization
model for vehicle routing problems with heterogeneous constraints
(energy consumption, time windows, backhaul, nonlinear / partial charging).

TAHA is built on a POMO-style multi-start autoregressive policy and combines:

| Component | Location | Description |
| --- | --- | --- |
| **Unified VRP environment** | `rl4co/envs/routing/unified_vrp/` | Trains on a mixture of VRP variants. Each constraint (energy, time windows, backhaul, nonlinear charging, partial charging) is activated independently with Bernoulli sampling, so a single model generalizes across the whole variant family. |
| **Switch-LoRA MoE encoder** | `rl4co/models/nn/moe.py`, `rl4co/models/nn/graph/attnnet.py` | Loss-free Switch routing over lightweight LoRA experts on top of a shared dense base FFN (4 experts in the default config). |
| **SA adapter** | `rl4co/models/zoo/am/adapter.py` | A middle layer between encoder and decoder that injects route context and energy / constraint feasibility features into node embeddings (enabled by default, `use_adapter=True`). |
| **Heterogeneous attention** | `rl4co/models/nn/graph/attnnet.py` | Constraint-aware graph attention (`attention_type="heterogeneous"`), fusing node, constraint and distance embeddings. Alternative attention types (`mha`, `ceca`, `prefix_mha`, `knn_local`) are available in the same module. |

Training follows the POMO recipe: 8 dihedral augmentations (at validation / test),
multi-start decoding, a shared top-k baseline, and multi-task reward normalization
for the mixed constraint variants.

## Repository layout

```
TAHA/
├── run.py                      # entry point: training / testing
├── configs/
│   ├── main.yaml               # global defaults (Hydra)
│   ├── experiment/routing/pomo.yaml   # TAHA training configuration
│   ├── model/pomo.yaml         # POMO model hyperparameters
│   └── env/unified_vrp.yaml    # unified VRP environment
└── rl4co/                      # model & environment library
    ├── envs/routing/unified_vrp/     # unified VRP env + generator
    ├── envs/routing/evrp|evrptw/     # base EVRP / EVRPTW environments
    ├── models/zoo/pomo/              # POMO training loop (multi-start, top-k baseline)
    ├── models/zoo/am/                # attention policy, encoder, decoder, SA adapter
    ├── models/nn/                    # attention, transformer, MoE / LoRA layers
    ├── models/rl/                    # REINFORCE baselines (Lightning module)
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
# disable the MoE experts (falls back to a dense feed-forward layer)
python run.py model.policy_kwargs.moe_kwargs.encoder=null

# disable the SA adapter
python run.py model.policy_kwargs.use_adapter=False
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
