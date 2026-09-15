# HomeSCL-DA

**Cross-home acoustic activity recognition with selective cross-home supervised contrastive learning and domain-adversarial training.**

HomeSCL-DA recognises daily activities from pre-extracted CLAP audio embeddings. It is evaluated with leave-one-home-out (LOHO): the model trains on all homes but one and is tested on the unseen home.

## Method

```
CLAP [T x 1024] -> MLP -> BiGRU -> masked mean -> activity classifier
                                        |
                                        +-> GRL -> domain discriminator   (DANN / CDAN / CDAN-E)
                                        +-> projector -> cross-home SupCon
```

The training loss has three terms.

- **Activity loss.** Class-weighted cross-entropy on all source segments, including ADASYN synthetic minority samples.
- **Domain loss.** Adversarial domain cross-entropy through a gradient reversal layer, computed on home-balanced batches of *real* audio only.
  - `multi_source_target`: one domain per source home, plus one for the held-out home. The held-out home contributes unlabeled audio: its labeled segments with the labels discarded, and optional `unlabeled_tsv` windows.
  - `multi_home`: one domain per source home. The held-out home is not used at all.
- **Cross-home SupCon loss.** Starts at `supcon_start_epoch`. It uses only reliable real source segments, which are chosen once by the current classifier. A segment is kept when the classifier predicts its annotated class and the segment is both highly confident and close to the same-class prototype of *other* homes. Positives and negatives always come from different homes.

The held-out home's labels are never used for training or for model selection. The final-epoch model is the one tested.

## Repository structure

```
HomeSCL-DA/
├── main.py                  # entry point: LOHO loop + summary
├── configs/default.json     # all hyper-parameters and data paths
├── homescl_da/
│   ├── config.py            # Config dataclass, JSON loader, validation
│   ├── dataset.py           # TSV loading, segments, feature cache, datasets, samplers
│   ├── adasyn.py            # ADASYN oversampling
│   ├── model.py             # MLP-BiGRU, gradient reversal, domain discriminator
│   ├── losses.py            # class weights, weighted CE, cross-home SupCon
│   ├── selection.py         # classifier-guided selection for SupCon
│   ├── trainer.py           # training loop, evaluation, one LOHO fold
│   └── utils.py             # seeding, plots
├── scripts/make_toy_data.py # random dataset for a quick sanity check
└── requirements.txt
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python ≥ 3.9 and PyTorch ≥ 2.0 are required. `matplotlib` is optional; without it, plots are skipped. `imbalanced-learn` provides ADASYN.

## Data format

One folder per home. Paths in the config are relative to the directory you run `main.py` from.

```
data/home_1/
├── labeled.tsv        # required
├── unlabeled.tsv      # optional, domain branch only
├── batch_index.tsv    # optional
└── features/
    ├── batch-0        # torch.save({"tensor": array [N, 1024]})
    ├── batch-1
    └── ...
```

| File | Columns | Notes |
|---|---|---|
| `labeled.tsv` | `batch_no`, `index`, `event_label`, `onset` | One row per labeled step. `batch_no` is the feature file name and `index` is the row in its tensor. If a step has several labels, the one with the later `onset` is kept. Labels not listed in `classes` are ignored. |
| `unlabeled.tsv` | `batch_no`, `index` | Steps without an activity label. They are cut into windows of `domain_window_len` steps and used only by the domain discriminator for the held-out home. |
| `batch_index.tsv` | `batch_no`, `index` | If given, unlabeled rows not listed here are dropped. |
| `features/<batch_no>` | `{"tensor": [N, 1024]}` | One CLAP embedding per step, as a NumPy array or torch tensor. |

A *segment* is a maximal run of consecutive steps with the same label inside one session file. When `dedup_chains` is true, segments of the same class in consecutive session files (`batch-12`, `batch-13`, …) are treated as one event, and only the longest one is kept.

## How to run

**Quick sanity check.** This uses random data and takes about a minute on CPU.

```bash
python scripts/make_toy_data.py --out toy_data
python main.py --config toy_data/config.json --epochs 3 --heldout home_2
```

**Full LOHO run.** Edit `configs/default.json` to point at your data, then run:

```bash
python main.py --config configs/default.json
```

**Common variations:**

```bash
# single fold
python main.py --heldout home_6

# domain-adaptation ablations (the activity architecture is identical in all four)
python main.py --ablation baseline
python main.py --ablation dann
python main.py --ablation cdan
python main.py --ablation cdan_e

# domain definition
python main.py --domain-mode multi_source_target   # source homes + unlabeled held-out home
python main.py --domain-mode multi_home            # source homes only (no target audio)
python main.py --no-unlabeled                      # ignore unlabeled_tsv files

# runtime
python main.py --epochs 50 --seed 42 --device cuda:0 --num-workers 8 --output-dir runs --no-plots
```

Command-line flags override the matching keys in the JSON config. Every other hyper-parameter is set in the config file.

## Main hyper-parameters (`configs/default.json`)

| Key | Default | Meaning |
|---|---|---|
| `epochs`, `lr`, `batch_size` | 50, 1e-3, 64 | Adam with weight decay 1e-4 and a StepLR schedule |
| `gru_hidden`, `gru_layers`, `dropout` | 128, 2, 0.5 | BiGRU encoder |
| `adasyn_k`, `adasyn_multiplier` | 5, 1.0 | ADASYN neighbours; target count per class = multiplier x majority count |
| `ablation`, `domain_mode` | dann, multi_source_target | Domain-adversarial branch |
| `domain_warmup_epochs`, `domain_lambda_cap` | 5, 0.2 | GRL coefficient schedule: sigmoid ramp, capped |
| `domain_samples_per_home` | 10 | Per-domain quota in each domain batch |
| `supcon_start_epoch`, `supcon_weight`, `supcon_temperature` | 11, 0.2, 0.1 | Cross-home SupCon |
| `select_top_frac`, `select_min_per_cell` | 0.1, 2 | Selection within each (class, home) cell |
| `dedup_chains`, `dedup_keep` | true, longest | Merge cross-file repeats of one event |
| `heldout` | null | List of held-out homes; null runs every fold |

## Outputs

```
outputs/<ablation>/
├── config_used.json
├── summary.csv                       # macro-F1 per held-out home
└── heldout_<home>/
    ├── model_final.pt
    ├── discriminator_final.pt        # not written for baseline
    ├── history.csv                   # per-epoch losses, domain accuracy, held-out F1
    ├── selection_audit.csv           # per-segment selection scores and decisions
    ├── classification_report.txt
    ├── confusion_matrix.csv / .png
    ├── curves.png
    └── result.json
```

The held-out F1 printed each epoch (`track_heldout`) is a diagnostic only. It never affects training or which model is reported.

## Reproducibility

Each fold re-seeds Python, NumPy and PyTorch and enables deterministic algorithms. On GPU, `CUBLAS_WORKSPACE_CONFIG=:4096:8` is set automatically.

The training and held-out loaders share one random generator. As a result, setting `track_heldout` to false changes the shuffling order.

## License

Released for anonymous review. License information will be added upon publication.
