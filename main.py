"""HomeSCL-DA (proposed), leave-one-home-out.

    python main.py                          # HomeSCL-DA: selective cross-home SupCon + DANN
    python main.py --no-da                  # HomeSCL without domain adaptation (SupCon only)
    python main.py --heldout home_6         # one fold only
    python main.py --supcon-selection standard        # SupCon on all source segments
    python main.py --supcon-selection random_matched  # random, same counts as selective

The domain-adaptation ablations (baseline / DANN / CDAN / CDAN-E, no SupCon)
"""
import argparse
import json
import os
import warnings
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd

from homescl_da.config import DOMAIN_MODES, SUPCON_SELECTIONS, load_config
from homescl_da.dataset import (FeatureStore, build_segments, load_labeled_frames,
                                load_unlabeled_windows)
from homescl_da.trainer import run_fold

PROPOSED_DA = "dann"            # domain loss of HomeSCL-DA


def parse_args():
    p = argparse.ArgumentParser(description="HomeSCL-DA leave-one-home-out experiment")
    p.add_argument("--config", default="configs/default.json")
    p.add_argument("--heldout", nargs="+", help="held-out home name(s); default: config / all")
    p.add_argument("--domain-mode", dest="domain_mode", choices=DOMAIN_MODES)
    p.add_argument("--epochs", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--batch-size", dest="batch_size", type=int)
    p.add_argument("--num-workers", dest="num_workers", type=int)
    p.add_argument("--output-dir", dest="output_dir")
    p.add_argument("--device", help="auto | cpu | cuda | cuda:0 ...")
    p.add_argument("--no-unlabeled", dest="use_unlabeled_target", action="store_const", const=False)
    p.add_argument("--no-plots", dest="save_plots", action="store_const", const=False)
    p.add_argument("--supcon-selection", dest="supcon_selection", choices=SUPCON_SELECTIONS,
                   help="which source segments enter SupCon (default: config, 'selective')")
    p.add_argument("--no-da", dest="no_da", action="store_true",
                   help="HomeSCL without domain adaptation: SupCon kept, domain branch off")
    return p.parse_args()


def main():
    warnings.filterwarnings("ignore")
    args = vars(parse_args())
    no_da = args.pop("no_da")
    cfg = load_config(args.pop("config"), args)
    if not cfg.use_supcon:
        raise SystemExit("HomeSCL needs use_supcon=true in the config "
                         "(for runs without SupCon use main_ablation.py).")
    if cfg.ablation not in ("dann", "baseline") and not no_da:
        print(f"  note: config ablation={cfg.ablation!r} ignored; HomeSCL-DA uses {PROPOSED_DA}")
    cfg.ablation = "baseline" if no_da else PROPOSED_DA   # baseline = no discriminator
    name = "homescl_no_da" if no_da else "homescl_da"
    if cfg.supcon_selection != "selective":             # keep ablation runs apart
        name += f"_{cfg.supcon_selection}"
    out_root = Path(cfg.output_dir) / name
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "config_used.json").write_text(json.dumps(cfg.to_dict(), indent=2))

    print(f"device={cfg.torch_device} classes={cfg.n_classes} homes={len(cfg.homes)}")
    print("Loading labeled data...")
    segments = build_segments(load_labeled_frames(cfg), cfg)
    store = FeatureStore({h["name"]: h["feature_dir"] for h in cfg.homes}, cfg.cache_size)

    folds = cfg.heldout or [h["name"] for h in cfg.homes]
    unlabeled = {}
    if not no_da and cfg.domain_mode == "multi_source_target" and cfg.use_unlabeled_target:
        for home_id, h in enumerate(cfg.homes):
            if h["name"] in folds:
                unlabeled[h["name"]] = load_unlabeled_windows(h, home_id, cfg)

    results = [run_fold(cfg, home, segments, unlabeled, store, out_root) for home in folds]

    summary = pd.DataFrame(results)
    summary.to_csv(out_root / "summary.csv", index=False)
    title = ("HomeSCL (no DA)" if no_da else "HomeSCL-DA") + f"  supcon={cfg.supcon_selection}"
    print(f"\n{'=' * 50}\nLOHO summary  {title}\n{'=' * 50}")
    for r in results:
        print(f"{r['heldout']:<32}{r['macro_f1']:>9.4f}")
    print(f"{'MEAN':<32}{np.mean(summary['macro_f1']):>9.4f}")
    print(f"results -> {out_root}")


if __name__ == "__main__":
    main()
