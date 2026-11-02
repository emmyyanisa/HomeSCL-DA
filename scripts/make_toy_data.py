"""
Usage:
    python scripts/make_toy_data.py --out toy_data
    python main.py --config toy_data/config.json --epochs 3 --heldout home_3
"""
import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from homescl_da.config import DEFAULT_CLASSES, Config


CLASS_PROBS = [0.15, 0.2, 0.3, 0.2, 0.05, 0.1]
FRAME_SEC = 0.1


def write_tsv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("\t".join(header) + "\n")
        f.writelines("\t".join(map(str, r)) + "\n" for r in rows)


def make_home(root, home_no, rng, class_means, n_sessions, steps, dim):
    meta_dir = root / "dataset" / "metadata"
    feat_dir = root / f"database-home{home_no}.db"
    feat_dir.mkdir(parents=True, exist_ok=True)

    pc = f"PC{home_no:02d}"
    offset_vec = rng.normal(0, 0.5, dim)
    base_day = dt.date(2026, 1, 1) + dt.timedelta(days=home_no * 7)
    labeled, unlabeled, index_rows = [], [], []

    for s in range(n_sessions):
        bno = f"batch-{s}"
        date = (base_day + dt.timedelta(days=s // 4)).isoformat()
        session_start = dt.datetime.combine(
            dt.date(2000, 1, 1), dt.time(8 + (s % 4) * 3, int(rng.integers(0, 60)))
        )
        x = rng.normal(0, 1, (steps, dim)).astype(np.float32) + offset_vec

        def clock(i):
            return (session_start + dt.timedelta(seconds=i * FRAME_SEC)).strftime("%H:%M:%S")

        t = 0
        while t < steps:
            length = int(rng.integers(5, 25))
            seg = range(t, min(t + length, steps))
            if rng.random() < 0.25:
                unlabeled += [(pc, bno, i, date, clock(i)) for i in seg]
            else:
                c = int(rng.choice(len(DEFAULT_CLASSES), p=CLASS_PROBS))
                for i in seg:
                    x[i] += class_means[c]
                    onset = round(i * FRAME_SEC, 1)
                    labeled.append((pc, bno, i, date, clock(i), onset,
                                    round(onset + FRAME_SEC, 1), DEFAULT_CLASSES[c]))
            t += length

        index_rows += [(bno, f"{bno}.wav", i, date, clock(i)) for i in range(steps)]
        torch.save({"tensor": x}, feat_dir / bno)

    write_tsv(meta_dir / f"home{home_no}_labeled.tsv",
              ["participantPC", "batch_no", "index", "date", "start_time",
               "onset", "offset", "event_label"], labeled)
    write_tsv(meta_dir / f"home{home_no}_unlabeled.tsv",
              ["participantPC", "batch_no", "index", "date", "start_time"], unlabeled)
    write_tsv(meta_dir / f"batch_index_home{home_no}.tsv",
              ["batch_no", "filename", "index", "date", "time"], index_rows)

    return {
        "name": f"home_{home_no}",
        "labeled_tsv": str(meta_dir / f"home{home_no}_labeled.tsv"),
        "unlabeled_tsv": str(meta_dir / f"home{home_no}_unlabeled.tsv"),
        "batch_index_tsv": str(meta_dir / f"batch_index_home{home_no}.tsv"),
        "feature_dir": str(feat_dir),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="toy_data")
    p.add_argument("--homes", type=int, default=6)
    p.add_argument("--sessions", type=int, default=12)
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if len(CLASS_PROBS) != len(DEFAULT_CLASSES):
        sys.exit(f"CLASS_PROBS has {len(CLASS_PROBS)} entries but there are "
                 f"{len(DEFAULT_CLASSES)} classes: {DEFAULT_CLASSES}")

    dim = Config().clap_dim
    rng = np.random.default_rng(args.seed)
    root = Path(args.out)
    class_means = rng.normal(0, 0.15, (len(DEFAULT_CLASSES), dim)).astype(np.float32)

    homes = [make_home(root, i, rng, class_means, args.sessions, args.steps, dim)
             for i in range(1, args.homes + 1)]

    cfg = Config(homes=homes, output_dir=str(root / "outputs"), num_workers=0,
                 supcon_start_epoch=2, domain_warmup_epochs=0)
    (root / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    print(f"toy dataset -> {root}/  config -> {root / 'config.json'}")


if __name__ == "__main__":
    main()
