# MLP64-BiGRU ACOUSTIC LOHO ABLATION: BASELINE / DANN / CDAN / CDAN-E
# Cleaned from run_train_test_5_4_CALDA_SelCL_episode_clean(2).py
#
# Purpose of this file:
#   * one activity classifier only: MLP(1024->256->64) -> BiGRU -> classifier
#   * one task loss only: CE(final BiGRU classifier)
#   * compare only ABLATION and DOMAIN_ADAPT_MODE
#   * one MLP64-BiGRU classifier and one task CE loss; no separate MLP classification head
#   * no validation-home selector; held-out LOHO home supplies diagnostic F1 trend
#   * legacy auxiliary experimental branches removed
#   * always save/test the final epoch

import sys, warnings, random, collections, re, math
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from datetime import datetime
from collections import OrderedDict, defaultdict
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

torch.serialization.add_safe_globals([datetime])


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    import os
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def seed_worker(worker_id):
    worker_seed = (torch.initial_seed() + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ============================================================
# CONFIG -- experimental switches kept intentionally small
# ============================================================
ABLATION = "baseline"   # "baseline" | "dann" | "cdan" | "cdan_e"
if ABLATION not in ("baseline", "dann", "cdan", "cdan_e"):
    raise ValueError('ABLATION must be one of "baseline", "dann", "cdan", "cdan_e"')

DOMAIN_ADAPT_MODE = "multi_source_target"   # "multi_source_target" | "multi_home" | "binary_source_target"
if DOMAIN_ADAPT_MODE not in ("multi_source_target", "multi_home", "binary_source_target"):
    raise ValueError('DOMAIN_ADAPT_MODE must be "multi_source_target", "multi_home", or "binary_source_target"')

# Domain-adversarial training
DOMAIN_DISC_HIDDEN = 64
LAMBDA_DOMAIN_MAX = 1.0
DOMAIN_LAMBDA_CAP = 0.20
DOMAIN_WARMUP_EPOCHS = 5

# Dedicated REAL-audio domain stream, independent of activity/ADASYN sampling.
# multi_home:          10 samples per source home -> 50/domain batch for 5 sources.
# multi_source_target: 10 per source home + 10 target -> 60/domain batch.
# Synthetic SMOTE/ADASYN episodes never enter this domain stream.
USE_HOME_BALANCED_DOMAIN_LOADER = True
DOMAIN_SAMPLES_PER_HOME = 10
DOMAIN_BATCHES_PER_CYCLE_MIN = 8

# Legacy binary source-vs-target loader only.
DOMAIN_BATCH_SIZE = 64
DOMAIN_WINDOW_LEN = 50
DOMAIN_WINDOW_MIN_LEN = 5

# HOMES = [
#     {
#         "name":          "koyama_an_2",
#         "labeled_tsv": "./realhome_label_project/episode_clean/koyama_an_2_labeled.tsv",
#         #"labeled_tsv":   "./realhome_label_project/koyama_an_2_labeled.tsv",
#         "unlabeled_tsv": "./realhome_label_project/koyama_an_2_unlabeled.tsv",
#         "batch_idx_tsv": "./realhome_label_project/clap-vectors/output/batch_index_koyama_an_2.tsv",
#         "db_folder":     "./realhome_label_project/clap-vectors/output/database-DESKTOP-0D3M52K koyama-an 2回目.db",
#     },
#     {
#         "name":          "takaki_teraoka_2",
#         "labeled_tsv": "./realhome_label_project/episode_clean/takaki_teraoka_2_labeled.tsv",
#         #"labeled_tsv":   "./realhome_label_project/takaki_teraoka_2_labeled.tsv",
#         "unlabeled_tsv": "./realhome_label_project/takaki_teraoka_2_unlabeled.tsv",
#         "batch_idx_tsv": "./realhome_label_project/clap-vectors/output/batch_index_takaki_teraoka_2.tsv",
#         "db_folder":     "./realhome_label_project/clap-vectors/output/database-dt-n06ieir takaki teraoka 2回目.db",
#     },
#     {
#         "name":          "sqlite_JAXA_MINISFORUM",
#         "labeled_tsv": "./realhome_label_project/episode_clean/sqlite_JAXA_MINISFORUM_labeled.tsv",
#         #"labeled_tsv":   "./realhome_label_project/sqlite_JAXA_MINISFORUM_labeled.tsv",
#         "unlabeled_tsv": "./realhome_label_project/sqlite_JAXA_MINISFORUM_unlabeled.tsv",
#         "batch_idx_tsv": "./realhome_label_project/clap-vectors/output/batch_index_sqlite_JAXA_MINISFORUM.tsv",
#         "db_folder":     "./realhome_label_project/clap-vectors/output/sqlite JAXA MINISFORUM銀.db",
#     },
#     {
#         "name":          "engawa_MINISFORUM_1",
#         "labeled_tsv": "./realhome_label_project/episode_clean/engawa_MINISFORUM_1_labeled.tsv",
#         #"labeled_tsv":   "./realhome_label_project/engawa_MINISFORUM_1_labeled.tsv",
#         "unlabeled_tsv": "./realhome_label_project/engawa_MINISFORUM_1_unlabeled.tsv",
#         "batch_idx_tsv": "./realhome_label_project/clap-vectors/output/batch_index_engawa_MINISFORUM_1.tsv",
#         "db_folder":     "./realhome_label_project/clap-vectors/output/database-engawa-MINISFORUM黒 1回目.db",
#     },
#
#     {
#         "name": "koyama_an_1",
#         "labeled_tsv": "./realhome_label_project/episode_clean/koyama_an_1_labeled.tsv",
#         #"labeled_tsv": "./realhome_label_project/koyama_an_1_labeled.tsv",
#         "unlabeled_tsv": "./realhome_label_project/koyama_an_1_unlabeled.tsv",
#         "batch_idx_tsv": "./realhome_label_project/clap-vectors/output/batch_index_koyama_an_1.tsv",
#         "db_folder": "./realhome_label_project/clap-vectors/output/database-DESKTOP-0D3M52K koyama-an 1回目.db",
#     },
#     {
#         "name": "takaki_teraoka_1",
#         "labeled_tsv": "./realhome_label_project/episode_clean/takaki_teraoka_1_labeled.tsv",
#         #"labeled_tsv": "./realhome_label_project/takaki_teraoka_1_labeled.tsv",
#         "unlabeled_tsv": "./realhome_label_project/takaki_teraoka_1_unlabeled.tsv",
#         "batch_idx_tsv": "./realhome_label_project/clap-vectors/output/batch_index_takaki_teraoka_1.tsv",
#         "db_folder": "./realhome_label_project/clap-vectors/output/database-dt-n06ieir takaki teraoka 1回目.db",
#     },
# ]
#HOMES = [
#     {
#         "name":          "3_home1",
#         "labeled_tsv": "./realhome_label_project/episode_clean/koyama_an_2_labeled.tsv",
#         #"labeled_tsv":   "./realhome_label_project/koyama_an_2_labeled.tsv",
#         "unlabeled_tsv": "./realhome_label_project/koyama_an_2_unlabeled.tsv",
#         "batch_idx_tsv": "./realhome_label_project/clap-vectors/output/batch_index_koyama_an_2.tsv",
#         "db_folder":     "./realhome_label_project/clap-vectors/clap_vectors_data/database-home1.db",
#     },
#     {
#         "name":          "6_home2",
#         "labeled_tsv": "./realhome_label_project/episode_clean/takaki_teraoka_2_labeled.tsv",
#         "unlabeled_tsv": "./realhome_label_project/takaki_teraoka_2_unlabeled.tsv",
#         "batch_idx_tsv": "./realhome_label_project/clap-vectors/output/batch_index_takaki_teraoka_2.tsv",
#         "db_folder": "./realhome_label_project/clap-vectors/clap_vectors_data/database-home2.db",
#     },
#     {
#         "name":          "4_home3",
#     "labeled_tsv": "./realhome_label_project/episode_clean/sqlite_JAXA_MINISFORUM_labeled.tsv",
#     "unlabeled_tsv": "./realhome_label_project/sqlite_JAXA_MINISFORUM_unlabeled.tsv",
#     "batch_idx_tsv": "./realhome_label_project/clap-vectors/output/batch_index_sqlite_JAXA_MINISFORUM.tsv",
#     "db_folder":     "./realhome_label_project/clap-vectors/clap_vectors_data/database-home3.db",
#     },
#     {
#         "name":          "1_home4",
#     "labeled_tsv": "./realhome_label_project/episode_clean/engawa_MINISFORUM_1_labeled.tsv",
#     "unlabeled_tsv": "./realhome_label_project/engawa_MINISFORUM_1_unlabeled.tsv",
#     "batch_idx_tsv": "./realhome_label_project/clap-vectors/output/batch_index_engawa_MINISFORUM_1.tsv",
#     "db_folder":     "./realhome_label_project/clap-vectors/clap_vectors_data/database-home4.db",
#             },
#
#     {
#         "name": "2_home5",
#         "labeled_tsv": "./realhome_label_project/episode_clean/koyama_an_1_labeled.tsv",
#         "unlabeled_tsv": "./realhome_label_project/koyama_an_1_unlabeled.tsv",
#         "batch_idx_tsv": "./realhome_label_project/clap-vectors/output/batch_index_koyama_an_1.tsv",
#         "db_folder": "./realhome_label_project/clap-vectors/clap_vectors_data/database-home5.db",
#             },
#     {
#         "name": "5_home6",
#             "labeled_tsv": "./realhome_label_project/episode_clean/takaki_teraoka_1_labeled.tsv",
#             #"labeled_tsv": "./realhome_label_project/takaki_teraoka_1_labeled.tsv",
#             "unlabeled_tsv": "./realhome_label_project/takaki_teraoka_1_unlabeled.tsv",
#             "batch_idx_tsv": "./realhome_label_project/clap-vectors/output/batch_index_takaki_teraoka_1.tsv",
#             "db_folder": "./realhome_label_project/clap-vectors/clap_vectors_data/database-home6.db",
#         },
# ]
HOMES = [
{"name": "3_home_1", "labeled_tsv": "dataset/metadata/home1_labeled.tsv",
 "unlabeled_tsv": "dataset/metadata/home1_unlabeled.tsv",
 "batch_idx_tsv": "dataset/metadata/batch_index_home1.tsv", "db_folder": "dataset/data/database-home1.db"},
{"name": "6_home_2", "labeled_tsv": "dataset/metadata/home2_labeled.tsv",
 "unlabeled_tsv": "dataset/metadata/home2_unlabeled.tsv",
 "batch_idx_tsv": "dataset/metadata/batch_index_home2.tsv", "db_folder": "dataset/data/database-home2.db"},
{"name": "4_home_3", "labeled_tsv": "dataset/metadata/home3_labeled.tsv",
 "unlabeled_tsv": "dataset/metadata/home3_unlabeled.tsv",
 "batch_idx_tsv": "dataset/metadata/batch_index_home3.tsv", "db_folder": "dataset/data/database-home3.db"},
{"name": "1_home_4", "labeled_tsv": "dataset/metadata/home4_labeled.tsv",
 "unlabeled_tsv": "dataset/metadata/home4_unlabeled.tsv",
 "batch_idx_tsv": "dataset/metadata/batch_index_home4.tsv", "db_folder": "dataset/data/database-home4.db"},
{"name": "2_home_5", "labeled_tsv": "dataset/metadata/home5_labeled.tsv",
 "unlabeled_tsv": "dataset/metadata/home5_unlabeled.tsv",
 "batch_idx_tsv": "dataset/metadata/batch_index_home5.tsv", "db_folder": "dataset/data/database-home5.db"},
{"name": "5_home_6", "labeled_tsv": "dataset/metadata/home6_labeled.tsv",
 "unlabeled_tsv": "dataset/metadata/home6_unlabeled.tsv",
 "batch_idx_tsv": "dataset/metadata/batch_index_home6.tsv", "db_folder": "dataset/data/database-home6.db"},
]

# Run only specific folds, or None = all homes.
RUN_ONLY_HELDOUT = None

# Training: fixed-length, final-epoch evaluation.
EPOCHS = 50
LR = 1e-3
BATCH_SIZE = 64
NUM_WORKERS = 16
SCHED_STEP = 30
SCHED_GAMMA = 0.5
SEED = 42 #9081 #13 #42

# Acoustic encoder + BiGRU.
CLAP_DIM = 1024
MLP_HIDDEN1 = 256
MLP_HIDDEN2 = 64
GRU_HIDDEN = 128
GRU_LAYERS = 2
DROPOUT = 0.5
DOMAIN_FEAT_DIM = GRU_HIDDEN * 2

# Class balance / resampling.
WEIGHT_CAP = 5.0
OVERSAMPLE_FACTOR = 10
OVERSAMPLE_METHOD = "adasyn"   # "duplicate" | "smote" | "adasyn" | "smotetomek" | "smoteenn"
SMOTE_K_NEIGHBORS_DEFAULT = 5
RESAMPLE_TARGET_MULTIPLIER = 1.0
ALSO_WEIGHT_SAMPLE_AFTER_SYNTH = False
EPISODE_TENSOR_CACHE_SIZE = 4

# Consecutive-batch deduplication.
DEDUP_CHAINS = True
DEDUP_CHAIN_KEEP = "longest"      # "longest" | "first"
DEDUP_CLASSES = None               # None = all classes
if DEDUP_CHAIN_KEEP not in ("longest", "first"):
    raise ValueError('DEDUP_CHAIN_KEEP must be "longest" or "first"')

CLASSES = ["cooking", "eating", "going_out", "go_to_toilet", "taking_bath", "washing"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}
N_CLASSES = len(CLASSES)
IGNORE_IDX = -100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"device={DEVICE}  classes={N_CLASSES}  homes={len(HOMES)}")
print("Model: CLAP 1024 -> MLP 256 -> 64 -> BiGRU -> single activity classifier")
print(f"Ablation={ABLATION}  domain_mode={DOMAIN_ADAPT_MODE if ABLATION != 'baseline' else 'n/a'}")


# ============================================================
# LOAD / EPISODE BUILDING
# ============================================================
def step1_load_tsvs(labeled_tsv, batch_idx_tsv, home_name):
    labeled = pd.read_csv(labeled_tsv, sep="\t")
    batch_idx = pd.read_csv(batch_idx_tsv, sep="\t")

    # Keep the original join as a schema/integrity check, even though the
    # cleaned acoustic-only model does not consume time features.
    if "date" in batch_idx.columns and batch_idx["date"].dtype == object and len(batch_idx):
        if "-" in str(batch_idx["date"].iloc[0]):
            batch_idx["date"] = batch_idx["date"].str.replace("-", "", regex=False).astype(int)
    if "time" in batch_idx.columns:
        batch_idx = batch_idx.rename(columns={"time": "start_time_bidx"})

    join_cols = [c for c in ["batch_no", "index"] if c in batch_idx.columns and c in labeled.columns]
    if len(join_cols) != 2:
        raise ValueError(f"{home_name}: labeled/batch-index TSVs must contain batch_no and index")

    merged = labeled.merge(
        batch_idx[[c for c in ["batch_no", "index", "date", "start_time_bidx"] if c in batch_idx.columns]],
        on=["batch_no", "index"], how="left", suffixes=("", "_b"),
    )
    merged["home"] = home_name
    print(f"    [{home_name}] rows={len(merged):,}  batches={merged['batch_no'].nunique()}")
    return merged


def step2_resolve_multilabel(df):
    before = len(df)
    sort_cols = ["home", "batch_no", "index"]
    ascending = [True, True, True]
    if "onset" in df.columns:
        sort_cols.append("onset")
        ascending.append(False)
    df = df.sort_values(sort_cols, ascending=ascending)
    df = df.drop_duplicates(subset=["home", "batch_no", "index"], keep="first")
    df = df.sort_values(["home", "batch_no", "index"]).reset_index(drop=True)
    removed = before - len(df)
    if removed:
        print(f"    removed {removed} multilabel rows")
    return df


def step3_encode_labels(df):
    df = df.copy()
    df["label_idx"] = df["event_label"].map(CLASS_TO_IDX).fillna(IGNORE_IDX).astype(int)
    return df


def build_episodes(labeled_df, apply_dedup=True):
    """One sample = one contiguous same-label run inside one batch file."""
    df = labeled_df.sort_values(["home", "batch_no", "index"]).reset_index(drop=True)
    grp = df.groupby(["home", "batch_no"])
    label_changed = grp["label_idx"].apply(lambda s: s != s.shift()).reset_index(drop=True)
    index_gap = grp["index"].apply(lambda s: (s - s.shift()) != 1).reset_index(drop=True)
    df["_run_id"] = (label_changed | index_gap).cumsum()

    episodes = []
    n_sessions = df.groupby(["home", "batch_no"]).ngroups
    for (home, bno, _rid), sub in df.groupby(["home", "batch_no", "_run_id"], sort=False):
        label = int(sub["label_idx"].iloc[0])
        if label == IGNORE_IDX:
            continue
        indices = sub["index"].values.astype(np.int64)
        episodes.append({
            "home": home,
            "batch_no": bno,
            "label_idx": label,
            "indices": indices,
            "eid": f"{home}|{bno}|{label}|{indices[0]}-{indices[-1]}",
        })

    if not episodes:
        raise RuntimeError("No labeled episodes were built")
    lengths = [len(e["indices"]) for e in episodes]
    print(f"    Built {len(episodes)} labeled episodes from {n_sessions} sessions "
          f"(len min/median/max = {min(lengths)}/{int(np.median(lengths))}/{max(lengths)} steps)")

    if apply_dedup and DEDUP_CHAINS:
        episodes = _dedup_consecutive_chains(episodes)
    return episodes


def _dedup_consecutive_chains(episodes):
    def batch_num(bno):
        m = re.search(r"(\d+)", str(bno))
        return int(m.group(1)) if m else -1

    active = (set(range(len(CLASSES))) if DEDUP_CLASSES is None
              else {CLASS_TO_IDX[c] for c in DEDUP_CLASSES})
    by_key = defaultdict(list)
    passthrough = []
    for e in episodes:
        if e["label_idx"] in active:
            by_key[(e["home"], e["label_idx"])].append(e)
        else:
            passthrough.append(e)

    kept = list(passthrough)
    before = collections.Counter(e["label_idx"] for e in episodes)
    for (_home, _lab), eps in by_key.items():
        eps = sorted(eps, key=lambda e: batch_num(e["batch_no"]))
        chains, chain = [], [eps[0]]
        for a, b in zip(eps, eps[1:]):
            if batch_num(b["batch_no"]) == batch_num(a["batch_no"]) + 1:
                chain.append(b)
            else:
                chains.append(chain)
                chain = [b]
        chains.append(chain)
        for c in chains:
            if len(c) == 1 or DEDUP_CHAIN_KEEP == "first":
                kept.append(c[0])
            else:
                kept.append(max(c, key=lambda e: len(e["indices"])))

    after = collections.Counter(e["label_idx"] for e in kept)
    scope = "all classes" if DEDUP_CLASSES is None else ", ".join(sorted(DEDUP_CLASSES))
    print(f"    [DEDUP/{DEDUP_CHAIN_KEEP}] scope: {scope}")
    print(f"    [DEDUP/{DEDUP_CHAIN_KEEP}] {len(episodes)} episodes -> {len(kept)} real events "
          f"({len(episodes)-len(kept)} duplicate views dropped)")
    for c in range(N_CLASSES):
        b, a = before.get(c, 0), after.get(c, 0)
        print(f"      {CLASSES[c]:<14} {b:>5} -> {a:>5}")
    return kept


# ============================================================
# UNLABELED TARGET WINDOWS FOR DOMAIN ADAPTATION
# ============================================================
def step1_load_unlabeled_tsv(unlabeled_tsv_path, batch_idx_tsv_path, home_name):
    try:
        raw = pd.read_csv(unlabeled_tsv_path, sep="\t")
    except (FileNotFoundError, OSError) as e:
        print(f"    [unlabeled] {home_name}: could not read '{unlabeled_tsv_path}' ({e})")
        return None

    required = {"batch_no", "index"}
    missing = required - set(raw.columns)
    if missing:
        print(f"    [unlabeled] {home_name}: missing {sorted(missing)}; skipped")
        return None

    # Join only to verify that the row maps to a known session/index.
    try:
        batch_idx = pd.read_csv(batch_idx_tsv_path, sep="\t")
        merged = raw.merge(batch_idx[["batch_no", "index"]].drop_duplicates(),
                           on=["batch_no", "index"], how="inner")
    except Exception as e:
        print(f"    [unlabeled] {home_name}: batch-index join failed ({e}); skipped")
        return None

    merged["home"] = home_name
    print(f"    [unlabeled] {home_name}: {len(merged):,} rows loaded")
    return merged


def build_unlabeled_windows(df, window_len=DOMAIN_WINDOW_LEN,
                             min_len=DOMAIN_WINDOW_MIN_LEN):
    if df is None or len(df) == 0:
        return []
    df = df.sort_values(["home", "batch_no", "index"]).reset_index(drop=True)
    grp = df.groupby(["home", "batch_no"])
    index_gap = grp["index"].apply(lambda s: (s - s.shift()) != 1).reset_index(drop=True)
    df["_run_id"] = index_gap.cumsum()

    windows = []
    for (home, bno, _rid), sub in df.groupby(["home", "batch_no", "_run_id"], sort=False):
        idxs = sub["index"].values.astype(np.int64)
        for start in range(0, len(idxs), window_len):
            chunk = idxs[start:start + window_len]
            if len(chunk) >= min_len:
                windows.append({"home": home, "batch_no": bno, "indices": chunk})
    return windows


# ============================================================
# RESAMPLING
# ============================================================
def _pool_episode_features(episodes, db_folder_map, cache_size=8):
    cache = OrderedDict()

    def load_tensor(home, bno):
        key = (home, bno)
        if key in cache:
            cache.move_to_end(key)
            return cache[key]
        with open(Path(db_folder_map[home]) / bno, "rb") as f:
            bd = torch.load(f, weights_only=False)
        cache[key] = bd["tensor"]
        if len(cache) > cache_size:
            cache.popitem(last=False)
        return cache[key]

    feats = np.zeros((len(episodes), CLAP_DIM), dtype=np.float32)
    for i, ep in enumerate(episodes):
        full_tensor = load_tensor(ep["home"], ep["batch_no"])
        feats[i] = np.asarray(full_tensor[ep["indices"]]).mean(axis=0)
    return feats


def build_resampled_episodes(train_eps, db_folder_map, method,
                             k_neighbors=SMOTE_K_NEIGHBORS_DEFAULT):
    try:
        from imblearn.over_sampling import SMOTE, ADASYN
        from imblearn.combine import SMOTETomek, SMOTEENN
    except ImportError as e:
        raise ImportError("Install imbalanced-learn: pip install imbalanced-learn") from e

    is_combined = method in ("smotetomek", "smoteenn")
    y = np.array([ep["label_idx"] for ep in train_eps], dtype=np.int64)
    counts = collections.Counter(y.tolist())
    majority_count = max(counts.values())

    resamplable = {c: n for c, n in counts.items() if n >= 2}
    target_count = int(round(majority_count * RESAMPLE_TARGET_MULTIPLIER))
    sampling_strategy = {c: target_count for c, n in resamplable.items() if n < target_count}
    if not sampling_strategy:
        print(f"    [{method.upper()}] no resampling needed")
        return list(train_eps), []

    min_class_n = min(n for c, n in resamplable.items() if c in sampling_strategy)
    k = max(1, min(k_neighbors, min_class_n - 1))
    print(f"    [{method.upper()}] pooling {len(train_eps)} real episodes -> {CLAP_DIM}D")
    X = _pool_episode_features(train_eps, db_folder_map)

    if method == "smote":
        sampler = SMOTE(sampling_strategy=sampling_strategy, random_state=SEED, k_neighbors=k)
    elif method == "adasyn":
        sampler = ADASYN(sampling_strategy=sampling_strategy, random_state=SEED, n_neighbors=k)
    elif method == "smotetomek":
        sampler = SMOTETomek(
            sampling_strategy=sampling_strategy, random_state=SEED,
            smote=SMOTE(sampling_strategy=sampling_strategy, random_state=SEED, k_neighbors=k))
    elif method == "smoteenn":
        sampler = SMOTEENN(
            sampling_strategy=sampling_strategy, random_state=SEED,
            smote=SMOTE(sampling_strategy=sampling_strategy, random_state=SEED, k_neighbors=k))
    else:
        raise ValueError(f"Unknown resampling method: {method}")

    try:
        X_res, y_res = sampler.fit_resample(X, y)
    except ValueError as e:
        print(f"    [{method.upper()}] failed ({e}); using original episodes")
        return list(train_eps), []

    if not is_combined:
        n_real = len(X)
        if not np.allclose(X_res[:n_real], X):
            print(f"    [{method.upper()}] unexpected row order; using original episodes")
            return list(train_eps), []
        kept_real_eps = list(train_eps)
        synth_X, synth_y = X_res[n_real:], y_res[n_real:]
        n_dropped = 0
    else:
        orig_lookup = {}
        for i, row in enumerate(X):
            orig_lookup.setdefault(row.tobytes(), []).append(i)
        kept_idx, synth_rows, synth_labels = [], [], []
        used = collections.Counter()
        for row, lab in zip(X_res, y_res):
            key = row.tobytes()
            bucket = orig_lookup.get(key)
            if bucket is not None and used[key] < len(bucket):
                kept_idx.append(bucket[used[key]])
                used[key] += 1
            else:
                synth_rows.append(row)
                synth_labels.append(lab)
        kept_real_eps = [train_eps[i] for i in sorted(kept_idx)]
        synth_X = np.asarray(synth_rows, dtype=np.float32) if synth_rows else np.empty((0, CLAP_DIM), np.float32)
        synth_y = np.asarray(synth_labels, dtype=np.int64) if synth_labels else np.empty((0,), np.int64)
        n_dropped = len(train_eps) - len(kept_real_eps)

    synthetic_eps = []
    for i, (vec, label) in enumerate(zip(synth_X, synth_y)):
        synthetic_eps.append({
            "home": "SYNTHETIC",
            "batch_no": None,
            "label_idx": int(label),
            "indices": None,
            "eid": f"SYNTH|{method}|{int(label)}|{i}",
            "synthetic": True,
            "synthetic_vec": vec.astype(np.float32),
        })

    final_counts = collections.Counter(
        [ep["label_idx"] for ep in kept_real_eps] + [ep["label_idx"] for ep in synthetic_eps])
    print(f"    [{method.upper()}] +{len(synthetic_eps)} synthetic, -{n_dropped} real")
    print("    " + ", ".join(
        f"{CLASSES[c]}={counts.get(c,0)}->{final_counts.get(c,0)}" for c in range(N_CLASSES)))
    return kept_real_eps, synthetic_eps


# ============================================================
# DOMAIN ADAPTATION
# ============================================================
class _GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd):
    return _GradientReversalFn.apply(x, lambd)


class DomainDiscriminator(nn.Module):
    def __init__(self, in_dim, out_dim=1, hidden=DOMAIN_DISC_HIDDEN):
        super().__init__()
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        out = self.net(x)
        return out.squeeze(-1) if self.out_dim == 1 else out


def make_local_home_id_vector(eids, home_to_local_idx, device):
    ids = []
    for eid in eids:
        home_name = str(eid).split("|")[0]
        ids.append(home_to_local_idx.get(home_name, -1))
    return torch.tensor(ids, dtype=torch.long, device=device)


def make_domain_input(feat_pooled, activity_probs, ablation):
    if ablation == "dann":
        return feat_pooled
    b = feat_pooled.shape[0]
    outer = torch.bmm(feat_pooled.unsqueeze(2), activity_probs.unsqueeze(1))
    return outer.view(b, -1)


def entropy_conditioning_weight(activity_probs, eps=1e-8):
    entropy = -(activity_probs * torch.log(activity_probs + eps)).sum(dim=1)
    weight = 1.0 + torch.exp(-entropy)
    return weight / (weight.mean() + eps)


# ============================================================
# MODEL -- 2-LAYER MLP (256 -> 64) -> BiGRU -> ONE CLASSIFIER ONLY
# ============================================================
def masked_mean(x, mask):
    m = mask.unsqueeze(-1).to(x.dtype)
    return (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


class MLPBiGRU(nn.Module):
    """CLAP 1024 -> 2-layer MLP (256 -> 64) -> BiGRU -> pooled activity logits.

    There is NO separate MLP classifier head. The MLP is the feature extractor
    feeding the BiGRU, and only the BiGRU output is classified/reported.
    """
    def __init__(self, n_classes=N_CLASSES, gru_hidden=GRU_HIDDEN,
                 gru_layers=GRU_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(CLAP_DIM, MLP_HIDDEN1),
            nn.LayerNorm(MLP_HIDDEN1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(MLP_HIDDEN1, MLP_HIDDEN2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.bigru = nn.GRU(
            input_size=MLP_HIDDEN2,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(gru_hidden * 2, n_classes)

    def forward(self, x, mask=None):
        b, t, _ = x.shape
        if mask is None:
            mask = torch.ones(b, t, dtype=torch.bool, device=x.device)
        feat = self.mlp(x)
        out, _ = self.bigru(feat)
        feat_pooled = masked_mean(out, mask)
        logits = self.classifier(feat_pooled)
        return logits, feat_pooled


# ============================================================
# DATASETS
# ============================================================
class EpisodeDataset(Dataset):
    def __init__(self, episodes, db_folder_map, cache_size=EPISODE_TENSOR_CACHE_SIZE):
        self.episodes = episodes
        self.db_folder_map = db_folder_map
        self.cache_size = cache_size
        self._cache = OrderedDict()
        print(f"    Dataset: {len(self.episodes)} episodes")

    def __len__(self):
        return len(self.episodes)

    def _load_tensor(self, home, bno):
        key = (home, bno)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        pt_path = Path(self.db_folder_map[home]) / bno
        if not pt_path.exists():
            raise FileNotFoundError(f"Missing: {pt_path}")
        with open(pt_path, "rb") as f:
            bd = torch.load(f, weights_only=False)
        tensor = bd["tensor"]
        self._cache[key] = tensor
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return tensor

    def __getitem__(self, i):
        ep = self.episodes[i]
        label = torch.tensor(ep["label_idx"], dtype=torch.long)
        if ep.get("synthetic"):
            tensor = torch.tensor(ep["synthetic_vec"], dtype=torch.float32).unsqueeze(0)
        else:
            full_tensor = self._load_tensor(ep["home"], ep["batch_no"])
            tensor = torch.tensor(full_tensor[ep["indices"]], dtype=torch.float32)
        return tensor, label, ep["eid"]


def collate_fn(batch):
    tensors, labels, eids = zip(*batch)
    max_len = max(t.shape[0] for t in tensors)

    def pad2d(x):
        l, d = x.shape
        return x if l == max_len else torch.cat([x, torch.zeros(max_len-l, d, dtype=x.dtype)], dim=0)

    tensors_p = torch.stack([pad2d(t) for t in tensors])
    labels_t = torch.stack(labels)
    mask = torch.zeros(len(tensors), max_len, dtype=torch.bool)
    for j, t in enumerate(tensors):
        mask[j, :t.shape[0]] = True
    return tensors_p, labels_t, mask, eids


class DomainDataset(Dataset):
    def __init__(self, items, db_folder_map, cache_size=EPISODE_TENSOR_CACHE_SIZE):
        self.items = items
        self.db_folder_map = db_folder_map
        self.cache_size = cache_size
        self._cache = OrderedDict()
        print(f"    DomainDataset: {len(self.items)} windows")

    def __len__(self):
        return len(self.items)

    def _load_tensor(self, home, bno):
        key = (home, bno)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        pt_path = Path(self.db_folder_map[home]) / bno
        if not pt_path.exists():
            raise FileNotFoundError(f"Missing: {pt_path}")
        with open(pt_path, "rb") as f:
            bd = torch.load(f, weights_only=False)
        tensor = bd["tensor"]
        self._cache[key] = tensor
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return tensor

    def __getitem__(self, i):
        item = self.items[i]
        full_tensor = self._load_tensor(item["home"], item["batch_no"])
        return torch.tensor(full_tensor[item["indices"]], dtype=torch.float32)


def domain_collate_fn(batch):
    max_len = max(t.shape[0] for t in batch)

    def pad2d(x):
        l, d = x.shape
        return x if l == max_len else torch.cat([x, torch.zeros(max_len-l, d, dtype=x.dtype)], dim=0)

    tensors_p = torch.stack([pad2d(t) for t in batch])
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for j, t in enumerate(batch):
        mask[j, :t.shape[0]] = True
    return tensors_p, mask


class BalancedDomainDataset(Dataset):
    """Real-audio items used only by the balanced domain-adversarial stream."""
    def __init__(self, items, db_folder_map, cache_size=EPISODE_TENSOR_CACHE_SIZE):
        self.items = items
        self.db_folder_map = db_folder_map
        self.cache_size = cache_size
        self._cache = OrderedDict()
        print(f"    BalancedDomainDataset: {len(self.items)} windows")

    def __len__(self):
        return len(self.items)

    def _load_tensor(self, home, bno):
        key = (home, bno)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        pt_path = Path(self.db_folder_map[home]) / bno
        if not pt_path.exists():
            raise FileNotFoundError(f"Missing: {pt_path}")
        with open(pt_path, "rb") as f:
            bd = torch.load(f, weights_only=False)
        tensor = bd["tensor"]
        self._cache[key] = tensor
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return tensor

    def __getitem__(self, i):
        item = self.items[i]
        full_tensor = self._load_tensor(item["home"], item["batch_no"])
        tensor = torch.tensor(full_tensor[item["indices"]], dtype=torch.float32)
        return tensor, int(item["domain_id"])


class BalancedDomainBatchSampler:
    """Exact per-domain quota in every domain minibatch."""
    def __init__(self, items, group_batch_counts, min_batches=8, seed=SEED):
        self.group_batch_counts = OrderedDict(group_batch_counts)
        self.group_to_indices = OrderedDict()
        for group in self.group_batch_counts:
            idxs = [i for i, item in enumerate(items) if item.get("balance_group") == group]
            if not idxs:
                raise ValueError(f"No domain windows available for balance_group={group!r}")
            self.group_to_indices[group] = np.asarray(idxs, dtype=np.int64)
        coverage = [
            max(1, math.ceil(len(self.group_to_indices[g]) / max(n, 1)))
            for g, n in self.group_batch_counts.items()
        ]
        self.n_batches = max(int(min_batches), max(coverage))
        self.rng = np.random.RandomState(seed)

    def __iter__(self):
        for _ in range(self.n_batches):
            batch = []
            for group, per_batch in self.group_batch_counts.items():
                pool = self.group_to_indices[group]
                picks = self.rng.choice(pool, size=per_batch, replace=(len(pool) < per_batch))
                batch.extend(int(i) for i in picks)
            self.rng.shuffle(batch)
            yield batch

    def __len__(self):
        return self.n_batches


def balanced_domain_collate_fn(batch):
    tensors, domain_ids = zip(*batch)
    max_len = max(t.shape[0] for t in tensors)

    def pad2d(x):
        l, d = x.shape
        return x if l == max_len else torch.cat([x, torch.zeros(max_len-l, d, dtype=x.dtype)], dim=0)

    tensors_p = torch.stack([pad2d(t) for t in tensors])
    mask = torch.zeros(len(tensors), max_len, dtype=torch.bool)
    for j, t in enumerate(tensors):
        mask[j, :t.shape[0]] = True
    return tensors_p, mask, torch.tensor(domain_ids, dtype=torch.long)


def make_cycled_iter(loader):
    while True:
        for batch in loader:
            yield batch


# ============================================================
# LOSS / SAMPLING
# ============================================================
def compute_class_weights(episodes, classes, cap=WEIGHT_CAP):
    counts = np.zeros(len(classes), dtype=np.float32)
    for ep in episodes:
        counts[ep["label_idx"]] += 1
    counts = np.clip(counts, 1, None)
    dominant_idx = int(np.argmax(counts))
    print("    Train episode counts: " + ", ".join(f"{c}={int(n)}" for c, n in zip(classes, counts)))
    sqrt_w = np.sqrt(counts.max() / counts)
    sqrt_w = sqrt_w / sqrt_w[dominant_idx]
    sqrt_w = np.clip(sqrt_w, None, cap)
    return torch.tensor(sqrt_w, dtype=torch.float32)


class CELoss(nn.Module):
    def __init__(self, weight=None, ignore_index=IGNORE_IDX):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction="none")
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        valid = targets != self.ignore_index
        if valid.sum() == 0:
            return logits.sum() * 0.0
        return self.criterion(logits, targets)[valid].mean()


def compute_loss(gru_logits, labels, ce_loss):
    if (labels != IGNORE_IDX).sum() == 0:
        return None
    loss = ce_loss(gru_logits, labels)
    if torch.isnan(loss):
        return None
    return loss


def make_batch_sampler(dataset, cw):
    weights = []
    max_w = float(max(cw))
    for ep in dataset.episodes:
        base_w = float(cw[ep["label_idx"]])
        scaled = 1.0 + (OVERSAMPLE_FACTOR - 1.0) * (base_w - 1.0) / (max_w - 1.0 + 1e-9)
        weights.append(scaled)
    g = torch.Generator()
    g.manual_seed(SEED)
    return torch.utils.data.WeightedRandomSampler(weights, len(weights), replacement=True, generator=g)


# ============================================================
# TRAIN / EVALUATE
# ============================================================
def train_one_epoch(model, loader, optimizer, ce_loss,
                    domain_disc=None, target_iter=None, balanced_domain_iter=None,
                    lambda_domain=0.0, domain_mode="multi_home", home_to_local_idx=None):
    model.train()
    if domain_disc is not None:
        domain_disc.train()

    total_loss = 0.0
    total_task = 0.0
    total_domain = 0.0
    n = 0
    domain_correct = 0
    domain_total = 0

    for tensors, labels, mask, eids in loader:
        tensors = tensors.to(DEVICE)
        labels = labels.to(DEVICE)
        mask = mask.to(DEVICE)

        optimizer.zero_grad()
        gru_logits, feat_src = model(tensors, mask)
        task_loss = compute_loss(gru_logits, labels, ce_loss)
        if task_loss is None:
            continue

        loss = task_loss
        domain_loss_val = 0.0

        if domain_disc is not None and lambda_domain > 0.0:
            # multi_home / multi_source_target use a separate REAL-audio balanced
            # domain stream, so ADASYN and activity-class imbalance cannot distort D.
            if (USE_HOME_BALANCED_DOMAIN_LOADER
                    and domain_mode in ("multi_home", "multi_source_target")
                    and balanced_domain_iter is not None):
                d_x, d_mask, d_targets = next(balanced_domain_iter)
                d_x = d_x.to(DEVICE)
                d_mask = d_mask.to(DEVICE)
                d_targets = d_targets.to(DEVICE)
                d_activity_logits, d_feat = model(d_x, d_mask)
                d_probs = F.softmax(d_activity_logits, dim=-1)
                d_in = make_domain_input(d_feat, d_probs, ABLATION)
                d_logits = domain_disc(grad_reverse(d_in, lambda_domain))
                if ABLATION == "cdan_e":
                    w = entropy_conditioning_weight(d_probs)
                    per = F.cross_entropy(d_logits, d_targets, reduction="none")
                    domain_loss = (w * per).mean()
                else:
                    domain_loss = F.cross_entropy(d_logits, d_targets)
                loss = loss + domain_loss
                domain_loss_val = domain_loss.item()
                with torch.no_grad():
                    domain_correct += (d_logits.argmax(1) == d_targets).sum().item()
                    domain_total += d_targets.numel()

            else:
                src_probs = F.softmax(gru_logits, dim=-1)  # legacy/fallback path

                if domain_mode == "multi_home":
                    local_ids = make_local_home_id_vector(eids, home_to_local_idx, DEVICE)
                    valid_dom = local_ids >= 0
                    if valid_dom.any():
                        domain_in = make_domain_input(feat_src[valid_dom], src_probs[valid_dom], ABLATION)
                        domain_logits = domain_disc(grad_reverse(domain_in, lambda_domain))
                        domain_targets = local_ids[valid_dom]
                        if ABLATION == "cdan_e":
                            w = entropy_conditioning_weight(src_probs[valid_dom])
                            per = F.cross_entropy(domain_logits, domain_targets, reduction="none")
                            domain_loss = (w * per).mean()
                        else:
                            domain_loss = F.cross_entropy(domain_logits, domain_targets)
                        loss = loss + domain_loss
                        domain_loss_val = domain_loss.item()
                        with torch.no_grad():
                            domain_correct += (domain_logits.argmax(1) == domain_targets).sum().item()
                            domain_total += domain_targets.numel()

                elif domain_mode == "multi_source_target":
                    if target_iter is not None:
                        local_ids = make_local_home_id_vector(eids, home_to_local_idx, DEVICE)
                        valid_src_dom = local_ids >= 0
                        t_tensors, t_mask = next(target_iter)
                        t_tensors = t_tensors.to(DEVICE)
                        t_mask = t_mask.to(DEVICE)
                        t_gru_logits, feat_tgt = model(t_tensors, t_mask)
                        tgt_probs = F.softmax(t_gru_logits, dim=-1)

                        losses_dom = []
                        if valid_src_dom.any():
                            din_src = make_domain_input(feat_src[valid_src_dom], src_probs[valid_src_dom], ABLATION)
                            logit_src = domain_disc(grad_reverse(din_src, lambda_domain))
                            target_src = local_ids[valid_src_dom]
                            if ABLATION == "cdan_e":
                                w_src = entropy_conditioning_weight(src_probs[valid_src_dom])
                                per_src = F.cross_entropy(logit_src, target_src, reduction="none")
                                losses_dom.append((w_src * per_src).mean())
                            else:
                                losses_dom.append(F.cross_entropy(logit_src, target_src))
                            with torch.no_grad():
                                domain_correct += (logit_src.argmax(1) == target_src).sum().item()
                                domain_total += target_src.numel()

                        din_tgt = make_domain_input(feat_tgt, tgt_probs, ABLATION)
                        logit_tgt = domain_disc(grad_reverse(din_tgt, lambda_domain))
                        target_domain_id = len(home_to_local_idx)
                        target_tgt = torch.full((len(t_tensors),), target_domain_id,
                                                dtype=torch.long, device=DEVICE)
                        if ABLATION == "cdan_e":
                            w_tgt = entropy_conditioning_weight(tgt_probs)
                            per_tgt = F.cross_entropy(logit_tgt, target_tgt, reduction="none")
                            losses_dom.append((w_tgt * per_tgt).mean())
                        else:
                            losses_dom.append(F.cross_entropy(logit_tgt, target_tgt))
                        with torch.no_grad():
                            domain_correct += (logit_tgt.argmax(1) == target_tgt).sum().item()
                            domain_total += target_tgt.numel()

                        if losses_dom:
                            domain_loss = torch.stack(losses_dom).mean()
                            loss = loss + domain_loss
                            domain_loss_val = domain_loss.item()

                else:  # binary_source_target
                    if target_iter is not None:
                        t_tensors, t_mask = next(target_iter)
                        t_tensors = t_tensors.to(DEVICE)
                        t_mask = t_mask.to(DEVICE)
                        t_gru_logits, feat_tgt = model(t_tensors, t_mask)
                        tgt_probs = F.softmax(t_gru_logits, dim=-1)

                        din_src = make_domain_input(feat_src, src_probs, ABLATION)
                        din_tgt = make_domain_input(feat_tgt, tgt_probs, ABLATION)
                        logit_src = domain_disc(grad_reverse(din_src, lambda_domain))
                        logit_tgt = domain_disc(grad_reverse(din_tgt, lambda_domain))
                        y_src = torch.zeros(logit_src.shape[0], device=DEVICE)
                        y_tgt = torch.ones(logit_tgt.shape[0], device=DEVICE)
                        bce = F.binary_cross_entropy_with_logits

                        if ABLATION == "cdan_e":
                            w_src = entropy_conditioning_weight(src_probs)
                            w_tgt = entropy_conditioning_weight(tgt_probs)
                            domain_loss = (
                                w_src * bce(logit_src, y_src, reduction="none")
                            ).mean() + (
                                w_tgt * bce(logit_tgt, y_tgt, reduction="none")
                            ).mean()
                        else:
                            domain_loss = bce(logit_src, y_src) + bce(logit_tgt, y_tgt)

                        loss = loss + domain_loss
                        domain_loss_val = domain_loss.item()
                        with torch.no_grad():
                            pred_src = (torch.sigmoid(logit_src) > 0.5).float()
                            pred_tgt = (torch.sigmoid(logit_tgt) > 0.5).float()
                            domain_correct += (pred_src == y_src).sum().item()
                            domain_correct += (pred_tgt == y_tgt).sum().item()
                            domain_total += y_src.numel() + y_tgt.numel()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if domain_disc is not None:
            torch.nn.utils.clip_grad_norm_(domain_disc.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_task += task_loss.item()
        total_domain += domain_loss_val
        n += 1

    return {
        "total": total_loss / max(n, 1),
        "task": total_task / max(n, 1),
        "domain": total_domain / max(n, 1),
        "domain_acc": domain_correct / domain_total if domain_total else float("nan"),
    }


@torch.no_grad()
def evaluate(model, loader, ce_loss, return_preds=False):
    model.eval()
    pred_list, label_list = [], []
    total_loss = 0.0
    n = 0

    for tensors, labels, mask, _eids in loader:
        tensors = tensors.to(DEVICE)
        labels = labels.to(DEVICE)
        mask = mask.to(DEVICE)
        logits, _ = model(tensors, mask)
        loss = compute_loss(logits, labels, ce_loss)
        if loss is not None:
            total_loss += loss.item()
            n += 1

        real = labels != IGNORE_IDX
        if real.any():
            pred_list.append(logits.argmax(-1).cpu()[real.cpu()])
            label_list.append(labels.cpu()[real.cpu()])

    if not label_list:
        empty = np.array([])
        if return_preds:
            return float("nan"), 0.0, empty, empty
        return float("nan"), 0.0

    pred = torch.cat(pred_list).numpy()
    labels = torch.cat(label_list).numpy()
    macro_f1 = f1_score(labels, pred, average="macro", zero_division=0,
                        labels=list(range(N_CLASSES)))
    avg_loss = total_loss / max(n, 1)
    if return_preds:
        return avg_loss, macro_f1, pred, labels
    return avg_loss, macro_f1


# ============================================================
# RUN ONE LOHO FOLD
# ============================================================
def run_fold(held_out_home):
    """Train on all other homes; use held-out home for diagnostic F1 and final test."""
    train_homes_cfg = [h for h in HOMES if h["name"] != held_out_home]
    held_out_cfg = next(h for h in HOMES if h["name"] == held_out_home)
    train_home_names = [h["name"] for h in train_homes_cfg]
    fold_name = f"heldout_{held_out_home}"

    print("\n" + "█" * 70)
    print(f"LOHO FOLD — held-out home: {held_out_home}")
    print(f"train homes: {train_home_names}")
    print("█" * 70)

    set_seed(SEED)

    # Load labeled data.
    all_dfs = []
    db_folder_map = {}
    for h in HOMES:
        all_dfs.append(step1_load_tsvs(h["labeled_tsv"], h["batch_idx_tsv"], h["name"]))
        db_folder_map[h["name"]] = h["db_folder"]
    labeled_df = step3_encode_labels(step2_resolve_multilabel(pd.concat(all_dfs, ignore_index=True)))

    print("Building episodes...")
    episodes_all = build_episodes(labeled_df, apply_dedup=True)
    train_eps = [e for e in episodes_all if e["home"] in train_home_names]
    test_eps = [e for e in episodes_all if e["home"] == held_out_home]

    if OVERSAMPLE_METHOD in ("smote", "adasyn", "smotetomek", "smoteenn"):
        kept_real, synthetic = build_resampled_episodes(
            train_eps, db_folder_map, OVERSAMPLE_METHOD, SMOTE_K_NEIGHBORS_DEFAULT)
        train_eps_final = kept_real + synthetic
    else:
        train_eps_final = train_eps

    train_ds = EpisodeDataset(train_eps_final, db_folder_map)
    test_ds = EpisodeDataset(test_eps, db_folder_map)

    class_weights = compute_class_weights(train_eps_final, CLASSES, WEIGHT_CAP)
    cw = class_weights.numpy()
    g = torch.Generator()
    g.manual_seed(SEED)

    use_weighted_sampler = (OVERSAMPLE_METHOD == "duplicate") or ALSO_WEIGHT_SAMPLE_AFTER_SYNTH
    if use_weighted_sampler:
        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, sampler=make_batch_sampler(train_ds, cw),
            collate_fn=collate_fn, num_workers=NUM_WORKERS,
            worker_init_fn=seed_worker, generator=g, persistent_workers=False)
    else:
        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True,
            collate_fn=collate_fn, num_workers=NUM_WORKERS,
            worker_init_fn=seed_worker, generator=g, persistent_workers=False)

    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=NUM_WORKERS,
        worker_init_fn=seed_worker, generator=g, persistent_workers=False)

    # Domain setup. Balanced modes get an independent REAL-audio stream.
    target_iter = None
    balanced_domain_iter = None
    home_to_local_idx = None
    if ABLATION != "baseline":
        if DOMAIN_ADAPT_MODE in ("multi_home", "multi_source_target"):
            home_to_local_idx = {name: i for i, name in enumerate(train_home_names)}

        target_items = []
        if DOMAIN_ADAPT_MODE in ("binary_source_target", "multi_source_target"):
            # Preserve the existing target pool: labels are discarded for domain use.
            target_items = [
                {"home": e["home"], "batch_no": e["batch_no"], "indices": e["indices"]}
                for e in test_eps
            ]
            unlabeled_df = step1_load_unlabeled_tsv(
                held_out_cfg["unlabeled_tsv"], held_out_cfg["batch_idx_tsv"], held_out_home)
            unlabeled_windows = build_unlabeled_windows(unlabeled_df)
            target_items += unlabeled_windows
            print(f"    [domain] target windows: {len(test_eps)} labeled-audio + "
                  f"{len(unlabeled_windows)} unlabeled = {len(target_items)}")

        if (USE_HOME_BALANCED_DOMAIN_LOADER
                and DOMAIN_ADAPT_MODE in ("multi_home", "multi_source_target")):
            domain_items = []
            # REAL source episodes only: train_eps, never train_eps_final synthetics.
            for e in train_eps:
                home = e["home"]
                domain_items.append({
                    "home": home, "batch_no": e["batch_no"], "indices": e["indices"],
                    "domain_id": home_to_local_idx[home], "balance_group": home,
                })
            group_counts = OrderedDict(
                (home, DOMAIN_SAMPLES_PER_HOME) for home in train_home_names)

            if DOMAIN_ADAPT_MODE == "multi_source_target":
                if not target_items:
                    print("    [domain] WARNING: no target windows; balanced domain loss disabled")
                    domain_items = []
                else:
                    target_domain_id = len(train_home_names)
                    for item in target_items:
                        domain_items.append({**item, "domain_id": target_domain_id,
                                             "balance_group": "__target__"})
                    group_counts["__target__"] = DOMAIN_SAMPLES_PER_HOME

            if domain_items:
                domain_ds = BalancedDomainDataset(domain_items, db_folder_map)
                domain_batch_sampler = BalancedDomainBatchSampler(
                    domain_items, group_counts, min_batches=DOMAIN_BATCHES_PER_CYCLE_MIN, seed=SEED)
                domain_loader = DataLoader(
                    domain_ds, batch_sampler=domain_batch_sampler,
                    collate_fn=balanced_domain_collate_fn, num_workers=NUM_WORKERS,
                    worker_init_fn=seed_worker, persistent_workers=False)
                balanced_domain_iter = make_cycled_iter(domain_loader)
                print("    [domain] HOME-BALANCED batch=" + str(sum(group_counts.values())) +
                      " (" + ", ".join(f"{g}:{n}" for g, n in group_counts.items()) + ")")

        elif DOMAIN_ADAPT_MODE == "multi_home":
            print(f"    [domain] multi_home fallback: {len(train_home_names)} source-home classes")
        else:  # binary_source_target legacy pooled source-vs-target path
            if target_items:
                target_ds = DomainDataset(target_items, db_folder_map)
                target_loader = DataLoader(
                    target_ds, batch_size=DOMAIN_BATCH_SIZE, shuffle=True,
                    collate_fn=domain_collate_fn, num_workers=NUM_WORKERS,
                    worker_init_fn=seed_worker, persistent_workers=False)
                target_iter = make_cycled_iter(target_loader)
            else:
                print("    [domain] WARNING: no target-domain windows; domain loss disabled")

    model = MLPBiGRU().to(DEVICE)
    ce_loss = CELoss(weight=class_weights.to(DEVICE))

    domain_disc = None
    domain_params = []
    if ABLATION != "baseline":
        if DOMAIN_ADAPT_MODE == "binary_source_target":
            domain_out_dim = 1
        elif DOMAIN_ADAPT_MODE == "multi_source_target":
            domain_out_dim = len(train_home_names) + 1
        else:
            domain_out_dim = len(train_home_names)

        domain_in_dim = DOMAIN_FEAT_DIM if ABLATION == "dann" else DOMAIN_FEAT_DIM * N_CLASSES
        domain_disc = DomainDiscriminator(domain_in_dim, out_dim=domain_out_dim).to(DEVICE)
        domain_params = list(domain_disc.parameters())

        chance = 0.5 if DOMAIN_ADAPT_MODE == "binary_source_target" else 1.0 / domain_out_dim
        print(f"Domain discriminator chance accuracy ≈ {chance:.3f}")

    optimizer = torch.optim.Adam(
        list(model.parameters()) + domain_params, lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=SCHED_STEP, gamma=SCHED_GAMMA)

    print(f"\nModel params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print("Task loss: CE(BiGRU) only")
    if domain_disc is not None:
        print(f"Domain loss: {ABLATION.upper()} / {DOMAIN_ADAPT_MODE}")

    history_f1 = []
    print(f"\n{'Ep':>4}  {'TrainLoss':>9}  {'HeldoutF1':>9}  Domain")
    print("-" * 70)

    for ep in range(1, EPOCHS + 1):
        if domain_disc is not None and ep > DOMAIN_WARMUP_EPOCHS:
            p = (ep - DOMAIN_WARMUP_EPOCHS) / max(EPOCHS - DOMAIN_WARMUP_EPOCHS, 1)
            raw_lambda = LAMBDA_DOMAIN_MAX * (2.0 / (1.0 + np.exp(-5.0 * p)) - 1.0)
            lambda_domain = min(raw_lambda, DOMAIN_LAMBDA_CAP)
        else:
            lambda_domain = 0.0

        tr = train_one_epoch(
            model, train_loader, optimizer, ce_loss,
            domain_disc=domain_disc, target_iter=target_iter,
            balanced_domain_iter=balanced_domain_iter,
            lambda_domain=lambda_domain, domain_mode=DOMAIN_ADAPT_MODE,
            home_to_local_idx=home_to_local_idx)

        _heldout_loss, heldout_f1 = evaluate(model, test_loader, ce_loss)
        history_f1.append(heldout_f1)
        scheduler.step()

        if domain_disc is None:
            dom_s = "n/a"
        else:
            acc_s = f"{tr['domain_acc']:.3f}" if np.isfinite(tr["domain_acc"]) else "n/a"
            dom_s = f"loss={tr['domain']:.4f} λ={lambda_domain:.3f} acc={acc_s}"
        print(f"{ep:>4}  {tr['total']:>9.4f}  {heldout_f1:>9.4f}  {dom_s}")

    # Always save the FINAL epoch.
    last_model_path = f"last_model_mlp64_bigru_{ABLATION}_{DOMAIN_ADAPT_MODE}_{fold_name}.pt"
    torch.save(model.state_dict(), last_model_path)
    if domain_disc is not None:
        torch.save(domain_disc.state_dict(),
                   f"last_domain_mlp64_bigru_{ABLATION}_{DOMAIN_ADAPT_MODE}_{fold_name}.pt")
    print(f"\nSaved final epoch {EPOCHS}: {last_model_path}")

    # F1-only trend plot.
    plt.figure(figsize=(7, 4))
    plt.plot(range(1, EPOCHS + 1), history_f1, marker="o", markersize=2)
    plt.xlabel("Epoch")
    plt.ylabel("Macro F1")
    plt.title(f"Held-out F1 trend — {held_out_home}")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    curve_path = f"f1_trend_mlp64_bigru_{ABLATION}_{DOMAIN_ADAPT_MODE}_{fold_name}.png"
    plt.savefig(curve_path, dpi=140)
    plt.close()
    print(f"F1 trend -> {curve_path}")

    # Reload the exact final checkpoint and report only the final MLP->BiGRU prediction.
    model.load_state_dict(torch.load(last_model_path, map_location=DEVICE, weights_only=True))
    test_loss, test_f1, pred, labels = evaluate(model, test_loader, ce_loss, return_preds=True)
    print(f"\nTEST {held_out_home}: loss={test_loss:.4f}  macro-F1={test_f1:.4f}")
    print("\n── Episode-level classification report ──")
    print(classification_report(
        labels, pred, labels=list(range(N_CLASSES)), target_names=CLASSES,
        digits=4, zero_division=0))

    cm = confusion_matrix(labels, pred, labels=list(range(N_CLASSES)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES, ax=axes[0])
    axes[0].set_title(f"Confusion — counts ({fold_name})")
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES, ax=axes[1])
    axes[1].set_title(f"Confusion — normalized ({fold_name})")
    plt.tight_layout()
    cm_path = f"confusion_mlp64_bigru_{ABLATION}_{DOMAIN_ADAPT_MODE}_{fold_name}.png"
    plt.savefig(cm_path, dpi=140)
    plt.close(fig)
    print(f"Confusion matrix -> {cm_path}")

    return {
        "held_out": held_out_home,
        "gru_f1": round(test_f1, 4),
        "test_loss": round(test_loss, 4),
        "model_epoch": EPOCHS,
    }


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    fold_names = [
        h["name"] for h in HOMES
        if RUN_ONLY_HELDOUT is None or h["name"] in RUN_ONLY_HELDOUT
    ]
    print(f"\nRunning {len(fold_names)} LOHO fold(s): {fold_names}")
    print(f"ABLATION={ABLATION}; DOMAIN_ADAPT_MODE={DOMAIN_ADAPT_MODE}")

    all_results = [run_fold(name) for name in fold_names]

    print("\n" + "█" * 64)
    print(f"LOHO SUMMARY — {ABLATION} / {DOMAIN_ADAPT_MODE}")
    print("█" * 64)
    print(f"{'Held-out home':<30} {'GRU-F1':>10}")
    print("-" * 44)
    for r in all_results:
        print(f"{r['held_out']:<30} {r['gru_f1']:>10.4f}")
    if all_results:
        mean_gru = np.mean([r["gru_f1"] for r in all_results])
        print("-" * 44)
        print(f"{'MEAN':<30} {mean_gru:>10.4f}")
