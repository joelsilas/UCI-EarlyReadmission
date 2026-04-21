import argparse
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


from models import DiverseStackingEnsemble
from splitting import SEEDS, create_splits

REPO_ROOT = Path(__file__).resolve().parent.parent
ENGINEERED_PATH = REPO_ROOT / "artifacts" / "engineered.parquet"
MODELS_DIR = REPO_ROOT / "artifacts" / "models"

logger = logging.getLogger(__name__)


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        if i == 0:
            mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        else:
            mask = (y_prob > bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        ece += mask.sum() * abs(float(y_true[mask].mean()) - float(y_prob[mask].mean()))
    return ece / len(y_true)


def compute_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float
) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    p_curve, r_curve, _ = precision_recall_curve(y_true, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    return {
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(auc(r_curve, p_curve)),
        "f1_class1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "precision_class1": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_class1": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece": compute_ece(y_true, y_prob),
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "specificity": float(specificity),
        "npv": float(npv),
    }


def compute_dca(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """Decision curve analysis (Vickers & Elkin 2006).

    net_benefit(pt) = TP/n - FP/n * pt/(1-pt)
    treat_all(pt)   = prevalence - (1 - prevalence) * pt/(1-pt)
    """
    if thresholds is None:
        thresholds = np.arange(0.01, 0.51, 0.01)
    n = len(y_true)
    prevalence = float(y_true.mean())
    rows = []
    for pt in thresholds:
        y_pred = (y_prob >= pt).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        odds = pt / (1.0 - pt)
        model_nb = tp / n - (fp / n) * odds
        treat_all_nb = prevalence - (1.0 - prevalence) * odds
        rows.append((float(pt), float(model_nb), float(treat_all_nb), prevalence))
    return pd.DataFrame(rows, columns=["threshold", "model_nb", "treat_all_nb", "prevalence"])


def evaluate_seed(
    df: pd.DataFrame, seed: int, models_dir: Path, threshold: float
) -> tuple[dict[str, float], pd.DataFrame]:
    split = create_splits(df, seed=seed)
    model_path = models_dir / f"seed_{seed}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Run train.py for this seed.")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    test_proba = model.predict_proba(split.X_test)
    y_test = split.y_test.values
    metrics = compute_metrics(y_test, test_proba, threshold)
    dca = compute_dca(y_test, test_proba)
    return metrics, dca


def print_summary(per_seed: dict[int, dict[str, float]]) -> None:
    metric_keys = [
        "auroc",
        "auprc",
        "f1_class1",
        "precision_class1",
        "recall_class1",
        "brier",
        "ece",
        "threshold",
    ]

    width = 12 + 11 * len(metric_keys)
    print("\n" + "=" * width)
    print("PER-SEED TEST METRICS")
    print("=" * width)
    print(f"{'seed':>10} " + " ".join(f"{k:>10}" for k in metric_keys))
    print("-" * width)
    for seed, metrics in per_seed.items():
        row = f"{seed:>10} " + " ".join(f"{metrics[k]:>10.4f}" for k in metric_keys)
        print(row)

    print("\n" + "=" * width)
    print(f"{len(per_seed)}-SEED MEAN +/- STD")
    print("=" * width)
    for k in metric_keys:
        values = [m[k] for m in per_seed.values()]
        print(f"  {k:<20} {np.mean(values):.4f} +/- {np.std(values):.4f}")
    print("=" * width + "\n")


def print_confusion_matrices(per_seed: dict[int, dict[str, float]]) -> None:
    print("=" * 72)
    print("CONFUSION MATRICES (at tuned threshold)")
    print("=" * 72)
    for seed, m in per_seed.items():
        n = m["tn"] + m["fp"] + m["fn"] + m["tp"]
        prev = (m["tp"] + m["fn"]) / n if n > 0 else 0.0
        print(
            f"\nseed {seed}   thr={m['threshold']:.2f}   n={n}   prev={prev:.4f}"
        )
        print(f"                 pred 0     pred 1")
        print(
            f"  actual 0   [  {m['tn']:>7d}    {m['fp']:>7d} ]    "
            f"spec={m['specificity']:.4f}  NPV={m['npv']:.4f}"
        )
        print(
            f"  actual 1   [  {m['fn']:>7d}    {m['tp']:>7d} ]    "
            f"recall={m['recall_class1']:.4f}  prec={m['precision_class1']:.4f}"
        )

    tn = sum(m["tn"] for m in per_seed.values())
    fp = sum(m["fp"] for m in per_seed.values())
    fn = sum(m["fn"] for m in per_seed.values())
    tp = sum(m["tp"] for m in per_seed.values())
    n = tn + fp + fn + tp
    prev = (tp + fn) / n if n > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    print(f"\naggregate (summed across {len(per_seed)} seeds)   n={n}   prev={prev:.4f}")
    print(f"                 pred 0     pred 1")
    print(
        f"  actual 0   [  {tn:>7d}    {fp:>7d} ]    "
        f"spec={spec:.4f}  NPV={npv:.4f}"
    )
    print(
        f"  actual 1   [  {fn:>7d}    {tp:>7d} ]    "
        f"recall={recall:.4f}  prec={precision:.4f}"
    )
    print("=" * 72 + "\n")


def print_dca_summary(
    per_seed_dca: dict[int, pd.DataFrame],
    report_pts: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.30),
) -> None:
    seeds = list(per_seed_dca.keys())
    col_w = 14
    width = 8 + col_w * (2 * len(seeds) + 2)
    print("=" * width)
    print("DECISION CURVE ANALYSIS (net benefit)")
    print("=" * width)

    header = f"{'pt':>6}  "
    for s in seeds:
        header += f"{f'seed{s}_model':>{col_w}}{f'seed{s}_all':>{col_w}}"
    header += f"{'model_mean':>{col_w}}{'model_std':>{col_w}}"
    print(header)
    print("-" * width)

    for pt in report_pts:
        row = f"{pt:>6.2f}  "
        model_vals = []
        for s in seeds:
            df = per_seed_dca[s]
            i = int(np.argmin(np.abs(df["threshold"].values - pt)))
            model_nb = float(df["model_nb"].iloc[i])
            all_nb = float(df["treat_all_nb"].iloc[i])
            model_vals.append(model_nb)
            row += f"{model_nb:>{col_w}.4f}{all_nb:>{col_w}.4f}"
        row += f"{np.mean(model_vals):>{col_w}.4f}{np.std(model_vals):>{col_w}.4f}"
        print(row)
    print("=" * width + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained ensembles on test set")
    parser.add_argument("--input", default=str(ENGINEERED_PATH))
    parser.add_argument("--models-dir", default=str(MODELS_DIR))
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    input_path = Path(args.input)
    models_dir = Path(args.models_dir)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Engineered parquet not found at {input_path}. Run preprocess.py first."
        )

    threshold_path = models_dir / "thresholds.json"
    if not threshold_path.exists():
        raise FileNotFoundError(
            f"thresholds.json not found at {threshold_path}. Run train.py first."
        )
    with open(threshold_path) as f:
        thresholds = json.load(f)

    logger.info(f"Loading {input_path}")
    df = pd.read_parquet(input_path)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    per_seed: dict[int, dict[str, float]] = {}
    per_seed_dca: dict[int, pd.DataFrame] = {}
    dca_dir = REPO_ROOT / "artifacts" / "dca"
    dca_dir.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        if str(seed) not in thresholds:
            raise KeyError(f"No threshold for seed {seed} in {threshold_path}. Run train.py.")
        logger.info(f"Evaluating seed {seed}")
        metrics, dca = evaluate_seed(df, seed, models_dir, thresholds[str(seed)])
        per_seed[seed] = metrics
        per_seed_dca[seed] = dca
        dca_path = dca_dir / f"seed_{seed}.parquet"
        dca.to_parquet(dca_path, index=False)
        logger.info(f"  saved DCA curve → {dca_path}")

    print_summary(per_seed)
    print_confusion_matrices(per_seed)
    print_dca_summary(per_seed_dca)


if __name__ == "__main__":
    main()
