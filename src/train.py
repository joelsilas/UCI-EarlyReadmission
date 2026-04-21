"""Train DiverseStackingEnsemble across 5 seeds.

For each seed:
  1. Recreate the train/val/test split deterministically from the engineered parquet.
  2. Fit DiverseStackingEnsemble on training data.
  3. Find the optimal classification threshold on validation data (max F1 of class 1).
  4. Pickle the fitted ensemble and append the threshold to thresholds.json.

Reads:  artifacts/engineered.parquet
Writes: artifacts/models/seed_{seed}.pkl  (pickled DiverseStackingEnsemble)
        artifacts/models/thresholds.json  (per-seed validation-tuned thresholds)
"""

import argparse
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from models import DiverseStackingEnsemble
from splitting import SEEDS, create_splits

REPO_ROOT = Path(__file__).resolve().parent.parent
ENGINEERED_PATH = REPO_ROOT / "artifacts" / "engineered.parquet"
MODELS_DIR = REPO_ROOT / "artifacts" / "models"

logger = logging.getLogger(__name__)


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Threshold that maximises F1 of class 1 over [0.05, 0.95] in 0.01 steps."""
    best_t, best_score = 0.5, -1.0
    for t in np.arange(0.05, 0.96, 0.01):
        score = f1_score(y_true, (y_prob >= t).astype(int), pos_label=1, zero_division=0)
        if score > best_score:
            best_score, best_t = score, float(t)
    return best_t


def train_seed(df: pd.DataFrame, seed: int) -> tuple[DiverseStackingEnsemble, float]:
    logger.info(f"=== seed {seed} ===")
    split = create_splits(df, seed=seed)
    logger.info(
        f"split sizes: train={len(split.X_train)}, "
        f"val={len(split.X_val)}, test={len(split.X_test)}, "
        f"train pos_rate={float(split.y_train.mean()):.4f}"
    )

    model = DiverseStackingEnsemble()
    model.fit(split.X_train, split.y_train)

    val_proba = model.predict_proba(split.X_val)
    threshold = find_optimal_threshold(split.y_val.values, val_proba)
    logger.info(f"  optimal threshold (val F1): {threshold:.3f}")
    return model, threshold


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DiverseStackingEnsemble across 5 seeds")
    parser.add_argument("--input", default=str(ENGINEERED_PATH))
    parser.add_argument("--models-dir", default=str(MODELS_DIR))
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=SEEDS,
        help=f"Seeds to train (default: {SEEDS})",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    input_path = Path(args.input)
    models_dir = Path(args.models_dir)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Engineered parquet not found at {input_path}. Run preprocess.py first."
        )

    logger.info(f"Loading {input_path}")
    df = pd.read_parquet(input_path)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    models_dir.mkdir(parents=True, exist_ok=True)

    threshold_path = models_dir / "thresholds.json"
    if threshold_path.exists():
        with open(threshold_path) as f:
            thresholds: dict = json.load(f)
    else:
        thresholds = {}

    for seed in args.seeds:
        model, threshold = train_seed(df, seed)
        model_path = models_dir / f"seed_{seed}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        logger.info(f"  saved model → {model_path}")
        thresholds[str(seed)] = threshold
        with open(threshold_path, "w") as f:
            json.dump(thresholds, f, indent=2, sort_keys=True)

    logger.info(f"Saved thresholds → {threshold_path}")


if __name__ == "__main__":
    main()
