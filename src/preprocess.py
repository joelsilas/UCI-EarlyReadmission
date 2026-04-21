"""
Reads:  data/raw/diabetic_data.csv (relative to repo root)
Writes: artifacts/engineered.parquet

The output parquet preserves row order; train.py and evaluate.py both
recreate splits from this file via splitting.create_splits and rely on
that ordering for split determinism.
"""

import argparse
import logging
from functools import lru_cache
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "raw" / "diabetic_data.csv"
ENGINEERED_PATH = REPO_ROOT / "artifacts" / "engineered.parquet"

logger = logging.getLogger(__name__)

# --- Constants ---

ZERO_VARIANCE_MEDS = [
    "acetohexamide",
    "troglitazone",
    "examide",
    "citoglipton",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
]

ACTIVE_MED_COLUMNS = [
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "tolazamide",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
]

DECEASED_DISCHARGE_IDS = {11, 13, 14, 19, 20, 21}

AGE_MIDPOINTS = {
    "[0-10)": 5.0,
    "[10-20)": 15.0,
    "[20-30)": 25.0,
    "[30-40)": 35.0,
    "[40-50)": 45.0,
    "[50-60)": 55.0,
    "[60-70)": 65.0,
    "[70-80)": 75.0,
    "[80-90)": 85.0,
    "[90-100)": 95.0,
}

# CCS single-level categories for diabetes (49 = without complications, 50 = with).
CCS_DIABETES_CATEGORIES = frozenset({"49", "50"})


# --- ICD-9 → CCS mapping ---

_MAPPER = None


def _get_mapper():
    global _MAPPER
    if _MAPPER is None:
        from icdmappings import Mapper

        _MAPPER = Mapper()
    return _MAPPER


def _ccs_candidates(code: str) -> list[str]:
    """Generate CCS-lookup candidate strings for a UCI-style ICD-9 code.

    UCI codes are stored at 3-digit / 3.1–3.2 decimal precision but the CCS
    lookup table is keyed at full 5-digit / 3.2 decimal precision. Try the
    most specific forms first, fall back to .9/.0 (unspecified/default) suffixes.
    """
    code = code.strip()
    if code[0] in ("V", "v", "E", "e"):
        return [code, code + ".0", code + ".9", code + "0", code + "9"]
    if "." in code:
        base, dec = code.split(".", 1)
        stripped = base + dec
        return [
            code,
            code + "0",
            code + "9",
            stripped.ljust(5, "0"),
            stripped.ljust(5, "9"),
            code + "00",
            code + "99",
        ]
    return [
        code + ".00",
        code + ".9",
        code + ".0",
        code + "90",
        code + "99",
        code + "00",
        code + "09",
    ]


@lru_cache(maxsize=4096)
def map_icd9_to_ccs(code: str) -> str:
    if not code or code == "0":
        return "OTHER"
    mapper = _get_mapper()
    for candidate in _ccs_candidates(code):
        result = mapper.map([candidate], source="icd9", target="ccs")
        if result and result[0]:
            return str(result[0])
    return "OTHER"


def is_diabetes_ccs(ccs_code: str) -> bool:
    return ccs_code in CCS_DIABETES_CATEGORIES


# --- Preprocessing (ccs-diag strategy) ---


def preprocess_ccs_diag(df: pd.DataFrame) -> pd.DataFrame:
    """Apply ccs-diag preprocessing strategy.

    encounter_id is preserved — v2-patient-history needs it for ordering and
    splitting.create_splits drops it before features reach the model.
    """
    df = df.copy()
    n_start = len(df)

    df = df.drop(columns=["weight", "payer_code"])
    df = df.drop(columns=ZERO_VARIANCE_MEDS)

    df = df[df["gender"] != "Unknown/Invalid"]
    df = df[~df["discharge_disposition_id"].isin(DECEASED_DISCHARGE_IDS)]

    df["readmitted"] = (df["readmitted"] == "<30").astype(int)

    df["has_glu_test"] = df["max_glu_serum"].notna().astype(int)
    df["max_glu_serum"] = df["max_glu_serum"].fillna("None")
    df["has_a1c_test"] = df["A1Cresult"].notna().astype(int)
    df["A1Cresult"] = df["A1Cresult"].fillna("None")

    df["medical_specialty"] = df["medical_specialty"].fillna("Unknown")
    non_unknown = df["medical_specialty"][df["medical_specialty"] != "Unknown"]
    top_specialties = set(non_unknown.value_counts().head(10).index.tolist()) | {"Unknown"}
    df["medical_specialty"] = df["medical_specialty"].apply(
        lambda x: x if x in top_specialties else "Other"
    )

    race_mode = df["race"].mode()[0]
    df["race"] = df["race"].fillna(race_mode)

    for col in ["diag_1", "diag_2", "diag_3"]:
        df[col] = df[col].fillna("0")
        df[f"{col}_group"] = df[col].apply(map_icd9_to_ccs)

    df["primary_diag_diabetes"] = df["diag_1_group"].apply(is_diabetes_ccs).astype(int)
    df["any_diag_diabetes"] = (
        df["diag_1_group"].apply(is_diabetes_ccs)
        | df["diag_2_group"].apply(is_diabetes_ccs)
        | df["diag_3_group"].apply(is_diabetes_ccs)
    ).astype(int)
    df = df.drop(columns=["diag_1", "diag_2", "diag_3"])

    df = df.reset_index(drop=True)
    n_categories = df["diag_1_group"].nunique()
    coverage = (df["diag_1_group"] != "OTHER").mean()
    logger.info(
        f"CCS mapping: {n_categories} unique categories, primary-diag coverage {coverage:.1%}"
    )
    logger.info(f"Preprocessed: {n_start} → {len(df)} rows, {len(df.columns)} columns")
    return df


# --- Feature engineering (v2-patient-history strategy) ---


def engineer_v2_patient_history(df: pd.DataFrame) -> pd.DataFrame:
    """Add within-patient prior aggregates (using encounter_id ordering), then v1 features.

    Prior-only (left-biased) aggregates: each row sees only encounters that came
    before it for the same patient. Under GroupShuffleSplit by patient_nbr, all
    encounters of a patient sit in the same partition — no leakage across folds.
    """
    df = df.copy()

    # Patient-history aggregates first (require encounter_id ordering).
    df = df.sort_values(["patient_nbr", "encounter_id"]).reset_index(drop=True)
    grp = df.groupby("patient_nbr", sort=False)

    df["prior_encounters_count"] = grp.cumcount()
    df["is_first_encounter"] = (df["prior_encounters_count"] == 0).astype(int)

    for src, dst in [
        ("num_medications", "prior_num_medications_sum"),
        ("time_in_hospital", "prior_time_in_hospital_sum"),
        ("number_inpatient", "prior_inpatient_sum"),
        ("number_emergency", "prior_emergency_sum"),
    ]:
        df[dst] = grp[src].cumsum() - df[src]

    # Expanding prior means; first encounter has no prior → fill with current value
    # (delta vs prior mean is then 0 for first-time patients).
    for src, dst in [
        ("num_medications", "prior_num_medications_mean"),
        ("time_in_hospital", "prior_time_in_hospital_mean"),
    ]:
        prior_sum = grp[src].cumsum() - df[src]
        prior_n = df["prior_encounters_count"]
        df[dst] = (prior_sum / prior_n.where(prior_n > 0)).fillna(df[src])

    df["num_medications_vs_prior_mean"] = (
        df["num_medications"] - df["prior_num_medications_mean"]
    )
    df["time_in_hospital_vs_prior_mean"] = (
        df["time_in_hospital"] - df["prior_time_in_hospital_mean"]
    )

    # v1 baseline features on top.
    for col in ["admission_type_id", "discharge_disposition_id", "admission_source_id"]:
        df[col] = df[col].astype(str)

    df["age_numeric"] = df["age"].map(AGE_MIDPOINTS)
    df = df.drop(columns=["age"])

    # Diagnosis groups already present from preprocessing.
    raw_diag_cols = [c for c in ["diag_1", "diag_2", "diag_3"] if c in df.columns]
    if raw_diag_cols:
        df = df.drop(columns=raw_diag_cols)

    med_cols = [c for c in ACTIVE_MED_COLUMNS if c in df.columns]
    df["num_med_changed"] = df[med_cols].isin(["Up", "Down"]).sum(axis=1)
    df["num_active_meds"] = (df[med_cols] != "No").sum(axis=1)

    df["total_visits"] = (
        df["number_inpatient"] + df["number_outpatient"] + df["number_emergency"]
    )

    df["meds_x_time"] = df["num_medications"] * df["time_in_hospital"]
    df["labs_per_day"] = df["num_lab_procedures"] / df["time_in_hospital"].clip(lower=1)
    df["procs_per_day"] = df["num_procedures"] / df["time_in_hospital"].clip(lower=1)

    df["service_utilization"] = (
        df["num_lab_procedures"] + df["num_procedures"] + df["num_medications"]
    )

    df["has_inpatient_history"] = (df["number_inpatient"] > 0).astype(int)
    df["has_emergency_history"] = (df["number_emergency"] > 0).astype(int)

    n_cat = df.select_dtypes(include="object").shape[1]
    n_num = df.select_dtypes(include="number").shape[1]
    logger.info(f"Engineered: {df.shape[1]} columns ({n_cat} categorical, {n_num} numerical)")
    return df


# --- Main ---


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess UCI Diabetes data for best model")
    parser.add_argument(
        "--data-path",
        default=str(DATA_PATH),
        help=f"Path to diabetic_data.csv (default: {DATA_PATH})",
    )
    parser.add_argument(
        "--output",
        default=str(ENGINEERED_PATH),
        help=f"Output parquet path (default: {ENGINEERED_PATH})",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    data_path = Path(args.data_path)
    output_path = Path(args.output)

    if not data_path.exists():
        raise FileNotFoundError(
            f"CSV not found at {data_path}. "
            "Download diabetic_data.csv from https://archive.ics.uci.edu/dataset/296"
        )

    logger.info(f"Loading {data_path}")
    df = pd.read_csv(data_path, na_values="?", low_memory=False)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    logger.info("Preprocessing (ccs-diag strategy)")
    df = preprocess_ccs_diag(df)

    logger.info("Feature engineering (v2-patient-history)")
    df = engineer_v2_patient_history(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved engineered data → {output_path} ({len(df)} rows, {len(df.columns)} cols)")


if __name__ == "__main__":
    main()
