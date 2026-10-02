"""
Offline Dataset Builder for Personal Baseline Delta.

Simulates the production calibration flow across historical sessions in `data/processed/features.csv`:
1. Reads `data/processed/features.csv` (strictly read-only).
2. For each unique (person_id, session_id):
   - Extracts the FIRST N frames where label == 'correct' (simulating 2-3s calibration window).
   - Computes personal baseline: f_baseline = mean(calib_frames).
   - Computes delta features: Δf = f_current - f_baseline.
3. Constructs hybrid feature representations:
   - 'selective' (40 features: 32 absolute + 8 delta)
   - 'clean_selective' (28 features: 20 clean absolute + 8 delta)
   - 'full' (64 features: 32 absolute + 32 delta)
4. Saves augmented dataset inside the baseline_calibration directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baseline_calibration.feature_schema import (
    FEATURE_NAMES_32,
    SENSITIVE_DELTA_FEATURES_8,
    CLEAN_FEATURES_20,
    get_hybrid_feature_names,
)
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "features.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "data"


def compute_baseline_for_session(
    df_session: pd.DataFrame,
    feature_cols: List[str],
    n_calib_frames: int = 30,
    min_frames_threshold: int = 10,
) -> Tuple[np.ndarray, int]:
    """
    Computes baseline vector for a single session using the first N correct frames.

    Args:
        df_session: DataFrame containing rows for one (person_id, session_id).
        feature_cols: List of 32 feature columns.
        n_calib_frames: Number of initial correct frames to collect (default 30).
        min_frames_threshold: Minimum frames required; falls back if fewer.

    Returns:
        Tuple of (baseline_vector: np.ndarray shape (len(feature_cols),), frames_used: int).
    """
    correct_mask = (df_session["label"] == "correct")
    correct_frames = df_session.loc[correct_mask, feature_cols]

    if len(correct_frames) == 0:
        # Extreme fallback: if no correct frames in session, take mean of whole dataset correct
        # This will be flagged by caller if needed
        raise ValueError("Session contains no 'correct' frames for baseline calibration.")

    # Strictly take the FIRST N frames (matching production calibration window)
    calib_slice = correct_frames.head(n_calib_frames)

    if len(calib_slice) < min_frames_threshold:
        # Fallback: if fewer than min_frames_threshold, use all available correct frames in session
        calib_slice = correct_frames

    baseline_vector = calib_slice.values.mean(axis=0).astype(np.float32)
    return baseline_vector, len(calib_slice)


def build_offline_baseline_dataset(
    dataset_path: Optional[Union[str, Path]] = None,
    n_calib_frames: int = 30,
    scheme: str = "selective",
    save_output: bool = True,
    output_filename: Optional[str] = None,
) -> pd.DataFrame:
    """
    Process dataset and generate hybrid features for all frames.

    Args:
        dataset_path: Path to features.csv (read-only).
        n_calib_frames: Number of calibration frames (default 30 ≈ 2-3s).
        scheme: 'selective' (40D), 'clean_selective' (28D), or 'full' (64D).
        save_output: Whether to persist the generated CSV inside baseline_calibration/data/.
        output_filename: Custom filename for the output CSV.

    Returns:
        pd.DataFrame containing metadata, absolute features, and delta/hybrid features.
    """
    path = Path(dataset_path) if dataset_path else DEFAULT_DATASET_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Source features file not found: {path}")

    print(f"[DatasetBuilder] Loading dataset from: {path}")
    df = pd.read_csv(path)

    # Validate essential columns
    for col in ["person_id", "session_id", "label"]:
        if col not in df.columns:
            raise KeyError(f"Dataset missing required column: '{col}'")

    for col in FEATURE_NAMES_32:
        if col not in df.columns:
            raise KeyError(f"Dataset missing required feature column: '{col}'")

    # Global fallback baseline for sessions without enough correct frames
    global_correct = df[df["label"] == "correct"][FEATURE_NAMES_32].values
    global_fallback_baseline = global_correct.mean(axis=0)

    # Compute baselines per (person_id, session_id)
    session_baselines: Dict[Tuple[str, str], np.ndarray] = {}
    grouped = df.groupby(["person_id", "session_id"])

    stats = {"total_sessions": 0, "normal_calibrated": 0, "fallback_sessions": 0}

    for (person_id, session_id), group in grouped:
        stats["total_sessions"] += 1
        try:
            b_vec, n_used = compute_baseline_for_session(
                group, FEATURE_NAMES_32, n_calib_frames=n_calib_frames
            )
            session_baselines[(person_id, session_id)] = b_vec
            stats["normal_calibrated"] += 1
        except Exception:
            # Fall back to person's overall correct or global correct
            person_correct = df[(df["person_id"] == person_id) & (df["label"] == "correct")][FEATURE_NAMES_32]
            if len(person_correct) >= 5:
                session_baselines[(person_id, session_id)] = person_correct.values.mean(axis=0)
            else:
                session_baselines[(person_id, session_id)] = global_fallback_baseline
            stats["fallback_sessions"] += 1

    print(f"[DatasetBuilder] Baseline stats: {stats}")

    # Compute deltas for all rows
    print("[DatasetBuilder] Computing delta features...")
    records_deltas = []
    for idx, row in df.iterrows():
        key = (row["person_id"], row["session_id"])
        baseline = session_baselines[key]
        features_curr = row[FEATURE_NAMES_32].values.astype(np.float32)
        delta_curr = features_curr - baseline
        records_deltas.append(delta_curr)

    deltas_all_arr = np.array(records_deltas, dtype=np.float32)  # shape (N_samples, 32)

    # Create delta DataFrame
    delta_df_cols = [f"delta_{c}" for c in FEATURE_NAMES_32]
    df_deltas_full = pd.DataFrame(deltas_all_arr, columns=delta_df_cols, index=df.index)

    # Select columns according to scheme
    meta_cols = ["image_path", "session_id", "person_id", "label"]
    existing_meta = [c for c in meta_cols if c in df.columns]

    if scheme == "selective":
        # 32 absolute + 8 sensitive deltas
        sens_delta_cols = [f"delta_{c}" for c in SENSITIVE_DELTA_FEATURES_8]
        out_df = pd.concat([df[existing_meta], df[FEATURE_NAMES_32], df_deltas_full[sens_delta_cols]], axis=1)

    elif scheme == "clean_selective":
        # 20 clean absolute + 8 sensitive deltas
        sens_delta_cols = [f"delta_{c}" for c in SENSITIVE_DELTA_FEATURES_8]
        out_df = pd.concat([df[existing_meta], df[CLEAN_FEATURES_20], df_deltas_full[sens_delta_cols]], axis=1)

    elif scheme == "full":
        # 32 absolute + 32 deltas
        out_df = pd.concat([df[existing_meta], df[FEATURE_NAMES_32], df_deltas_full], axis=1)

    elif scheme == "delta_only_8":
        sens_delta_cols = [f"delta_{c}" for c in SENSITIVE_DELTA_FEATURES_8]
        out_df = pd.concat([df[existing_meta], df_deltas_full[sens_delta_cols]], axis=1)

    else:
        raise ValueError(f"Unknown scheme: {scheme}")

    if save_output:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fname = output_filename or f"features_hybrid_{scheme}_N{n_calib_frames}.csv"
        out_path = OUTPUT_DIR / fname
        out_df.to_csv(out_path, index=False)
        print(f"[DatasetBuilder] Saved hybrid dataset to: {out_path} ({len(out_df)} rows, {len(out_df.columns)} cols)")

    return out_df


if __name__ == "__main__":
    df_hybrid = build_offline_baseline_dataset(scheme="selective", n_calib_frames=30)
    print("Columns:", list(df_hybrid.columns))
