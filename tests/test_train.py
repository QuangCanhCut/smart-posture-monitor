from pathlib import Path

import pandas as pd

from src.preprocessing import EXPECTED_FEATURE_COLUMNS, PreparedDataset
from src import train


def _fake_prepared_dataset() -> PreparedDataset:
    rows = []
    labels = ["correct", "forward_slouch", "lean_left", "lean_right"]
    persons = ["person01", "person02", "person03", "person04", "person05"]

    for person_index, person_id in enumerate(persons):
        for label_index, label in enumerate(labels):
            values = {
                feature: float(person_index * 10 + label_index + feature_index / 100)
                for feature_index, feature in enumerate(EXPECTED_FEATURE_COLUMNS)
            }
            rows.append(
                {
                    "image_path": f"data/raw/{label}/{person_id}_session01/frame_{label_index:04d}.jpg",
                    "session_id": "session01",
                    "person_id": person_id,
                    "recording_id": f"{person_id}__session01",
                    "label": label,
                    **values,
                }
            )

    df = pd.DataFrame(rows)
    X = df[EXPECTED_FEATURE_COLUMNS].copy()
    y = df["label"].map(
        {
            "correct": 0,
            "forward_slouch": 1,
            "lean_left": 2,
            "lean_right": 3,
        }
    ).astype(int)
    groups = df["person_id"].astype(str)
    metadata = df[
        ["image_path", "session_id", "person_id", "recording_id", "label"]
    ].copy()

    return PreparedDataset(
        df=df,
        X=X,
        y=y,
        groups=groups,
        metadata=metadata,
        feature_columns=list(EXPECTED_FEATURE_COLUMNS),
    )


def test_build_model_uses_final_xgboost_params() -> None:
    model = train.build_model()
    params = model.get_params()

    for name, value in train.XGB_PARAMS.items():
        assert params[name] == value

    assert params["objective"] == "multi:softprob"
    assert params["num_class"] == 4
    assert params["random_state"] == train.RANDOM_STATE


def test_create_or_load_person_split_has_no_leakage(tmp_path: Path) -> None:
    prepared = _fake_prepared_dataset()
    split = train.create_or_load_person_split(
        prepared=prepared,
        dataset_sha256="fake-sha",
        split_manifest_path=tmp_path / "split_manifest.json",
    )

    train_persons = set(split.manifest["train_persons"])
    test_persons = set(split.manifest["test_persons"])
    train_recordings = set(split.manifest["train_recording_ids"])
    test_recordings = set(split.manifest["test_recording_ids"])

    assert train_persons
    assert test_persons
    assert train_persons.isdisjoint(test_persons)
    assert train_recordings.isdisjoint(test_recordings)
    assert len(split.train_indices) + len(split.test_indices) == len(prepared.X)


def test_canonical_calibration_audit_columns() -> None:
    audit = pd.DataFrame(
        [
            {
                "recording_id": "person01__session01",
                "person_id": "person01",
                "total_samples": 100,
                "correct_samples": 40,
                "calibration_samples": 30,
                "remaining_correct": 10,
                "remaining_total": 70,
                "baseline_created": True,
                "order_source": "image_path_frame_index",
                "skip_reason": "",
            }
        ]
    )

    canonical = train._canonical_calibration_audit(audit)

    assert canonical.loc[0, "removed_samples"] == 30
    assert canonical.loc[0, "remaining_samples"] == 70
    assert list(canonical.columns) == [
        "recording_id",
        "person_id",
        "total_samples",
        "correct_samples",
        "calibration_samples",
        "removed_samples",
        "remaining_samples",
        "remaining_correct",
        "baseline_created",
        "order_source",
        "skip_reason",
    ]
