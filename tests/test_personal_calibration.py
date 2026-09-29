import numpy as np
import pytest

from src.personal_calibration import PersonalCalibration


def _sample(value: float) -> np.ndarray:
    return np.full(PersonalCalibration.FEATURE_DIM, value, dtype=float)


def test_baseline_median_dung() -> None:
    calibration = PersonalCalibration(target_samples=3)
    assert calibration.add_sample(_sample(1.0))
    assert calibration.add_sample(_sample(3.0))
    assert calibration.add_sample(_sample(5.0))

    baseline = calibration.baseline

    assert calibration.is_calibrated
    assert calibration.collected_samples == 3
    np.testing.assert_allclose(
        baseline,
        _sample(3.0),
    )


def test_transform_delta_dung() -> None:
    calibration = PersonalCalibration(target_samples=2)
    calibration.add_sample(_sample(1.0))
    calibration.add_sample(_sample(3.0))

    delta = calibration.transform(_sample(8.0))

    np.testing.assert_allclose(
        delta,
        _sample(6.0),
    )


def test_reset_hoat_dong() -> None:
    calibration = PersonalCalibration(target_samples=1)
    calibration.add_sample(_sample(1.0))

    calibration.reset()

    assert calibration.collected_samples == 0
    assert not calibration.is_calibrated
    assert calibration.baseline is None


@pytest.mark.parametrize(
    "bad_vector",
    [
        np.array([np.nan] * PersonalCalibration.FEATURE_DIM),
        np.array([np.inf] * PersonalCalibration.FEATURE_DIM),
        np.array([-np.inf] * PersonalCalibration.FEATURE_DIM),
    ],
)
def test_nan_inf_bi_reject(bad_vector: np.ndarray) -> None:
    calibration = PersonalCalibration(target_samples=1)

    assert not calibration.add_sample(bad_vector)
    assert calibration.collected_samples == 0


def test_vector_sai_dimension_bi_reject() -> None:
    calibration = PersonalCalibration(target_samples=1)

    assert not calibration.add_sample(np.array([1.0, 2.0]))
    assert calibration.collected_samples == 0


def test_khong_transform_khi_chua_calibrate() -> None:
    calibration = PersonalCalibration(target_samples=2)

    with pytest.raises(RuntimeError):
        calibration.transform(_sample(1.0))


def test_fit_transform_offline_dung() -> None:
    correct_samples = np.vstack(
        [
            _sample(1.0),
            _sample(3.0),
            _sample(5.0),
        ]
    )
    all_samples = np.vstack(
        [
            _sample(3.0),
            _sample(10.0),
        ]
    )

    baseline, delta = PersonalCalibration.fit_transform_offline(
        correct_samples=correct_samples,
        all_samples=all_samples,
    )

    np.testing.assert_allclose(baseline, _sample(3.0))
    np.testing.assert_allclose(
        delta,
        np.vstack([_sample(0.0), _sample(7.0)]),
    )
