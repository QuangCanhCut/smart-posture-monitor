import numpy as np
import pytest

from src.personal_calibration import PersonalCalibration


def test_baseline_median_dung() -> None:
    calibration = PersonalCalibration(feature_count=3)
    calibration.add_sample([1.0, 10.0, 100.0])
    calibration.add_sample([3.0, 20.0, 200.0])
    calibration.add_sample([5.0, 30.0, 300.0])

    baseline = calibration.calculate_baseline()

    np.testing.assert_allclose(baseline, np.array([3.0, 20.0, 200.0]))
    assert calibration.is_ready
    assert calibration.sample_count == 3


def test_transform_delta_dung() -> None:
    calibration = PersonalCalibration(feature_count=3)
    calibration.add_sample([1.0, 2.0, 3.0])
    calibration.add_sample([3.0, 4.0, 5.0])
    calibration.calculate_baseline()

    delta = calibration.transform([4.0, 8.0, 10.0])

    np.testing.assert_allclose(delta, np.array([2.0, 5.0, 6.0]))


def test_reset_hoat_dong() -> None:
    calibration = PersonalCalibration(feature_count=2)
    calibration.add_sample([1.0, 2.0])
    calibration.calculate_baseline()

    calibration.reset()

    assert calibration.sample_count == 0
    assert not calibration.is_ready
    with pytest.raises(RuntimeError):
        calibration.get_baseline()


@pytest.mark.parametrize(
    "bad_vector",
    [
        [np.nan, 1.0],
        [np.inf, 1.0],
        [-np.inf, 1.0],
    ],
)
def test_nan_inf_bi_reject(bad_vector: list[float]) -> None:
    calibration = PersonalCalibration(feature_count=2)

    with pytest.raises(ValueError):
        calibration.add_sample(bad_vector)


def test_vector_sai_dimension_bi_reject() -> None:
    calibration = PersonalCalibration(feature_count=3)

    with pytest.raises(ValueError):
        calibration.add_sample([1.0, 2.0])


def test_khong_transform_khi_chua_calibrate() -> None:
    calibration = PersonalCalibration(feature_count=2)

    with pytest.raises(RuntimeError):
        calibration.transform([1.0, 2.0])
