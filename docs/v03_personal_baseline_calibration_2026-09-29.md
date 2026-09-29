# V03 Personal Baseline Calibration

## 1. Goal
Implement V03 delta features: `delta_f = raw_f - session_baseline`, where the
session baseline is the median of an initial correct-posture calibration window.

## 2. Design
V03 is implemented beside V02. Existing V02 files and artifacts are not
overwritten. Runtime prediction uses `PosturePredictorV03` and
`PersonalCalibration`.

## 3. Files Added
- `src/personal_calibration.py`
- `src/posture_predictor_v03.py`
- `src/train_v03.py`
- `scripts/test_webcam_v03.py`
- `scripts/dry_run_v03_inference.py`
- `tests/test_personal_calibration.py`

## 4. Files Modified
No V02 source files are modified by the V03 implementation.

## 5. Calibration Algorithm
- Scope: per `recording_id` (`person_id__session_id`)
- Sort calibration candidates by `image_path`
- Use up to 30 initial `correct` samples
- Require at least 10 `correct` samples
- Baseline statistic: median
- Remove calibration rows from train/validation/test samples

## 6. Data Leakage Prevention
- Reused V02 person holdout test persons
- Created validation persons only from V02 train persons
- No person appears in more than one split
- Calibration rows are excluded before model fitting/evaluation
- Baseline subtraction is done in raw feature space before model preprocessing
- Scaler fitting happens inside the SVM sklearn pipeline on train folds only

## 7. Training Protocol
- Train persons: ['person03', 'person04', 'person05', 'person06', 'person08', 'person09', 'person11', 'person14']
- Validation persons: ['person02', 'person07', 'person13']
- Test persons: ['person01', 'person10', 'person12']
- Model selection: validation macro F1
- Final metrics: held-out test persons

## 8. Dataset Statistics
- Raw samples: 4014
- Raw persons: 14
- Raw sessions: 3
- Raw recordings: 16
- Calibration samples removed: 480
- Delta samples: 3534
- Skipped sessions: 0

## 9. Model Results
- Best model: XGBoost
- Validation accuracy: 0.6689
- Validation macro F1: 0.6710
- Test accuracy: 0.5462
- Test macro F1: 0.5631
- Test weighted F1: 0.5215

Per-class test metrics:
- correct: F1=0.7773, precision=0.6357, recall=1.0000
- forward_slouch: F1=0.4662, precision=0.3483, recall=0.7045
- lean_left: F1=0.3656, precision=0.9623, recall=0.2257
- lean_right: F1=0.6435, precision=0.6607, recall=0.6271

## 10. V02 vs V03
- V02 accuracy: 0.7934936350777935
- V02 macro F1: 0.7791264727955565
- V03 accuracy: 0.5462
- V03 macro F1: 0.5631

## 11. Realtime Usage
Run:

```bash
python scripts/test_webcam_v03.py
```

Keys:
- `C`: calibrate or recalibrate
- `R`: reset calibration
- `Q` or `Esc`: quit

## 12. Limitations
- Offline calibration assumes the earliest `correct` images in each recording
  are a valid calibration window.
- The first V03 experiment uses delta-only 29-dimensional features.
- Webcam access was not required for training and should be verified on the
  target machine.

## 13. Next Experiments
- Compare repeated live calibration windows for stability.
- Try `raw_delta` features only after delta-only behavior is understood.
- Add calibration quality checks such as median absolute deviation thresholds.
