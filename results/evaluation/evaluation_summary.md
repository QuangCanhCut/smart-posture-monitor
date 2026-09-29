# Smart Posture Monitor V03 - Held-out Evaluation

- Model: `XGBoost Tuned`
- Test persons: `['person01', 'person10', 'person12']`
- Test samples: `617`
- Dataset SHA256: `2bc189b67d6ac75dc8a95dfb487c743b16af0f99172eebf28ce725b6c5abbb62`

## Overall metrics

- accuracy: `0.632091`
- balanced_accuracy: `0.702947`
- macro_precision: `0.688106`
- macro_recall: `0.702947`
- macro_f1: `0.644193`
- weighted_f1: `0.624098`

## Training-CV reference

- Selected-model CV Macro F1 mean: `0.688936`
- Selected-model CV Macro F1 std: `0.140001`
- Held-out minus CV Macro F1: `-0.044743`

## Per-class metrics

| class          |   precision |   recall |   f1_score |   support |
|:---------------|------------:|---------:|-----------:|----------:|
| correct        |    0.640625 | 1        |   0.780952 |        82 |
| forward_slouch |    0.403587 | 0.681818 |   0.507042 |       132 |
| lean_left      |    0.954023 | 0.367257 |   0.530351 |       226 |
| lean_right     |    0.75419  | 0.762712 |   0.758427 |       177 |

## Per-person metrics

| person_id   |   samples |   accuracy |   macro_precision |   macro_recall |   macro_f1 |   weighted_f1 |
|:------------|----------:|-----------:|------------------:|---------------:|-----------:|--------------:|
| person01    |       118 |   0.635593 |          0.746977 |       0.73436  |   0.661859 |      0.664222 |
| person10    |       277 |   0.429603 |          0.539448 |       0.517319 |   0.466261 |      0.433562 |
| person12    |       222 |   0.882883 |          0.884716 |       0.898177 |   0.863529 |      0.885394 |

## Confusion matrix

Rows are true labels; columns are predicted labels.

```text
[[ 82   0   0   0]
 [  0  90   0  42]
 [ 44  97  83   2]
 [  2  36   4 135]]
```

## Protocol

- No fit/retrain.
- No hyperparameter tuning.
- No new split.
- Test persons come only from `models/split_manifest.json`.
- V03 evaluation rebuilds Personal Baseline Delta features before prediction.