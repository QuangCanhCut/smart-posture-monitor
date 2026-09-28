# Smart Posture Monitor V02 - Held-out Evaluation

- Model: `SVM RBF Tuned`
- Test persons: `['person01', 'person12', 'person13']`
- Test samples: `655`
- Dataset SHA256: `338d3abda82d26100dcc82b93ec2605e7bc727a860bd91261d2f8b2136d27de2`

## Overall metrics

- accuracy: `0.723664`
- balanced_accuracy: `0.727587`
- macro_precision: `0.752252`
- macro_recall: `0.727587`
- macro_f1: `0.715761`
- weighted_f1: `0.725446`

## Training-CV reference

- Selected-model CV Macro F1 mean: `0.770744`
- Selected-model CV Macro F1 std: `0.066037`
- Held-out minus CV Macro F1: `-0.054983`

## Per-class metrics

| class          |   precision |   recall |   f1_score |   support |
|:---------------|------------:|---------:|-----------:|----------:|
| correct        |    0.968153 | 0.894118 |   0.929664 |       170 |
| forward_slouch |    0.532    | 0.943262 |   0.680307 |       141 |
| lean_left      |    0.992188 | 0.675532 |   0.803797 |       188 |
| lean_right     |    0.516667 | 0.397436 |   0.449275 |       156 |

## Per-person metrics

| person_id   |   samples |   accuracy |   macro_precision |   macro_recall |   macro_f1 |   weighted_f1 |
|:------------|----------:|-----------:|------------------:|---------------:|-----------:|--------------:|
| person01    |       148 |   0.817568 |          0.819116 |       0.788183 |   0.687932 |      0.807975 |
| person12    |       252 |   0.888889 |          0.905529 |       0.887909 |   0.888611 |      0.891337 |
| person13    |       255 |   0.505882 |          0.598174 |       0.522388 |   0.434191 |      0.419994 |

## Confusion matrix

Rows are true labels; columns are predicted labels.

```text
[[152   2   0  16]
 [  0 133   0   8]
 [  0  27 127  34]
 [  5  88   1  62]]
```

## Protocol

- No fit/retrain.
- No hyperparameter tuning.
- No new split.
- Test persons come only from `models/split_manifest.json`.