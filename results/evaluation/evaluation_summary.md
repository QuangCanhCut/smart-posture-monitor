# Smart Posture Monitor V02 - Held-out Evaluation

- Model: `SVM RBF Tuned`
- Test persons: `['person01', 'person10', 'person12']`
- Test samples: `707`
- Dataset SHA256: `2bc189b67d6ac75dc8a95dfb487c743b16af0f99172eebf28ce725b6c5abbb62`

## Overall metrics

- accuracy: `0.793494`
- balanced_accuracy: `0.788075`
- macro_precision: `0.784227`
- macro_recall: `0.788075`
- macro_f1: `0.779126`
- weighted_f1: `0.796576`

## Training-CV reference

- Selected-model CV Macro F1 mean: `0.607130`
- Selected-model CV Macro F1 std: `0.170123`
- Held-out minus CV Macro F1: `+0.171996`

## Per-class metrics

| class          |   precision |   recall |   f1_score |   support |
|:---------------|------------:|---------:|-----------:|----------:|
| correct        |    0.875    | 0.936047 |   0.904494 |       172 |
| forward_slouch |    0.563536 | 0.772727 |   0.651757 |       132 |
| lean_left      |    0.994924 | 0.867257 |   0.926714 |       226 |
| lean_right     |    0.703448 | 0.576271 |   0.63354  |       177 |

## Per-person metrics

| person_id   |   samples |   accuracy |   macro_precision |   macro_recall |   macro_f1 |   weighted_f1 |
|:------------|----------:|-----------:|------------------:|---------------:|-----------:|--------------:|
| person01    |       148 |   0.797297 |          0.79485  |       0.771347 |   0.667631 |      0.768973 |
| person10    |       307 |   0.745928 |          0.783704 |       0.779371 |   0.744852 |      0.751592 |
| person12    |       252 |   0.849206 |          0.888073 |       0.853165 |   0.851527 |      0.851693 |

## Confusion matrix

Rows are true labels; columns are predicted labels.

```text
[[161   2   0   9]
 [  0 102   0  30]
 [  9  17 196   4]
 [ 14  60   1 102]]
```

## Protocol

- No fit/retrain.
- No hyperparameter tuning.
- No new split.
- Test persons come only from `models/split_manifest.json`.