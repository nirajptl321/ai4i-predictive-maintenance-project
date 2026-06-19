# MLP and XGBoost Error Analysis

Test metrics were computed once after model and threshold selection.
Diagnostic failure-mode columns are used here only for interpretation.

## Test metrics

| model         | split   |   threshold |   accuracy |   precision |   recall |   f1_score |   f2_score |   roc_auc |   average_precision |   true_negative |   false_positive |   false_negative |   true_positive |
|:--------------|:--------|------------:|-----------:|------------:|---------:|-----------:|-----------:|----------:|--------------------:|----------------:|-----------------:|-----------------:|----------------:|
| MLPClassifier | test    |        0.75 |   0.955333 |    0.416667 | 0.784314 |   0.544218 |   0.666667 |  0.962787 |            0.591856 |            1393 |               56 |               11 |              40 |
| XGBoost       | test    |        0.8  |   0.972    |    0.565217 | 0.764706 |   0.65     |   0.714286 |  0.965    |            0.636256 |            1419 |               30 |               12 |              39 |

## MLPClassifier

- False negatives: 11
- False positives: 56

### Mean numeric feature values by prediction group

| error_type     |   air_temperature_k |   process_temperature_k |   rotational_speed_rpm |   torque_nm |   tool_wear_min |
|:---------------|--------------------:|------------------------:|-----------------------:|------------:|----------------:|
| false_negative |             299.691 |                 309.773 |                1548.09 |      41.382 |         172.182 |
| false_positive |             301.216 |                 310.396 |                1452.82 |      49.288 |         153.875 |
| true_negative  |             299.885 |                 309.968 |                1535.32 |      39.668 |         105.795 |
| true_positive  |             300.468 |                 310.088 |                1473.97 |      53.348 |         151.85  |

### Diagnostic failure-mode summary for errors

- `false_negative` count = 11; diagnostic mode sums = `{'twf': 6, 'hdf': 3, 'pwf': 2, 'osf': 0, 'rnf': 0}`
- `false_positive` count = 56; diagnostic mode sums = `{'twf': 0, 'hdf': 0, 'pwf': 0, 'osf': 0, 'rnf': 0}`

## XGBoost

- False negatives: 12
- False positives: 30

### Mean numeric feature values by prediction group

| error_type     |   air_temperature_k |   process_temperature_k |   rotational_speed_rpm |   torque_nm |   tool_wear_min |
|:---------------|--------------------:|------------------------:|-----------------------:|------------:|----------------:|
| false_negative |             299.642 |                 309.75  |                1545.17 |      42.242 |         180.167 |
| false_positive |             301.663 |                 311.03  |                1404.3  |      53.44  |         115.1   |
| true_negative  |             299.9   |                 309.963 |                1534.83 |      39.757 |         107.496 |
| true_positive  |             300.503 |                 310.103 |                1472.97 |      53.39  |         148.872 |

### Diagnostic failure-mode summary for errors

- `false_negative` count = 12; diagnostic mode sums = `{'twf': 7, 'hdf': 2, 'pwf': 2, 'osf': 1, 'rnf': 0}`
- `false_positive` count = 30; diagnostic mode sums = `{'twf': 0, 'hdf': 0, 'pwf': 0, 'osf': 0, 'rnf': 0}`
