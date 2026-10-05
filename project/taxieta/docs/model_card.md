# Model card: TaxiETA (Milestone 10)

Template adapted from Mitchell et al., “Model Cards for Model Reporting” (2018).
TODO(M10): generate the metric tables automatically from the model store / MLflow on every retrain.

## Model details
- Developer / owner:
- Version and date:
- Type: (e.g., LightGBM regressor on log-duration)
- Training algorithm, parameters, fairness constraints, features:
- Parent version / lineage:
- License, citation, contact:

## Intended use
- Primary uses:
- Primary users:
- Out-of-scope uses: (e.g., “not for driver performance evaluation or pay”)

## Factors
Borough, time of day / week, airport vs non-airport, trip length, …

## Metrics
- Performance measures:
- Decision thresholds (LongTrip classifier):
- How variation is measured (intervals, bootstrap):

## Evaluation data
Datasets, why they were chosen, preprocessing:

## Training data
Window, sampling, label source, known gaps (e.g. few outer-borough trips):

## Quantitative analyses
### Unitary (by one factor)
| Slice | n | MAE (min) | % within 20% |
| --- | --- | --- | --- |

### Intersectional (e.g., borough × rush hour)
| Slice | n | MAE (min) | % within 20% |
| --- | --- | --- | --- |

## Ethical considerations
Fairness findings, privacy (what the prediction log stores and for how long), compression effects on rare slices.

## Caveats and recommendations
