# Learning log

Answer in your own words, with numbers from *your* runs. This is where the learning sticks.

## M1 — Framing and requirements (Ch. 1–2)
1. Why is “% within ±20%” closer to the business than RMSE?
2. Name one way the model could hit its ML metric and still hurt the business.
3. Your latency SLO is written with percentiles. Why not the average?
4. Why might a regression framing, with the destination as an *input* feature, beat one classifier per destination?

**Heuristic MAE on May 2019:** _____ min

## M2 — Data engineering (Ch. 3)
- CSV vs Parquet — size: ___ / ___ MB · load all columns: ___ / ___ s · load 3 columns: ___ / ___ s
- DataFrame iteration by row vs by column vs NumPy: ___ / ___ / ___ s
1. Which format would you use for training data, and which for appending events one by one? Why?
2. Compare the three dataflow modes (database, service request, real-time transport) for getting zone speeds to the prediction service.
3. Which of your features are batch (static) and which are streaming (dynamic)?
4. Why does the label arrive on a different schedule from the features? What does that imply for training?

## M3 — Training data (Ch. 4)
| Sample | Borough mix vs full | Duration distribution vs full | Biased? |
| --- | --- | --- | --- |
| Simple random | | | |
| Stratified | | | |
| Convenience | | | |

| LongTrip variant | Precision | Recall | F1 | PR-AUC |
| --- | --- | --- | --- | --- |
| Plain | | | | |
| Undersampled | | | | |
| Class weights | | | | |
| Focal / two-phase | | | | |

1. Which nonprobability sampling method did your convenience sample resemble? What bias did it introduce?
2. When is weak supervision a better investment than more hand labels? When is it worse?
3. Why must resampling happen after the train/test split?
4. Does undersampling change predicted probabilities? What does that mean for calibration?

## M4 — Features and leakage (Ch. 5)
| # | Experiment | MAE (min) | What leaked |
| --- | --- | --- | --- |
| 1 | + trip_distance / fare / tip | | |
| 2 | Random vs time split | | |
| 3 | Scaler/imputer fit on all data | | |
| 4 | Historical aggregate incl. target week | | |
| 5 | Duplicates across splits | | |

1. Why is splitting by time mandatory here and optional for classifying cat photos?
2. Which features would you drop if the streaming pipeline went down? What's the fallback?
3. You added a feature and MAE improved 30%. List three checks before celebrating.
4. What are the costs of too many features in production?

## M5 — Model development and offline evaluation (Ch. 6)
| Model | Valid MAE | Test MAE | % within 20% | p50 latency | Notes |
| --- | --- | --- | --- | --- | --- |
| Random | | | | | |
| Zero rule | | | | | |
| Heuristic | | | | | |
| Linear | | | | | |
| LightGBM | | | | | |
| MLP | | | | | |
| Ensemble | | | | | |

1. Did the complex model beat LightGBM by enough to justify its cost? Which evidence decided it?
2. Which assumption of your best model (IID, smoothness, …) is violated by this data?
3. What did slicing reveal that the overall metric hid?
4. Is your system good, useful, both, or neither?

## M6 — Deployment (Ch. 7)
| Load (req/s) | p50 ms | p90 ms | p99 ms | Throughput |
| --- | --- | --- | --- | --- |
| 10 | | | | |
| 50 | | | | |
| 200 | | | | |

1. Which prediction mode is your system: batch, online with batch features, or streaming prediction? Why?
2. Where would two separate pipelines have crept in without the shared `features/` code?
3. When would you choose batch prediction for ETA despite its staleness?
4. Which deployment myth did this milestone disprove for you?

## M7 — Distribution shift and monitoring (Ch. 8)
- First detector to fire on the 2020 replay: ______ on ______ · MAE degradation visible from labels on ______
1. Which monitor gave the earliest useful warning, and which gave the most false alarms?
2. Describe the degenerate feedback loop if riders cancel long ETAs and only accepted trips become training data. How would you correct it?
3. Give an edge case that isn't an outlier, and an outlier that isn't an edge case.
4. Why is monitoring feature distributions useful for debugging but weak for detecting performance drops?

## M8 — Continual learning and test in production (Ch. 9)
| Trigger | Retrains over 2020 replay | Final MAE |
| --- | --- | --- |
| Time (daily) | | |
| Performance | | |
| Volume | | |
| Drift | | |

1. For this system, is the answer to “how often should I retrain?” a schedule or a trigger? Use your measurements.
2. Why isn't interleaving a natural fit here? What would you change to use it?
3. What was the biggest bottleneck to faster updates: fresh data, labels, evaluation time or compute?
4. When would you still choose shadow deployment over a canary?

## M9 — Infrastructure (Ch. 10)
1. Explain scheduler vs orchestrator using the components you ran.
2. Which feature-store problems (management, computation, consistency) did Feast solve for you, and which not?
3. Which Airflow drawback from the book (monolithic, not parameterized, static DAGs) still applies to Airflow 3, and which were fixed? (Test it: `Param`, dynamic task mapping, per-task environments.)
4. What would change on Kubernetes in the cloud instead of Docker Compose on a laptop?

## M10 — UX, teams, responsible AI (Ch. 11)
1. Name one decision this system should never automate, even if accurate.
2. Which trade-off (privacy–accuracy, compactness–fairness, consistency–accuracy, speed–accuracy) was hardest? Why?
3. What in your model card should a stakeholder read before using the model?
4. Why is “act early” cheaper than fixing fairness after launch? Point to a place in your pipeline.
