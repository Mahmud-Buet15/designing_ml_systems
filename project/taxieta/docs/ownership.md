# Ownership and runbooks (Milestone 10)

## Who owns what
| Area | Owner | Backup | Notes |
| --- | --- | --- | --- |
| Raw data & ingestion | | | |
| Labels (natural + weak supervision) | | | |
| Feature definitions (`features/`) | | | |
| Models & evaluation gates | | | |
| Serving API | | | |
| Platform (Airflow, Redpanda, MLflow) | | | |
| On-call | | | |

## Runbooks

### Alert: p99 latency > 200 ms for 10 min
1. Check fallback rate in the prediction log (`fallback=1`).
2. …

### Alert: daily MAE > 1.2x reference for 2 days
1. Check the latest `reports/drift/*.json`: real shift or an injected bug?
2. …

### Alert: ingest reject rate > 5%
1. Open `reports/ingest/<month>.json` and find which rule spiked.
2. …
