# TaxiETA — starter repo

A hands-on companion to *Designing Machine Learning Systems* (Chip Huyen). You build a
production-style system that predicts NYC taxi trip durations. It runs through the whole
lifecycle: data lake, streaming features, training, serving, drift monitoring, gated
retraining and Airflow 3 orchestration. It follows the milestones in the project doc
**“Hands-On Project: Build a Production ML System (NYC Taxi ETA)”**.

The plumbing works out of the box. The learning parts are left as `TODO(Mn)` markers, each tied
to a milestone. Run `grep -rn "TODO(M4)" src dags` to see what's left for Milestone 4.

## 1. Quick start (offline, ~5 minutes)

Needs Python 3.11 and [uv](https://docs.astral.sh/uv/). Synthetic data lets you check the
plumbing before you download gigabytes.

```bash
make setup                                   # .venv with the project + dev tools
make fixture MONTHS=2019-01:2019-06,2020-03:2020-04
.venv/bin/taxieta ingest --months 2019-01:2019-06,2020-03:2020-04
make query                                   # DuckDB over the Parquet lake
make train                                   # trains v0001 and makes it champion
make test                                    # 18 tests: leakage firewall, stream, API, gates
make serve                                   # API on :8000, in another terminal:
curl -s -X POST localhost:8000/predict -H 'content-type: application/json' \
  -d '{"pickup_ts":"2019-05-06T08:15:00","pu_zone":161,"do_zone":132}'
.venv/bin/taxieta drift-check --reference 2019-04 --current 2020-04   # shifted fixture month
```

> Fixture data is a **plumbing test, not a dataset.** It has the same schema and the same kinds of
> bad rows as the real files, and its 2020 months are deliberately shifted. Never draw
> modelling conclusions from it.

## 2. Real data

```bash
make download MONTHS=2019-01:2019-06,2020-01:2020-06   # ~100 MB per month from the TLC
make ingest SAMPLE=0.1                                  # 10% simple random sample while developing
```

Source: [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page).
Files come from `https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_YYYY-MM.parquet`.

## 3. The full stack (Docker Compose + Airflow 3)

```bash
make airflow-uid        # Linux only (file permissions); skip on macOS
make up                 # docker compose up -d --build
```

| Service | URL | What it is |
| --- | --- | --- |
| Airflow 3 | http://localhost:8080 | `airflow standalone`, no login (local dev only) |
| Prediction API | http://localhost:8000/docs | FastAPI, champion model + heuristic fallback |
| MLflow | http://localhost:5001 | Experiment tracking (5001: macOS uses 5000 for AirPlay) |
| Redpanda Console | http://localhost:8081 | Browse the `trip_started` / `trip_completed` topics |
| Kafka | `localhost:19092` (host), `redpanda:9092` (containers) | Redpanda, Kafka-compatible |

Give Docker Desktop at least 6 GB of RAM. Stream a month through the live system:

```bash
.venv/bin/taxieta replay --month 2019-05 --speedup 600            # -> Redpanda (consumer builds zone speeds)
.venv/bin/taxieta replay --month 2019-05 --sink http --limit 2000 # -> API predictions + label log
```

### Airflow 3 DAGs

The three DAGs are chained by **assets** (data-aware scheduling). None of them runs on a clock:

```
taxieta_ingest  --(LAKE asset)-->  taxieta_monitoring  --(DRIFT asset)-->  taxieta_retrain  --(CHAMPION asset)
 fetch -> ingest -> DQ gate        drift_check -> branch                   train -> gate -> branch
                                     signal_drift | no_drift                 promote -> reload_api | keep_champion
```

Try it: in the Airflow UI, trigger **taxieta_ingest** with `{"month": "2020-04", "source": "fixture"}`,
or `"download"` for real data. The LAKE event carries the month as metadata, monitoring
detects the shift and emits DRIFT, and retraining runs its evaluation gates. That's the book's
Stage 4 continual learning with a drift-based trigger (Ch. 9).

Airflow 3 features in use:

- the Task SDK (`from airflow.sdk import dag, task, Asset, Metadata, Param`)
- `@task.bash` (the last stdout line becomes the XCom)
- `@task.branch` for the conditional dependency "promote if gates pass, else keep the champion"
- asset events with metadata (`yield Metadata(LAKE, extra={"month": ...})`) read through `triggering_asset_events`

**Why the DAGs never `import taxieta`:** Airflow pins hundreds of packages. `docker/Dockerfile.airflow`
installs TaxiETA into its **own virtualenv**, and tasks call the `taxieta` CLI through `$TAXIETA_BIN`.
Conflicting dependencies can't break either side (Ch. 10), and every task can be run by hand.

Check DAGs without Docker: install Airflow in a separate venv (see `.github/workflows/ci.yml`), then run
`python scripts/check_dags.py dags`, or run one end to end with
`airflow dags test taxieta_ingest -c '{"month":"2019-05","source":"fixture"}'` and
`TAXIETA_BIN=$PWD/.venv/bin/taxieta`.

## 4. Where each milestone lives

| Milestone | Files | Done for you | Left for you (grep the TODO tag) |
| --- | --- | --- | --- |
| M1 Framing | `docs/design_doc.md`, `training/baselines.py` | Heuristic ETA, metrics | The design doc |
| M2 Data engineering | `ingest/`, `lake_queries.py`, `stream/` | Download, validation with rejection report, lake, DuckDB, replayer, zone-speed consumer | Format benchmarks, batch version of the streaming feature, missingness |
| M3 Training data | `labeling/` | Samplers incl. reservoir, label joiner, 2 labeling functions | More LFs + Snorkel eval, LongTrip imbalance study, active learning |
| M4 Features & leakage | `features/build.py`, `training/split.py` | Shared feature code, leak firewall test, time split, point-in-time zone speed | Hashing trick, centroid distance, the leakage lab |
| M5 Model development | `training/train.py`, `training/metrics.py` | LightGBM, baselines, borough slices, MLflow logging, model store | Four phases, tuning, MLP, ensembles, robustness/calibration tests |
| M6 Deployment | `serving/` | FastAPI, timeout fallback, prediction log, batch job, Dockerfile | Load tests, micro-batching, hybrid batch/online, ONNX + quantization |
| M7 Monitoring | `monitoring/drift.py` | KS + PSI drift report used by Airflow | Windows, seasonality, bug injection, alerts |
| M8 Continual learning | `continual/`, `dags/taxieta_retrain.py` | Gates, promotion, Thompson sampler, drift-triggered DAG | Stateful training, freshness experiment, shadow/canary/A-B |
| M9 Infrastructure | `docker-compose.yml`, `docker/`, `dags/`, `.github/`, `.devcontainer/` | Compose stack, Airflow 3, CI, dev container | Cron comparison, Feast, complete model store, pin images |
| M10 Responsible AI | `docs/model_card.md`, `docs/ownership.md` | Templates + fallback path | Fairness audit, model card, privacy review |

Answer each milestone's questions in `LEARNING_LOG.md` as you go.

## 5. Command reference

```text
taxieta fixture | download | ingest | query | replay | consume | train | batch-predict
        drift-check | gate | promote | champion | serve          (taxieta <cmd> -h)
```

Configuration comes from environment variables in `src/taxieta/config.py`: `TAXIETA_DATA`,
`TAXIETA_REPORTS`, `KAFKA_BOOTSTRAP`, `MLFLOW_TRACKING_URI`, `FALLBACK_TIMEOUT_MS`.

## 6. Troubleshooting

- **Port 5000 busy on macOS**: that's AirPlay. MLflow is mapped to 5001 for this reason.
- **Airflow can't write `./data` on Linux**: run `make airflow-uid` before `make up`.
- **`no champion model` from the API**: run `make train` (or the retrain DAG) first, then `POST /reload`.
- **Replay seems frozen**: raise `--speedup`, or add `--no-sleep` for a quick test.
- **Different results between runs**: that's the point of Milestone 5's debugging section. Fix seeds and check the logged data window.
