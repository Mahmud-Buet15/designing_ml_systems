# TaxiETA -- common commands. Run `make help`.
MONTHS ?= 2019-01:2019-06,2020-01:2020-06
SAMPLE ?= 0.1
PY ?= .venv/bin

.PHONY: help setup fixture download ingest query train serve replay consume test lint up down airflow-uid clean

help:  ## list targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-12s %s\n",$$1,$$2}'

setup:  ## create .venv with uv and install the project + dev tools
	uv venv .venv --python 3.11
	uv pip install --python .venv -e ".[stream,tracking,serve,dev,solutions]"
#Note: -e ".[stream,tracking,serve,dev,solutions]" means install extra packages for stream processing, tracking, serving, development and solutions

fixture:  ## synthetic TLC-shaped data (offline / CI)
	$(PY)/taxieta fixture --months $(MONTHS)

download:  ## real TLC data for MONTHS (several GB for all months!)
	$(PY)/taxieta download --months $(MONTHS)

ingest:  ## validate + write the lake (SAMPLE=0.1 down-samples while developing)
	$(PY)/taxieta ingest --months $(MONTHS) --sample-frac $(SAMPLE)

query:  ## DuckDB OLAP queries over the lake
	$(PY)/taxieta query

train:  ## train a model version (default windows: Jan-Mar train, Apr valid, May test)
	$(PY)/taxieta train

serve:  ## run the API locally on :8000
	$(PY)/taxieta serve

replay:  ## stream a month into Redpanda: make replay MONTH=2019-05
	$(PY)/taxieta replay --month $(MONTH) --speedup 600

consume:  ## streaming zone-speed consumer
	$(PY)/taxieta consume

test:  ## run tests (uses synthetic data in a temp dir)
	$(PY)/pytest

lint:
	$(PY)/ruff check src tests dags

airflow-uid:  ## Linux only: let the Airflow container write to ./data
	echo "AIRFLOW_UID=$$(id -u)" > .env

up:  ## start redpanda, console, mlflow, api, consumer, airflow
	docker compose up -d --build

down:
	docker compose down

clean:  ## delete generated data (keeps raw downloads)
	rm -rf data/lake data/features data/online data/models reports/ingest reports/drift reports/gates reports/train_*.json
