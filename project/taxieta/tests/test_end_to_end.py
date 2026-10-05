"""Train -> register -> serve -> gate, on fixture data."""

import pytest

from taxieta import model_store
from taxieta.continual.gates import run_gates
from taxieta.monitoring.drift import check
from taxieta.training.train import train


@pytest.fixture(scope="module")
def champion(lake):
    out = train("2019-01", "2019-02", "2019-03", params={"num_leaves": 15}, num_boost_round=200)
    return out["version"]


def test_model_beats_baselines(champion):
    meta = model_store.load(champion)["meta"]
    test = meta["metrics"]["test"]
    assert test["model"]["mae_min"] < test["zero_rule"]["mae_min"]
    assert model_store.champion_version() == champion


def test_api_predicts_and_logs(champion):
    fastapi = pytest.importorskip("fastapi")  # noqa: F841
    from fastapi.testclient import TestClient

    from taxieta.serving.app import app

    with TestClient(app) as c:
        r = c.post("/predict", json={"pickup_ts": "2019-03-04T08:00:00", "pu_zone": 161, "do_zone": 230})
        assert r.status_code == 200
        body = r.json()
        assert body["eta_seconds"] > 0 and body["model_version"] == champion
        assert c.post("/predict", json={"pickup_ts": "2019-03-04T08:00:00", "pu_zone": 0,
                                        "do_zone": 1}).status_code == 422


def test_drift_detected_on_shifted_month(champion):
    assert check("2019-02", "2020-04")["drift_detected"] is True


def test_gates_produce_a_decision(champion):
    v2 = train("2019-01:2019-02", "2019-03", "2020-04", params={"num_leaves": 15}, num_boost_round=200)["version"]
    decision = run_gates(v2, "2019-03")
    assert isinstance(decision["promote"], bool) and len(decision["gates"]) == 2
