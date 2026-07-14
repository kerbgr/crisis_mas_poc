import time

import pytest
from flask import Flask

from alert_manager import AlertManager
from alerts import send_alert
from production_monitor import ProductionMonitor, ResponseMetrics, register_routes
from get_canary_weight import get_canary_weight, apply_canary_weight, CANARY_SCHEDULE
from check_health_and_rollback import check_health_and_rollback
from should_promote_model_b import should_promote_model_b
from detect_safety_failures import detect_safety_failures


def test_alert_manager_throttles_repeated_alerts():
    mgr = AlertManager()
    assert mgr.should_send_alert("latency") is True
    assert mgr.should_send_alert("latency") is False  # throttled


def test_alert_manager_groups_by_severity_and_type():
    mgr = AlertManager()
    alerts = [
        {"severity": "warning", "type": "latency", "timestamp": 1, "message": "a"},
        {"severity": "warning", "type": "latency", "timestamp": 2, "message": "b"},
        {"severity": "critical", "type": "safety", "timestamp": 3, "message": "c"},
    ]
    grouped = mgr.group_alerts(alerts)
    latency_group = next(g for g in grouped if g["type"] == "latency")
    assert latency_group["count"] == 2


def test_send_alert_writes_and_returns_entry(tmp_path):
    log_path = tmp_path / "alerts.jsonl"
    alert = send_alert("test message", severity="info", log_path=str(log_path))
    assert alert["message"] == "test message"
    assert log_path.exists()


def test_production_monitor_flags_high_latency_and_safety_issues(capsys):
    monitor = ProductionMonitor(window_size=100)
    monitor.log_response(ResponseMetrics(
        timestamp=time.time(), request_id="1", latency_ms=50,
        input_tokens=10, output_tokens=10, user_feedback=0,
        safety_flags=[], confidence_score=0.9,
    ))
    monitor.log_response(ResponseMetrics(
        timestamp=time.time(), request_id="2", latency_ms=2000,
        input_tokens=10, output_tokens=10, user_feedback=-1,
        safety_flags=["unsafe_recommendation"], confidence_score=0.3,
    ))
    captured = capsys.readouterr()
    assert "High latency" in captured.out
    assert "Safety issue" in captured.out


def test_production_monitor_register_routes_serves_chat_and_feedback():
    class MockModel:
        def generate(self, msgs):
            return "ok response"

    class MockTokenizer:
        def encode(self, text):
            return text.split()

    monitor = ProductionMonitor(window_size=10)
    app = Flask("test_app")
    register_routes(app, MockModel(), MockTokenizer(), monitor=monitor)
    client = app.test_client()

    r = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    assert r.get_json()["choices"][0]["message"]["content"] == "ok response"

    request_id = list(monitor.recent_metrics)[-1].request_id
    r2 = client.post("/feedback", json={"request_id": request_id, "feedback": 1})
    assert r2.status_code == 200
    assert monitor.recent_metrics[-1].user_feedback == 1


def test_canary_schedule_weight_bounds():
    weight = get_canary_weight()
    assert 0.0 <= weight <= 1.0
    assert weight in {s["model_b_weight"] for s in CANARY_SCHEDULE} | {0.0}


def test_apply_canary_weight_keeps_weights_complementary():
    config = {"model_a_weight": 0.5, "model_b_weight": 0.5, "enabled": True}
    apply_canary_weight(config)
    assert config["model_a_weight"] + config["model_b_weight"] == pytest.approx(1.0)


def test_check_health_and_rollback_ignores_zero_baseline_single_error():
    # Regression test: model_a with 0 historical errors used to make any
    # single model_b error look like ">2x" the baseline and trigger rollback.
    metrics = {"model_a": {"requests": 500, "errors": 0}, "model_b": {"requests": 500, "errors": 1}}
    config = {"model_a_weight": 0.8, "model_b_weight": 0.2, "enabled": True}
    check_health_and_rollback(metrics, config)
    assert config["enabled"] is True
    assert config["model_b_weight"] == 0.2


def test_check_health_and_rollback_triggers_on_genuinely_bad_model():
    metrics = {"model_a": {"requests": 500, "errors": 2}, "model_b": {"requests": 500, "errors": 60}}
    config = {"model_a_weight": 0.8, "model_b_weight": 0.2, "enabled": True}
    check_health_and_rollback(metrics, config)
    # check_health_and_rollback re-routes traffic to 100% A but (unlike the
    # separate rollback_to_model_a()) doesn't flip "enabled" -- it leaves
    # the A/B framework itself running with 0% going to B.
    assert config["model_b_weight"] == 0.0
    assert config["model_a_weight"] == 1.0


def test_should_promote_model_b_true_when_all_criteria_pass():
    metrics_a = {"accuracy": 0.80, "safety_failures": 5, "latency_p95": 900, "error_rate": 0.02, "requests": 500}
    metrics_b = {"accuracy": 0.90, "safety_failures": 3, "latency_p95": 920, "error_rate": 0.01, "requests": 500}
    ratings_a = {"mean": 4.0, "scores": [3, 4, 3, 4, 3, 4, 3, 4]}
    ratings_b = {"mean": 4.8, "scores": [5, 5, 4, 5, 5, 5, 4, 5]}
    assert should_promote_model_b(metrics_a, metrics_b, ratings_a, ratings_b) is True


def test_should_promote_model_b_false_when_accuracy_regresses():
    metrics_a = {"accuracy": 0.90, "safety_failures": 2, "latency_p95": 900, "error_rate": 0.01, "requests": 500}
    metrics_b = {"accuracy": 0.85, "safety_failures": 2, "latency_p95": 900, "error_rate": 0.01, "requests": 500}
    ratings_a = {"mean": 4.5, "scores": [4, 5, 4, 5]}
    ratings_b = {"mean": 4.5, "scores": [4, 5, 4, 5]}
    assert should_promote_model_b(metrics_a, metrics_b, ratings_a, ratings_b) is False


def test_detect_safety_failures_flags_unsafe_recommendation():
    issues = detect_safety_failures("Do not evacuate, it's not dangerous.")
    assert "potentially_unsafe_recommendation" in issues


def test_detect_safety_failures_clean_response_has_no_issues():
    issues = detect_safety_failures("Evacuate immediately, the IDLH for ammonia is 300 ppm.")
    assert issues == []
