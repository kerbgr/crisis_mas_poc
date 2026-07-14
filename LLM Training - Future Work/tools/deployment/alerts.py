"""Shared alert dispatch used by rollback_to_model_a.py,
check_health_and_rollback.py, concept_drift_detector.py, and
data_drift_detector.py. None of those had a working `send_alert`/
`send_to_pagerduty`/`send_to_slack` anywhere in the codebase -- this
provides one real (if minimal) implementation: log to alerts.jsonl and
print. Swap in real Slack/PagerDuty webhooks for production use.
"""

import json
import time


def send_alert(message, severity="warning", log_path="alerts.jsonl"):
    alert = {"timestamp": time.time(), "severity": severity, "message": message}

    with open(log_path, "a") as f:
        f.write(json.dumps(alert) + "\n")

    print(f"[{severity.upper()}] {message}")
    return alert


def send_to_slack(alert):
    """Placeholder -- integrate with your Slack webhook."""
    pass


def send_to_pagerduty(alert):
    """Placeholder -- integrate with the PagerDuty Events API."""
    pass
