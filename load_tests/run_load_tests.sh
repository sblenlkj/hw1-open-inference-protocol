#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-http://localhost:8000}"
RUN_TIME="${RUN_TIME:-30s}"

mkdir -p load_tests/results

run_scenario() {
  local users="$1"
  local spawn_rate="$2"
  local name="users_${users}"

  echo "Running ${name}: users=${users}, spawn_rate=${spawn_rate}"

  uv run locust \
    -f load_tests/locustfile.py \
    --headless \
    --host "${HOST}" \
    --users "${users}" \
    --spawn-rate "${spawn_rate}" \
    --run-time "${RUN_TIME}" \
    --csv "load_tests/results/${name}" \
    --only-summary
}

run_scenario 1 1
run_scenario 5 5
run_scenario 20 10
run_scenario 50 20
