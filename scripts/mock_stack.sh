#!/usr/bin/env bash
# Start the full mock recording stack on THIS node and verify it works.
#
#   bash scripts/mock_stack.sh
#
# Starts (reusing what already runs): roscore, mock sensors, the recording
# dashboard (:8080), and the standalone annotator (:8090). Fails loudly
# with the offending log if any piece does not come up healthy.
#
# Recordings land in /tmp/tso_sensorium_mock, which is LOCAL TO THIS NODE:
# on a different cluster node the library starts empty and the URL changes.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROS_PORT=11350
DASHBOARD_PORT=8080
ANNOTATOR_PORT=8090
LOG_DIR="${TSO_LOG_DIR:-/tmp/tso_sensorium_logs}"
# Interpreter for the annotator, which needs the gui + lerobot extras and no
# ROS. Override with the env var to point at your own environment; defaults
# to whatever `python` resolves to on PATH.
LEROBOT_PYTHON="${TSO_ANNOTATE_PYTHON:-python}"

export PATH="$HOME/.pixi/bin:$PATH"
export ROS_MASTER_URI="http://localhost:${ROS_PORT}"
mkdir -p "$LOG_DIR"
cd "$REPO_DIR"

echo "== Cleaning stale services"
fuser -k ${DASHBOARD_PORT}/tcp 2>/dev/null || true
fuser -k ${ANNOTATOR_PORT}/tcp 2>/dev/null || true
# Always restart the mock sensors: a leftover publisher attached to a
# dead master passes a pgrep check while publishing into the void.
pkill -f tso_sensorium.scripts.mock_sensors 2>/dev/null || true
sleep 1

echo "== roscore (port ${ROS_PORT})"
if pixi run -e ros1 rostopic list > /dev/null 2>&1; then
  echo "   reusing running roscore"
else
  nohup pixi run -e ros1 roscore -p ${ROS_PORT} > "$LOG_DIR/roscore.log" 2>&1 &
  for _ in $(seq 1 30); do
    pixi run -e ros1 rostopic list > /dev/null 2>&1 && break
    sleep 1
  done
  pixi run -e ros1 rostopic list > /dev/null 2>&1 || {
    echo "!! roscore did not come up"; tail -20 "$LOG_DIR/roscore.log"; exit 1; }
fi

echo "== mock sensors"
nohup pixi run -e ros1 python -m tso_sensorium.scripts.mock_sensors \
  > "$LOG_DIR/mock_sensors.log" 2>&1 &

echo "== recording dashboard (:${DASHBOARD_PORT})"
nohup pixi run -e ros1 python -m tso_sensorium.scripts.record_service \
  --config_path configs/recording/mock_service.yaml \
  > "$LOG_DIR/record_service.log" 2>&1 &

echo "== annotator (:${ANNOTATOR_PORT})"
nohup env PYTHONPATH="$REPO_DIR" "$LEROBOT_PYTHON" -m tso_sensorium.scripts.annotate \
  --config_path configs/annotation/mock.yaml \
  > "$LOG_DIR/annotate.log" 2>&1 &

wait_healthy() {
  local name=$1 url=$2 log=$3
  for _ in $(seq 1 45); do
    if curl -s -o /dev/null --max-time 2 "$url"; then return 0; fi
    sleep 2
  done
  echo "!! $name did not answer at $url"; tail -25 "$log"; exit 1
}
wait_healthy "dashboard" "http://localhost:${DASHBOARD_PORT}/api/status" "$LOG_DIR/record_service.log"
wait_healthy "annotator" "http://localhost:${ANNOTATOR_PORT}/api/status" "$LOG_DIR/annotate.log"

echo "== waiting for live sensors"
ALIVE=0
for _ in $(seq 1 15); do
  ALIVE=$(curl -s "http://localhost:${DASHBOARD_PORT}/api/status" | python3 -c \
    "import json,sys; s=json.load(sys.stdin); print(sum(x['alive'] for x in s['sensors']))")
  [ "$ALIVE" = "3" ] && break
  sleep 2
done
if [ "$ALIVE" != "3" ]; then
  echo "!! only $ALIVE/3 sensors alive — service log:"; tail -30 "$LOG_DIR/record_service.log"; exit 1
fi

NODE_IP=$(hostname -I | awk '{print $1}')
echo
echo "All healthy on node $NODE_IP (recordings: /tmp/tso_sensorium_mock, logs: $LOG_DIR)"
echo "  Recording dashboard:  http://${NODE_IP}:${DASHBOARD_PORT}"
echo "  Annotator/generation: http://${NODE_IP}:${ANNOTATOR_PORT}"
