#!/usr/bin/env bash
# One-command repro. Syncs this repo to a compute host, builds the amd64 image, runs the SIFT churn
# sweep + multi-seed CIs + regime, syncs results back, and regenerates figures locally.
# Config via env: UIDX_HOST (ssh target, default sandeep@mini), UIDX_REMOTE (remote dir, default uidx).
set -euo pipefail
HOST="${UIDX_HOST:-sandeep@mini}"
REMOTE="${UIDX_REMOTE:-uidx}"
HERE="$(cd "$(dirname "$0")" && pwd)"

echo ">> sync → $HOST:$REMOTE"
rsync -az --delete --exclude results/ --exclude data/ "$HERE/" "$HOST:$REMOTE/"

echo ">> build image + download data + run sweep on $HOST"
ssh "$HOST" "cd ~/$REMOTE
  mkdir -p data results
  [ -f data/sift-128-euclidean.hdf5 ] || curl -fL --retry 3 -o data/sift-128-euclidean.hdf5 https://ann-benchmarks.com/sift-128-euclidean.hdf5
  docker build --platform linux/amd64 -t uidx:latest . >/dev/null
  run() { docker run --rm --platform linux/amd64 -v \"\$PWD\":/work -w /work uidx:latest \
    python src/experiment.py --dataset sift --sift-path /work/data/sift-128-euclidean.hdf5 \
    --n-pool 200000 --warmup 20000 --steady 200000 --eval-every 2000 --f-list 0.5 --R 16 --burst-block 20000 \"\$@\"; }
  for s in 7 8 9 10; do run --seed \$s --tag headline_s\$s > logs_\$s.txt 2>&1 & done
  wait; echo runs-done"

echo ">> sync results back + figures"
rsync -az "$HOST:$REMOTE/results/" "$HERE/results/"
python "$HERE/src/analyze_ci.py" "$HERE"/results/summary_sift_headline_s*.csv
python "$HERE/src/make_figures.py"
echo ">> done. See results/figures/"
