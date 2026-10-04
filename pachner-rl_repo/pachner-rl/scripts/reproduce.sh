#!/usr/bin/env bash
# Reproduce the central results. usage: bash scripts/reproduce.sh [quick|full]
#   quick (~1 min): 3 seeds of the paired comparison and 5 zero-shot 100-tet scrambles -> results/quick/ (nothing else is touched)
#   full  (~30 min on one CPU core): everything in the report. The shipped fine-tuned checkpoint is reused;
#         delete checkpoints/large100_checkpoint.pkl first to re-run the 200 fine-tuning episodes (~15 min).
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1      # sklearn MLPs on tiny batches are fastest single-threaded
MODE=${1:-quick}
python scripts/check_setup.py
if [ "$MODE" = quick ]; then
  OUT=results/quick; mkdir -p "$OUT"
  python scripts/evaluate_paired.py synth --seeds 3 --out "$OUT/paired_synth.jsonl"
  python scripts/evaluate_paired.py real  --seeds 3 --out "$OUT/paired_real.jsonl"
  python scripts/large_scale.py zeroshot --scrambles 5 --out "$OUT/large_zeroshot.json"
  python scripts/summarize_paired.py "$OUT"
  echo "quick run finished; raw trials and summary are in $OUT/ (the central results in results/ are untouched)"
else
  [ -f configs/instances.json ] || python scripts/make_instances.py --restarts 200 --pool 20
  python scripts/evaluate_paired.py synth --seeds 50
  python scripts/evaluate_paired.py real  --seeds 50
  python scripts/summarize_paired.py
  python scripts/training_stats.py
  python scripts/large_scale.py zeroshot --scrambles 20
  python scripts/large_scale.py finetune --episodes 200
  python scripts/large_scale.py compete --scrambles 15
  python scripts/make_figures.py
  python scripts/make_tables.py
  (cd report && pdflatex -interaction=nonstopmode main.tex >/dev/null && pdflatex -interaction=nonstopmode main.tex >/dev/null && echo "report/main.pdf rebuilt")
fi
