#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mode="${1:-}"
if [[ $# -gt 0 ]]; then shift; fi
case "$mode" in
  ablation)
    for k in 10 30 50; do
      python -u run_model.py --k-neighbours "$k" --alpha-values 0 0.25 0.5 0.75 1 \
        --cv-folds 5 --seed 42 --epochs 25 \
        --output-dir "outputs/ablation_k${k}" "$@"
    done
    ;;
  final)
    python -u run_model.py --models Qwen/Qwen2.5-3B-Instruct --alpha-values 0.5 \
      --k-neighbours 10 --cv-folds 1 --seed 42 --epochs 25 --output-dir outputs/final/qwen "$@"
    python -u run_model.py --models meta-llama/Llama-3.2-3B-Instruct --alpha-values 0.75 \
      --k-neighbours 10 --cv-folds 1 --seed 42 --epochs 25 --output-dir outputs/final/llama "$@"
    python -u run_model.py --models microsoft/Phi-3.5-mini-instruct --alpha-values 0.75 \
      --k-neighbours 10 --cv-folds 1 --seed 42 --epochs 25 --output-dir outputs/final/phi "$@"
    if [[ " $* " != *" --dry-run "* ]]; then
      python scripts/combine_final_results.py outputs/final
    fi
    ;;
  *) echo "Usage: $0 {ablation|final} [--resume] [--dry-run]" >&2; exit 2 ;;
esac
