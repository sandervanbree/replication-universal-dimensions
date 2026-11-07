#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS="${THREADS:-2}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export GENERALIZATION_THREADS="$OMP_NUM_THREADS"
export PYTHONDONTWRITEBYTECODE=1
export BONNER_CACHING_HOME="$PWD/data/cache/bonner-caching"
export BONNER_DATASETS_HOME="$PWD/data"
export BONNER_MODELS_HOME="$PWD/data/cache/bonner-models"
export TORCH_HOME="$PWD/data/cache/torch"
export HF_HOME="$PWD/data/cache/huggingface"
export MPLCONFIGDIR="$PWD/data/cache/matplotlib"

# Operation depths depend on architecture and are supplied with the results.
python - <<'PY'
from pathlib import Path
import shutil
from src.replication.config import load_config
config = load_config('config/analysis.json')
config.make_output_dirs()
target = config.tables_dir / 'node_depths.csv'
if not target.exists():
    shutil.copy2('results/main/node_depths.csv', target)
PY
python scripts/run_local.py --config config/analysis.json --workers "${WORKERS:-2}" \
    --threads-per-worker "${THREADS:-2}"
python generalization/run.py prepare --workers "${WORKERS:-2}"
python generalization/run.py smoke --workers "${WORKERS:-2}"
python generalization/run.py run --workers "${WORKERS:-2}"
python generalization/summarize_results.py
python - <<'PY'
import os
import subprocess
import sys
from src.replication.config import load_config
config = load_config('config/analysis.json')
report = config.output_root / 'report'
workers = os.environ.get('WORKERS', '2')
def run(script, *args):
    subprocess.run([sys.executable, script, *map(str, args)], check=True)
run('supplementary/rank_sensitivity.py', '--pca-root', config.pca_root,
    '--output', report / 'rank_tolerance.csv', '--workers', workers)
run('supplementary/report_stats.py', '--tolerance', report / 'rank_tolerance.csv',
    '--output', report, '--workers', workers)
run('supplementary/report_tables.py', '--tables', config.tables_dir, '--output', report)
run('supplementary/build_numbers.py', '--tables', config.tables_dir, '--output', report)
run('supplementary/plot_rsa.py', '--tables', config.tables_dir, '--output', config.figures_dir)
PY
