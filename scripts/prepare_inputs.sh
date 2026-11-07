#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 1 ]; then
    echo "Usage: bash scripts/prepare_inputs.sh /path/to/LAION-fMRI" >&2
    exit 2
fi
dataset_root=$(realpath "$1")
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS="${THREADS:-2}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONDONTWRITEBYTECODE=1

for mode in shared ood; do
    python scripts/cache_betas.py --data-dir "$dataset_root" --mode "$mode" \
        --all-stimuli --zscore-session
done
for model_set in resnet18_classification_imagenet1k_varied_seeds \
    classification_imagenet1k_varied_architectures resnet50_imagenet1k_varied_tasks \
    untrained_resnet18_varied_seeds; do
    python scripts/cache_model_features.py --data-dir "$dataset_root" \
        --yielder "$model_set" --full
done
python scripts/export_pca.py --features-root data/cache/bonner-models/features \
    --output-root data/pca --workers "${WORKERS:-2}" --threads-per-worker "${THREADS:-2}"
python scripts/build_generalization.py all --data-root "$dataset_root" --workers "${WORKERS:-2}"
