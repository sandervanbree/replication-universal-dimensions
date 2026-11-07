# LAION-fMRI universality replication

Code and results for Kunal Basu and Sander van Bree's re:vision replication of
Chen and Bonner, *Universal dimensions of visual representation*.
The original implementation is in `orig_codebase/`; its source commit is recorded
in `ORIG_CODEBASE.txt`, and its MIT license is retained there.

Use Python 3.10 on Linux:

```bash
python -m pip install -r requirements.txt
python scripts/check_results.py
```

The quick check verifies the included metadata and saved results, including the
generalization p-values from their saved null distributions. It does not refit models.

- `src/` and `scripts/`: input preparation and primary analysis.
- `generalization/`: exact split assignments, reliability estimates, and permutation analysis.
- `supplementary/`: participant statistics, rank-tolerance sensitivity, and report summaries.
- `metadata/`: ordered image IDs and model/layer inventory.
- `results/`: reference tables, figures, and compact generalization null distributions.

Large image, beta, feature, and PCA files are not included. Obtain the
[LAION-fMRI dataset](https://laion-fmri.hebartlab.com/), including the images and
single-trial betas, then run from this directory:

```bash
bash scripts/prepare_inputs.sh /path/to/LAION-fMRI
bash scripts/run_analysis.sh
```

Preparation writes to `data/`; analysis writes to `outputs/`. If using existing
prepared data, set its locations in `config/analysis.json` and skip preparation.
The scripts default to two workers and two numerical threads per worker;
`WORKERS` and `THREADS` can be set in the shell. Full reproduction requires
substantial computation and disk space. Use a fresh output directory when changing
analysis parameters; completed scores are reused.

The primary analysis uses the fixed unique-image PCA fit and 5,000 permutations.
Generalization tests shuffle held-out predictions, with brain similarity fixed;
CLIP-fold correlations and their nulls are averaged before calculating p-values.
The original-tolerance analysis is a supplementary sensitivity check.
