# Reproducibility

This repository is designed so a reviewer can reproduce the public computational evidence with standard Python tooling.

## Environment

Recommended:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## One-command validation

```bash
python validation/run_validation_suite.py --output-dir results/sample_outputs
```

## Expected outputs

The suite creates CSV files:

- `simulation.csv`
- `independent_reproduction.csv`
- `independent_reproduction_summary.csv`
- `invariance_scan.csv`
- `invariance_scan_summary.csv`
- `continuity_analysis.csv`

## Notes

The public validation scripts are cleaned reference implementations. They are not a full archive of all exploratory scripts. Exploratory internal version numbers were intentionally replaced by descriptive module names.

## Determinism

The scripts use fixed random seeds by default. Floating-point values may vary slightly across platforms or NumPy versions.
