#!/usr/bin/env bash
set -euo pipefail

python experiments/run_simulation.py --output results/sample_outputs/simulation.csv
python validation/independent_reproduction.py --output results/sample_outputs/independent_reproduction.csv
python validation/invariance_scan.py --output results/sample_outputs/invariance_scan.csv
python validation/continuity_analysis.py --output results/sample_outputs/continuity_analysis.csv
