# OMM–SBT: Oscillatory Mantle Model — Substantial Beating Theory

Reference implementation for the **OMM–SBT causal-kinetic framework**, a research model in which physical sectors are studied as emergent regimes of a coupled amplitude–phase–kinetic substrate.

This repository is intended for reproducibility, review, and further development. It does **not** claim to establish a validated final Theory of Everything. It provides the computational reference implementation for the manuscript and its validation claims.

## Core variables

The model state is:

\[
\Psi_t = (X_t, \Phi_t, K_t)
\]

where:

- `X`: oscillatory mantle amplitude,
- `Phi`: phase/orientation field,
- `K`: substrate kinetic field.

Derived quantities include:

- flux skeleton: \(J = |X|\nabla\Phi\),
- vorticity: \(W = \nabla \times J\),
- effective metric proxy: \(g_{\rm eff}\),
- horizon-like boundary: \(H\),
- radiation proxy: \(R\),
- Diffuse Interaction Rings (DIR): interaction regions where mantle overlap and flux convergence generate localized energetic stabilization.

## Installation

```bash
git clone https://github.com/NicolasGilbertAlbertRoux/omm-sbt-causal-kinetic-framework.git
cd omm-sbt-causal-kinetic-framework
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Quick start

Run one simulation:

```bash
python experiments/run_simulation.py --output results/sample_outputs/simulation.csv
```

Run the validation suite:

```bash
python validation/run_validation_suite.py --output-dir results/sample_outputs
```

Run individual validations:

```bash
python validation/independent_reproduction.py --output results/sample_outputs/independent_reproduction.csv
python validation/invariance_scan.py --output results/sample_outputs/invariance_scan.csv
python validation/continuity_analysis.py --output results/sample_outputs/continuity_analysis.csv
```

## Repository structure

```text
src/omm_sbt/             Core implementation
experiments/             Simulation entry points
validation/              Reproduction and validation scripts
configs/                 Default parameters
docs/manuscript/         Manuscript source placeholder
results/sample_outputs/  Reproducible sample outputs
tests/                   Smoke tests
```

## Reproducibility

All scripts use explicit random seeds by default. Output files are CSV files.

See `docs/REPRODUCIBILITY.md` for detailed instructions.

## License

Code is released under the MIT License.
