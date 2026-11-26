# Conflict Vision — Reproducibility Artifact

Reproduces the DICES analysis and the synthetic query-driven
qualification experiment accompanying the paper.

Validated on Windows with Python 3.11.3.
Dependencies: NumPy 2.3.5 and pandas 2.2.3.

## Run

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\run_all.py
```

The command runs the tests and both experiments.
Outputs are written to results/local_run/.
DICES results are checked against the reference with tolerance 1e-10.

The synthetic experiment uses seed 20261126 and 300 instances.
It supports the revised manuscript table; it does not reproduce
the earlier table whose original code was unavailable.

See DATA_NOTICE.md for dataset attribution and licensing.