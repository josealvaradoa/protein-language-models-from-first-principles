# Week 4 reproduction dashboard

The local reproduction dashboard is a small experiment-control surface. It
shows the fixed three-stage sequence for Weeks 1 through 3: Verify,
Reevaluate, and Retrain. Historical values are display-only evidence. They are
not presented as dashboard runs.

Launch it from a named feature branch, never `main` or `master`:

```bash
env PYTHONPATH=src uv run --locked --offline python scripts/run_reproduction_dashboard.py
```

The server binds only to `http://127.0.0.1:8765`. It exposes the dashboard
page, one stylesheet, one JavaScript asset, and a small same-origin API. It
does not serve arbitrary files.

Available checks are immutable server-catalog jobs. The UI shows the exact
command display and asks for confirmation, then sends only the selected job
identifier with a CSRF token. The browser cannot provide command arguments.
Each run records its branch, revision, timestamps, status, exit code, and log.
Run logs live under `runs/dashboard/<run-id>/log.txt`.

Verify runs fixed local setup or public-report checks. Reevaluate means rerun
the approved measurement contract. For Week 1, that is a split audit, not a
model evaluation. Retrain is a distinct fit stage and is not applicable to
Week 1.

The Week 1 audit reproduction plus Week 2 and Week 3 reevaluation and retrain
actions are currently locked with `reproduction_contract_pending`. The active
Week 4 section of `syllabus/protein-language-models-syllabus.md` in the parent
interview-prep workspace and frozen
[`foundations_reproduction_v1.toml`](../experiments/week_04/foundations_reproduction_v1.toml)
define the future checks, but they do not make those runs available. This
dashboard does not claim that full reproduction is available yet.
