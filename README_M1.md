# T16 Checkpoint 3 — M1 package

Owner: Dilnaz Bekturova. Proposed role: M1, model/matrices, S1, H1. Agree this role with the team and record the current allocation in the main README; do not rewrite earlier Git history.

## Run from this package root

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p "test_model.py" -v
python -m experiments.run_m1
```

This is an integration package for the existing repository, not a replacement for the team's repo. Do not overwrite teammates' files or existing `__init__.py` files. The copied public generator is supplied unchanged; source: course LMS, checkpoint3_generator_PUBLIC.py. Its docstring refers to ItO2025; we use the exact file supplied for the current assignment.

## Integration contract

```python
from src.model.cloud import build_matrices, build_pulp_model
mm = build_matrices(data)  # data = cp34_instance(16, 'B', 3)
model, y, t = build_pulp_model(matrix=mm)
```

M2 can call `model.solve(...)`, then read `y[k,l]`, `t[l]` and the objective. M1 deliberately does not solve the main LP or claim its objective/fractionality.

For M3: `mm.c`, `mm.A`, `mm.b`, `mm.senses`, `mm.lower`, `mm.upper`; all index orders are explicitly stored in `mm.columns`, `mm.rows`, `mm.families`. Read PuLP primal values by these names, not the solver's sorted variable order. Read row duals by row name. B4 is stored as <=, so its price has the opposite sign to the original >= row. Fixed variables (lower=upper=0) must be treated as fixed by the KKT checker; they do not have a one-sided reduced-cost condition without bound multipliers.

For M4: designated assignment row is `assign_9`, `mm.rows.index('assign_9') == 9`; RHS is 1. Rebuild a fresh model for each perturbation; modify `mm.b[9]` or the named PuLP row, consistently. Do not perturb CPU capacity instead.

For the common runner owner: call `experiments.run_m1.main()` from the existing `experiments/run_all.py`; preserve the other roles' work. Required command from the instructions is `python -m experiments.run_all`.

## Outputs

- results/m1_matrices.npz: all matrices and indexing metadata
- results/m1_rows.csv, m1_columns.csv: readable bookkeeping
- results/m1_model_sizes.csv: actual PuLP model sizes
- results/h1_binary_points.csv: eight-point enumeration
- results/T16_B3_relaxation.lp: inspectable reference LP

## Report

`docs/M1_report.tex` contains Task 1, a short signed-by-name S1 section and a one-page H1 typeset worked solution. Review the solution, do the required derivation on paper yourself and add your real signature/date; no signature is fabricated. Replace the report's corresponding M1 sections only. Check the complete team's final six-page main-text limit separately.

## Declarations — suggested accurate entry

Dilnaz Bekturova used ChatGPT/Codex to draft `src/model/cloud.py`, `src/model/triangle.py`, `tests/test_model.py`, `experiments/run_m1.py`, and the Task 1/S1/H1 report material. Assistance included translating the prescribed model to matrices, debugging, test design and numerical checks. The course generator was copied unchanged from the LMS. The member must review, understand and verify the submitted work, complete the hand derivation, and commit their contribution under their own real Git identity. List this help in the team's README and PDF Declarations. Do not describe AI-drafted source as unaided authorship.

## Suggested small commits after review

[Model] Add B3 reference matrices and GPU bounds
[Model] Add PuLP adapter and row-column bookkeeping
[Model] Test reference coefficients and H1 enumeration
[Report] Add M1 formulation, S1 and signed H1

Do not create a new repository, push, or change tags just to use this package. The user will integrate and commit it in the team's existing private repository.
