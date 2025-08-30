# Selection Perturbative Hyper‑Heuristic (SPHH)

This is my reproducible Python implementation of a **single‑point Selection Perturbative Hyper‑Heuristic (SPHH)** with a benchmark suite (f1–f24) from the Research Paper: **"A modified particle swarm optimization algorithm based on velocity updating mechanism by Chunfeng Wang and Wenxin Song"** and a full experiment pipeline: run all problems across multiple seeds, aggregate results, export publication‑ready tables, and plot **log‑scaled optimality‑gap** convergence curves.

> Python 3.10+ is recommended. Dependencies: **numpy**, **matplotlib**.

---

## ✨ What’s inside

- **SPHH optimizer & benchmark registry** — `single_point_sphh.py` defines:
  - Six low‑level perturbative heuristics (LLHs): `gaussian_full`, `gaussian_kdims`, `cauchy_full`, `random_reset_coord`, `opposition_blend`, `pull_to_best`.
  - Selection modes: **UCB1** bandit (`selection_mode="ucb"`) and **uniform random** (`"random"`).
  - Acceptance modes: **Simulated Annealing** (`acceptance_mode="sa"`) and **Greedy** (`"greedy"`).
  - Step‑size self‑adaptation using the **1/5 success rule** (update every 50 evals).
  - A small `OBJECTIVES` registry mapping names (e.g., `f3_D10`) to `(func, lo, hi, D)`.

- **Experiment config** — `exp_config.py` centralizes **seeds** and **per‑dimension evaluation budgets**, plus the `results/` directory path.

- **Batch runner** — `run_suite.py` iterates over all objective keys × seeds and writes one row per run to `results/seeds_all.csv`.

- **Aggregator** — `aggregate_suite.py` groups by function key and writes `results/summary_all.csv` with mean/std/best and runtime stats.

- **Table builders** — `make_tables.py` → `results/tables.md` (Markdown) and `results/tables.tex` (LaTeX).

- **Params snapshot** — `make_params_table.py` records your final settings to `results/params.md` / `results/params.tex` (selection/acceptance, budgets, etc.).

- **Convergence plots** — `plot_convergence.py` re‑runs selected base functions (`BASE_FUNCTIONS = ["f24"]` by default) and writes **PNG plots** to `results/plots/` and median **CSV traces** to `results/traces/` using **|f − f\*|** on a log y‑axis.

---

## 🚀 Quickstart (full pipeline)

```bash
# 1) (optional) create and activate a virtualenv
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 2) install dependencies
pip install numpy matplotlib

# 3) run all objectives across all seeds; write per‑run CSV
python run_suite.py

# 4) aggregate per‑run CSV into a per‑function summary
python aggregate_suite.py

# 5) build Markdown/LaTeX tables from the summary
python make_tables.py

# 6) record the experiment parameters you used (Markdown + LaTeX)
python make_params_table.py

# 7) (optional) plot convergence curves and write CSV traces
python plot_convergence.py
```

All outputs are written to the **`results/`** folder (created automatically). See the tree below.

---

## ⚙️ Setup

- **Python**: 3.10+ recommended  
- **Packages**: `numpy`, `matplotlib`  
- **Working dir**: run all commands from the repo root

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
pip install numpy matplotlib
```

---

## 🧪 SPHH in a nutshell

- **Single‑point search**: maintain one incumbent `x` within bounds `[ℓ, u]^D`.
- **Operator pool (LLHs)**: propose `x'` via one of six perturbation heuristics, then clamp to bounds.
- **Heuristic selection**: choose the next LLH using **UCB1** (or **random** for ablations).
- **Move acceptance**: accept with **SA** (or **greedy** for ablations).
- **Self‑adaptation**: per‑dimension step sizes shrink/expand via the **1/5 success rule**.
- **Verbose single run**: run `python single_point_sphh.py` to print iterations for a chosen problem.

Key constructor args (see `single_point_sphh.py`):
```python
SPHH(
  objective, bounds=(lo, hi), dim=D, max_evals=...,
  seed=..., selection_mode="ucb"|"random",
  acceptance_mode="sa"|"greedy", ucb_c=1.5, cooling_frac=0.2,
  init="random", verbose=False, print_every=...
)
```

---

## 📦 Repository structure & outputs

```
results/
  seeds_all.csv         # one row per (objective key, seed) run
  summary_all.csv       # aggregated stats per function key
  tables.md             # Markdown tables (by base function × D)
  tables.tex            # LaTeX tables
  params.md             # experiment parameters (human‑readable)
  params.tex            # LaTeX version of parameters
  plots/
    conv_gap_<base>.png # log‑scale optimality‑gap plots by D
  traces/
    trace_gap_<key>.csv # median optimality‑gap vs eval index
```

---

## 🛠️ Customization

### Seeds & budgets
Edit **`exp_config.py`**:
- `SEEDS = list(range(10))` → change number/values of seeds.
- `MAX_EVALS_BY_D = {10:..., 30:..., 50:...}` → per‑dimension evaluation budgets (fallback to `MAX_EVALS_DEFAULT`).

### Which objectives run
`run_suite.py` runs **every key** in `OBJECTIVES` (e.g., `f3_D10`, `f3_D30`, `f3_D50`, …). To restrict:
- Temporarily comment out entries in `OBJECTIVES` in `single_point_sphh.py`, **or**
- Add a filter inside `sorted_keys()` in `run_suite.py`.

### SPHH modes & verbosity
- Suite defaults: `selection_mode="ucb"`, `acceptance_mode="sa"`, `verbose=False` (set inside `run_suite.py`).
- For a one‑off verbose run, execute `python single_point_sphh.py` and adjust `which`, `max_evals`, `verbose`, `print_every` at the bottom.

### Convergence plots
- Edit `BASE_FUNCTIONS` in `plot_convergence.py` to select base functions (e.g., `"f17","f21","f23","f4"`).
- Update `FSTAR_BY_BASE` for any base with **non‑zero** optimum (e.g., `f2`, `f17`) to keep the **optimality‑gap** correct.

---

## 📊 Tables & figures

- `make_tables.py` generates compact tables with: `D`, `f_best`, `mean_f_best`, `f_best_std_dev`, `mean run time (s)` for each base function.
- `plot_convergence.py` produces log‑scale optimality‑gap curves by dimension and writes the per‑base median gap traces as CSV for reproducibility.

---

## 🧩 Reproducibility notes

- All randomness is seeded via `exp_config.SEEDS` and numpy’s `default_rng` in `SPHH`.
- Budgets are controlled centrally (`exp_config.budget_for_dim(D)`) so changing one place updates the entire pipeline.
- The aggregator and table builders are pure functions of the CSV outputs – delete `results/` and re‑run to regenerate everything.

---




