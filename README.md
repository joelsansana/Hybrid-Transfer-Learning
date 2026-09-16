# Transfer Learning with Hybrid Models

A reference implementation of **transfer learning** combined with **hybrid semi-physical / data-driven modelling** for a continuous stirred-tank esterification reactor. The motivating application is the chemical process industry (CPI), where plants are forced out of normal operating conditions (NOC) by changes in feedstock or market demand, making historical purely data-driven models obsolete.

The reaction modelled is the biodiesel-relevant transesterification/esterification

$$
\text{TG} + 3\,\text{M} \;\longrightarrow\; \text{G} + 3\,\text{E}
$$

(triglyceride + methanol → glycerol + ethyl ester), tracked through 4 mass balances and 1 energy balance. The target Key Performance Indicator (KPI) is the ethyl-ester mole fraction, `xRE`.

---

## Table of contents

1. [Why hybrid + transfer learning?](#why-hybrid--transfer-learning)
2. [Installation](#installation)
3. [Repository layout](#repository-layout)
4. [The reactor model (white-box)](#the-reactor-model-white-box)
5. [Data-driven regressors (black-box)](#data-driven-regressors-black-box)
6. [Hybrid strategies](#hybrid-strategies)
7. [Transfer-learning strategies](#transfer-learning-strategies)
8. [Datasets](#datasets)
9. [Running the demo](#running-the-demo)
10. [Outputs](#outputs)
11. [Known issues & deprecations](#known-issues--deprecations)
12. [Citation](#citation)
13. [License](#license)

---

## Why hybrid + transfer learning?

A single model class rarely fits both NOC and the original operating envelope:

- **Pure first-principles (white-box)** models extrapolate well outside NOC but cannot capture plant-specific biases, fouling, or feedstock variability without re-tuning.
- **Pure data-driven (black-box)** models fit the source operating envelope accurately but degrade sharply when fed out-of-distribution inputs.
- **Hybrid models** combine both: a mechanistic backbone carries physical consistency while a data-driven residual compensates for un-modelled effects.
- **Transfer learning** lets a model trained on the source NOC adapt to a target NOC using only a short labelled window.

This repo benchmarks four hybrid configurations and two transfer-learning configurations on a 4-day target horizon.

---

## Installation

```bash
git clone <repo-url>
cd Hybrid-Transfer-Learning
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

GEKKO requires a local IPOPT/APOPT build; on first import it will download a solver if not present.

---

## Repository layout

| File | Purpose |
|------|---------|
| `whitebox.py` | Pure SciPy implementation of the reactor ODE + `least_squares` parameter identification and sliding-window dynamic-rate identification. |
| `whitebox_gekko.py` | GEKKO-based implementation of the same model. Used by the demo driver. Supports both single-parameter regression (`dynamic=False`) and time-varying rate identification (`dynamic=True`). |
| `blackbox.py` | `ML` class wrapping nine scikit-learn regressors with `StandardScaler` + `GridSearchCV` (10-fold, `neg_root_mean_squared_error`). |
| `hybrid_transfer_learning.py` | End-to-end driver. Loads both datasets, trains WB/BB/KAH/PCoopH/SH, evaluates on 4 test windows, and benchmarks TL-KAH / TL-PH. Notebook-style (`#%%` cell markers). |
| `datasets/dataset1/` | Source-domain training data (`Full-Factorial_train`). |
| `datasets/dataset2/` | Target-domain test data (`Full Factorial_test`). |
| `requirements.txt` | Pinned dependency set. |
| `LICENSE` | MIT. |

---

## The reactor model (white-box)

The mechanistic model tracks five states: `[xTG, xM, xG, xE, TR]` (4 component mole fractions + reactor temperature in K). Inputs are `[No, Nm, To, Tm]` (molar flows of oil/methanol in mol·s⁻¹ and their feed temperatures in K).

Conservation laws (paraphrased from `whitebox.py:90`):

```
d(xR)/dt = (Nm*(xm - xR) + No*(xo - xR) + rx*VR) / nR
dTR/dt   = (Nm*cp_m*(Tm - TR) + No*cp_o*(To - TR) + VR*Σ(-dHr*r)) / (nR * cp_R)
```

where `rx = [-r, -3r, r, 3r]` (stoichiometry), `r` is a single lumped kinetic rate, and the rest are physical properties (molar masses, densities, heat capacities, reactor volume, reaction enthalpy).

Two solver backends are provided:

- **`whitebox.Reactor`** — `scipy.integrate.solve_ivp` (LSODA) with `scipy.optimize.least_squares` for parameter identification. Also implements `dyn_rates`, a sliding-window rate tracker that re-fits `r` between consecutive samples.
- **`whitebox_gekko.Reactor`** — GEKKO model solved in `IMODE=7` (sequential simulation) for prediction and `IMODE=2` (regression) for parameter ID. Single scalar rate is returned when `dynamic=False`; a time-varying rate vector when `dynamic=True`.

Hard-coded physical constants (duplicated in both files — see [Known issues](#known-issues--deprecations)):

| Component | M (kg/mol) | ρ (kg/m³) | cp (J/(kg·K)) |
|-----------|-----------:|----------:|---------------:|
| TG (oil)  | 0.853      | 954       | 2110           |
| M (methanol) | 0.032    | 757       | 2785           |
| G (glycerol) | 0.092    | 1340      | 2556           |
| E (ester) | 0.286      | 844       | 2146           |

Reactor volume `VR = 20 m³`, reaction enthalpy `ΔHr = −6309 J/mol`.

Initial steady state used as the ODE seed:

```
x0 = [0.0031, 0.4235, 0.1432, 0.4302, 333.55]   # [xTG, xM, xG, xE, TR/K]
```

---

## Data-driven regressors (black-box)

`ML` (`blackbox.py`) wraps each candidate in `Pipeline(StandardScaler, estimator)` and tunes it via `GridSearchCV(n_jobs=-1, cv=10, scoring='neg_root_mean_squared_error')`. Available estimators:

| Key  | Estimator                | Tuned hyperparameters |
|------|--------------------------|------------------------|
| `ols`   | `LinearRegression`       | `fit_intercept`        |
| `lasso` | `Lasso`                  | `alpha`                |
| `pls`   | `PLSRegression`          | `n_components`         |
| `cart`  | `DecisionTreeRegressor`  | `max_depth`            |
| `knn`   | `KNeighborsRegressor`    | `n_neighbors`          |
| `rfr`   | `RandomForestRegressor`  | `n_estimators`         |
| `gbr`   | `GradientBoostingRegressor` | `max_features`, `n_estimators`, `learning_rate` |
| `svr`   | `SVR(cache_size=1000)`   | `kernel`, `C`, `epsilon`, `gamma` |
| `mlp`   | `MLPRegressor`           | `hidden_layer_sizes`, `learning_rate_init` |

Usage:

```python
from blackbox import ML
model = ML()
model.train(X_train, y_train, method='lasso')   # prints best params + CV RMSE
y_pred = model.predict(X_test)
```

---

## Hybrid strategies

The driver (`hybrid_transfer_learning.py`) builds four configurations per candidate regressor, all evaluated on the same four test windows.

### 1. Data-Driven (DD) — baseline
`y_DD = f_BB(X)` trained on source features only.

### 2. White-Box (WB) — baseline
`y_WB = xR[3]` from the mechanistic model fitted to source data.

### 3. Parallel Cooperative Hybrid (PCoopH)
BB learns the residual against the WB prediction:

```
y_WB = WB(X)
y_PCoopH = y_WB + f_BB(X; y_train − y_WB)
```

This is the classic von Stosch–cooperative hybrid structure.

### 4. Knowledge-Augmented Hybrid (KAH)
WB prediction is appended as an extra feature for BB:

```
X_aug = [X | y_WB]
y_KAH = f_BB(X_aug)
```

This way the black-box has access to the mechanistic guess and can learn corrections.

### 5. Serial Hybrid (SH)
The black-box predicts the kinetic rate `r`, which is then fed into the WB ODE:

```
r_hat = f_BB(X)
y_SH = WB(X; r=r_hat)             # simulated trajectory
```

`dynamic=True` mode of `whitebox_gekko.Reactor.train` is used here to obtain the time-varying rate target.

The driver uses `[LASSO, MLP, PLS]` as the candidate set (`hybrid_transfer_learning.py:88`).

---

## Transfer-learning strategies

Two TL configurations benchmark the value of using the first-day KAH prediction as a feature for adapting to subsequent days. Both assume a 7-day labelled window for fitting and predict on day 7+ (`7*24` hourly samples per day after subsampling).

### TL-KAH
```
features = [X_target | y_WB_target | y_KAH_LASSO(day 1)]
```
A new LASSO is fit per test window on these augmented features.

### TL-PH
A LASSO residual is fit against the day-1 KAH prediction:
```
y = y_KAH(day 1) + f_LASSO(X_target; y_target(day 1) − y_KAH(day 1))
```
then evaluated on day 7+.

The two benchmarks are a fresh `LASSO.fit(X_target[7d+])` and the frozen KAH model with no day-7 adaptation.

---

## Datasets

Both datasets are BDsim-style full-factorial experiments logged at 60 s cadence and downsampled to hourly resolution by `hybrid_transfer_learning.py:21-24` (`df.iloc[:-1:60, :]`).

| Column          | Unit    | Meaning                          |
|-----------------|---------|----------------------------------|
| `t/s`           | s       | Time since experiment start      |
| `TR/C`          | °C      | Reactor temperature              |
| `Foil/(kg/h)`   | kg/h    | Oil feed flow                    |
| `DPfilter/Pa`   | Pa      | Filter pressure drop             |
| `T_meth/K`      | K       | Methanol feed temperature        |
| `Fmethanol/kg/h`| kg/h    | Methanol feed flow               |
| `T_in_oil/K`    | K       | Oil feed temperature             |
| `xRE Lab`       | —       | Ethyl-ester mole fraction (KPI)  |

- `dataset1/Full-Factorial_train.{xlsx,jmp}` — source-domain (in-NOC) data.
- `dataset2/Full Factorial_test.{xlsx,jmp}` — target-domain (out-of-NOC) data, split into 4 contiguous 7-day test windows in the driver.

Auxiliary CSVs (`csv_inputs.csv`, `csv_states.csv`, `csv_setpoints.csv`, `measurements_spectra_reactor.csv`) and reference figures (`KPI_*.png`, `measurements_*.png`, `spectra.png`) ship with each dataset for further analysis.

---

## Running the demo

```bash
python hybrid_transfer_learning.py
```

The script is organised as Spyder/Jupyter cells separated by `#%%` markers, so it can also be opened as a notebook in either IDE.

Runtime is dominated by the 10-fold `GridSearchCV` over the three candidate regressors and the GEKKO solve — expect tens of minutes on a laptop.

A `figures/` directory (gitignored) will be created with the result plots.

---

## Outputs

All figures are saved under `figures/`:

| File | Contents |
|------|----------|
| `DD_profiles.png` | Test 1–4 actual vs. data-driven predictions |
| `PCoopH_profiles.png` | Parallel-cooperative-hybrid predictions |
| `KAH_profiles.png` | Knowledge-augmented-hybrid predictions |
| `SH_profiles.png` | Serial-hybrid predictions |
| `HM_RMSE.png` | 2×2 RMSE bar chart (DD / KAH / PCoopH / SH × LASSO/MLP/PLS) per test window |
| `TL_RMSE.png`, `TL_MAPE.png` | Transfer-learning RMSE / MAPE bar charts |
| `TL_1_RMSE.png`, `TL_4_RMSE.png` | Headline transfer-learning vs. LASSO vs. KAH bars |
| `TL_profiles.png` | Day-7+ actual vs. TL-KAH / TL-PH / LASSO / KAH |

Final stdout reports wall-clock run time.

---

## Known issues & deprecations

A short audit of things worth fixing before publishing or extending:

1. **Asymmetric `Reactor` interfaces.** `whitebox.Reactor.train(time, u, xvals, par0)` vs. `whitebox_gekko.Reactor.train(t, y, u, par0, dynamic)` — argument order and semantics differ. The demo only uses the GEKKO one. Pick one and document the other as a fallback.
2. **Duplicated physical constants.** M, ρ, cp, VR, ΔHr appear verbatim in both white-box files. Extract a shared `constants.py`.
3. **Deprecated APIs.**
   - `plt.style.use('seaborn-white')` → use `'seaborn-v0_8-white'` or simply `'seaborn-whitegrid'`.
   - `mean_squared_error(..., squared=True)` is removed in scikit-learn ≥ 1.4; use `mean_squared_error(...)` (MSE) or `root_mean_squared_error(...)` (RMSE).
4. **Magic numbers.** `720` (samples/day after downsampling), `7*24` (7 days × 24 hourly samples), `720+5`, `2*720+10`, `4*720+10` slicing offsets — define as named constants.
5. **No module structure.** `hybrid_transfer_learning.py` is a script with cell markers, not importable. Refactor into `models/`, `data/`, `experiments/` packages for reuse.
6. **No tests.** Even smoke tests around the `ML` and `Reactor` interfaces would catch regression when sklearn/GEKKO APIs shift.
7. **`Reactor.predict` in `whitebox.py` ignores `xvals`.** Only the 4th state component is used as a target for parameter ID; this should be explicit in the docstring.
8. **`Reactor.train` in `whitebox_gekko.py` has no docstring** and the `dynamic` branch returns either a list or an ndarray depending on `dynamic`, which downstream code in the driver does not handle uniformly (`hybrid_transfer_learning.py:206` calls `wb_model.train(..., par0=par, dynamic=True)` and feeds the result to a regressor without normalisation).
9. **`ML.param_dict_maker`** mutates `self` while also returning the dict; rename to `_build_param_grids` and drop the side effect.
10. **No `requirements.txt`** until this commit. Now resolved.

---

## Citation

If you use this code in academic work, please cite:

> Sansana, J., Rendall, R., Castillo, I., Chiang, L., & Reis, M. S. (2024). **Hybrid modeling for improved extrapolation and transfer learning in the chemical processing industry.** *Chemical Engineering Science*, 300, 120568. https://doi.org/10.1016/j.ces.2024.120568

BibTeX:

```bibtex
@article{sansana2024hybrid,
  title   = {Hybrid modeling for improved extrapolation and transfer learning in the chemical processing industry},
  author  = {Sansana, Joel and Rendall, Ricardo and Castillo, Ivan and Chiang, Leo and Reis, Marco S.},
  journal = {Chemical Engineering Science},
  volume  = {300},
  pages   = {120568},
  year    = {2024},
  doi     = {10.1016/j.ces.2024.120568}
}
```

The esterification kinetics and BDsim-style datasets are adapted from publicly available benchmark simulations; see the `datasets/` README for sources.

---

## License

MIT — see `LICENSE`. © 2022 Joel Sansana.
