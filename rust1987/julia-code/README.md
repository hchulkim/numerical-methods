# Rust (1987) Bus Engine Replacement Model — Julia

A Julia translation of the MATLAB "Zurcher" code by **Fedor Iskhakov, Bertel
Schjerning, and John Rust** (`../matlab-code/`, August 2021 version), which
implements solution, estimation, simulation, and equilibrium analysis of the
bus engine replacement model of

> Rust, John (1987). "Optimal Replacement of GMC Bus Engines: An Empirical
> Model of Harold Zurcher." *Econometrica* 55(5), 999–1033.

The translation is **verified against the original**: estimating on Rust's
bus data (all four bus groups, `n = 175`, `β = 0.9999`) reproduces the
reference estimates `RC = 9.7686`, `c = 1.3428` used in the MATLAB scripts,
all analytic gradients are checked against finite differences, and the NPL
and NFXP estimators agree with each other to four decimals (see
`test/runtests.jl`).

---

## 1. The model in one page

Harold Zurcher manages a bus fleet. Each month, for each bus with discretized
odometer state `x ∈ {1, …, n}`, he chooses to

* **keep** the engine: utility `−0.001·c·(x−1) + ε_keep`, mileage moves up
  `j−1` grid cells with probability `π_j` (estimated from the data), or
* **replace** it: utility `−RC + ε_repl`, the odometer resets to zero and
  then accumulates one month of mileage.

The shocks `ε` are i.i.d. Type-1 Extreme Value, so the *integrated* (expected)
value function satisfies the smooth Bellman equation

```
V(x) = log( exp(vK(x)) + exp(vR) )            (log-sum-exp, recentered in code)
vK(x) = u_keep(x) + β·E[V(x′) | x, keep]
vR    = u_replace + β·E[V(x′) | replace]
```

and choice probabilities are logit: `P(keep|x) = 1/(1 + exp(vR − vK(x)))`.

Rust's original formulation iterates instead on the **expected value
function** `ev(x) = E[V(x′) | x, keep]` ("EV space"). Both formulations are
implemented (`EVBellman`, `IVBellman`); they are connected by `ev = P_keep·V`
and produce identical choice probabilities (tested).

**Estimation problem.** Given panel data on mileage `x` and replacements `d`,
estimate `θ = (RC, c, π)` by maximum likelihood. The catch: every trial value
of `θ` requires re-solving the fixed point — hence *Nested Fixed Point*
(NFXP):

* **outer loop** — Newton trust-region on the likelihood with analytic
  gradients and a BHHH Hessian approximation,
* **inner loop** — the fixed point `ev = Γ(ev)`, solved by the
  *poly-algorithm* below.

## 2. The poly-algorithm (the heart of NFXP)

`Γ` is a contraction with modulus `β = 0.9999`, so plain successive
approximations (SA) converge but at rate `0.9999` — uselessly slow near the
fixed point. Newton–Kantorovich (NK) converges quadratically but only
locally. Rust's NFXP manual combines them:

1. **SA stage:** iterate `ev ← Γ(ev)`. Watch the observed contraction ratio
   `tol_i/tol_{i−1}`. When it stabilizes near `β` (within `tol_ratio = 0.01`),
   SA has entered Newton's domain of attraction — switch.
2. **NK stage:** solve `(I − Γ′(ev))·δ = ev − Γ(ev)` and step `ev ← ev − δ`,
   where `Γ′` is the (sparse) Fréchet derivative of the Bellman operator.
   Each step is followed by one extra SA application for stability and an
   exact error bound. Typically converges in 2–4 steps to tolerance `1e−12`.

On this model the whole solve takes ~10 SA + ~4 NK iterations (run
`run_fxp.jl` to watch it). During estimation the inner solver is **warm
started** from the previous fixed point, so most likelihood evaluations need
only 1–2 NK steps.

Two more ingredients make NFXP fast and exact:

* **Analytic likelihood gradient** via the implicit function theorem: at the
  fixed point, `dev/dθ = (I − Γ′)⁻¹·∂Γ/∂θ`, reusing the same sparse LU
  factorization as the NK step (`src/nfxp.jl`, steps 1–3).
* **BHHH**: the outer product of the per-observation scores is used as the
  Hessian — always positive definite, and it doubles as the asymptotic
  variance estimator (`Avar = (N·H)⁻¹`).

## 3. Files

```
julia-code/
├── Project.toml            dependency manifest
│                           (CSV, DataFrames, Optim, JuMP, Ipopt, Plots)
├── data/
│   └── busdata1234.csv     Rust's bus data, converted from busdata1234.mat
│                           (8260 rows × 9 columns, headers documented below)
├── src/
│   ├── Zurcher.jl          module wrapper — include this, `using .Zurcher`
│   ├── model.jl            primitives: utility, mileage transitions, Bellman
│   │                       operators in EV and IV space + Fréchet derivatives
│   ├── dpsolver.jl         SA, Newton–Kantorovich, and the poly-algorithm
│   ├── data.jl             readbusdata, frequency estimator, sufficient stats
│   ├── nfxp.jl             NFXP likelihood (analytic gradient + BHHH) + estimator
│   ├── mpec.jl             MPEC estimation via JuMP + Ipopt
│   ├── npl.jl              NPL / CCP estimation, Ψ = Λ∘φ operator, BBL objective
│   ├── simulate.jl         panel simulation, ergodic distribution, demand curve
│   ├── msm.jl              method of simulated moments criterion
│   └── output.jl           printed estimate / Monte Carlo tables
├── test/
│   └── runtests.jl         correctness checks (run these first!)
└── run_*.jl                executable scripts, one per MATLAB run script
```

### Map to the MATLAB code

| MATLAB                  | Julia                         | Notes |
|-------------------------|-------------------------------|-------|
| `zurcher.m` (class)     | `src/model.jl`, `src/data.jl`, `src/simulate.jl` | split by topic |
| `dpsolver.m`            | `src/dpsolver.jl`             | identical algorithm & defaults |
| `nfxp.m` + `zurcher.ll` | `src/nfxp.jl`                 | same two-step → full-MLE flow |
| `mpec.m`                | `src/mpec.jl`                 | fmincon → JuMP + Ipopt; sparsity patterns and constraint Jacobian come from JuMP's AD instead of hand-coding |
| `npl.m`                 | `src/npl.jl`                  | `inv()` replaced by sparse LU |
| `bbl.m` (objective)     | `src/npl.jl` (`bbl_objective`)| `bbl.estim` was TODO upstream |
| `msm.m` (objective)     | `src/msm.jl`                  | `msm.estim` was TODO upstream |
| `output.m`              | `src/output.jl`               | |
| `struct2vec/vec2struct/getfields.m` | — not needed      | θ vectors handled directly |
| `run_busdata.m`         | `run_busdata.jl`              | NFXP partial + full MLE |
| `run_fxp.m`             | `run_fxp.jl`                  | verbose solver demo |
| `run_errorbound.m`      | `run_errorbound.jl`           | SA convergence plots |
| `run_dem.m`             | `run_dem.jl`                  | equilibrium + demand curve |
| `run_mc_nfxp.m`         | `run_mc_nfxp.jl`              | Monte Carlo study |
| `run_npl.m`             | `run_npl.jl`                  | NPL on the bus data |
| `run_npl_sim.m`         | `run_npl_sim.jl`              | NPL on simulated panel |
| `run_msm.m`             | `run_msm.jl`                  | MSM criterion surface |
| `run_bbl.m`             | `run_bbl.jl`                  | BBL criterion surface |
| `run_sparsity.m`, `run_nfxp_vs_mpec.m` | **not ported** | sparsity-pattern visualization is moot (JuMP detects sparsity automatically); the NFXP-vs-MPEC comparison is covered by `run_busdata.jl`, which runs both on the same data |

### Bus data columns (`data/busdata1234.csv`)

Converted losslessly from the MATLAB v4 file `busdata1234.mat` (matrix
`data`, 8260 × 9). Column meaning (inferred from `zurcher.readbusdata`):

| column         | content                                          |
|----------------|--------------------------------------------------|
| `id`           | bus identifier                                   |
| `bustype`      | bus group 1–4 (Rust's groups)                    |
| `year`, `month`| observation date (1975–1985)                     |
| `d_lag`        | lagged replacement dummy `d_{t−1}`               |
| `odometer_lag` | odometer at `t−1` (miles since last replacement) |
| `odometer`     | odometer at `t` (miles since last replacement)   |
| `odometer_raw` | cumulative odometer (never reset)                |
| `mileage_diff` | raw mileage change (negative at replacements)    |

Only `id`, `bustype`, `d_lag`, `odometer` are used in estimation.

### Dependency DAG of the source files

`src/Zurcher.jl` includes the files in topological order; an arrow `A → B`
means "B uses definitions from A":

```
                          model.jl
              (Model, utility, transitions,
            EV/IV Bellman ops + Fréchet deriv.)
               │              │            │
       ┌───────┘              │            └────────┐
       ▼                      ▼                     ▼
  dpsolver.jl              data.jl              output.jl ◄────────┐
 (SA, NK, poly)     (BusData, readbusdata,   (tables; needs       │
       │             freq_transition,         EstimationResult)   │
       │             ChoiceCounts)                                │
       │                      │                                   │
       └──────┬───────────────┤                                   │
              ▼               │                                   │
           nfxp.jl ◄──────────┘                                   │
   (likelihood fgh!, estim_nfxp, ────────────────────────────────►│
    EstimationResult, NFXPCache)                                  │
              │                                                   │
       ┌──────┴────────┐                                          │
       ▼               ▼                                          │
   mpec.jl          npl.jl ──────────────────────────────────────►│
 (JuMP+Ipopt;   (φ, Λ, Ψ, estim_npl,
  warm start     policy_matrix, BBL)
  via dpsolver)        │
                       ▼
                 simulate.jl
            (simdata, ergodic, eqb,
             equilibrium_demand)
                       │
                       ▼
                    msm.jl
              (moments, criterion)
```

So the read-through order that never references anything undefined is:

```
model.jl → dpsolver.jl → data.jl → nfxp.jl → mpec.jl → npl.jl
        → simulate.jl → msm.jl → output.jl
```

### Suggested order to run the scripts

Each `run_*.jl` is standalone (any order works), but this sequence builds
understanding incrementally — solver first, then estimators, then the
demonstrations that reuse both:

```
 test/runtests.jl          0. verify the translation on your machine
        │
        ▼
 run_fxp.jl                1. watch the poly-algorithm solve one model
        │                     (SA contraction → NK quadratic convergence)
        ▼
 run_errorbound.jl         2. why SA alone is hopeless at β≈1
        │
        ▼
 run_busdata.jl            3. headline: NFXP (partial + full MLE) and MPEC
        │                     on Rust's actual data
        ├──────────────────────────────┐
        ▼                              ▼
 run_npl.jl                4a.  run_mc_nfxp.jl        4b. sampling
   (CCP/NPL alternative          (Monte Carlo:            properties
    on the same data)             bias, MCSD)
        │
        ▼
 run_npl_sim.jl            5. NPL vs NFXP on a 595k-obs simulated panel
        │
        ▼
 run_dem.jl                6. economics: equilibrium mileage distribution
        │                     and the engine demand curve
        ▼
 run_msm.jl / run_bbl.jl   7. criterion surfaces of the simulation-based
                              estimators (slowest scripts)
```

## 4. Running

The repository's `default.nix` already provides Julia 1.11 with every needed
package. From `numerical-methods/`:

```bash
nix-shell --run "julia rust1987/julia-code/test/runtests.jl"   # checks first
nix-shell --run "julia rust1987/julia-code/run_busdata.jl"     # headline result
```

Outside nix: `julia --project=rust1987/julia-code` after
`Pkg.instantiate()`. Each script is standalone (`include`s the module
directly — no package installation required beyond the dependencies).

Expected output of `run_busdata.jl` (all four bus groups, `n=175`):

```
Method: NFXP (full MLE)
    Param.           Estimates          s.e.        t-stat
------------------------------------------------------------------------
    RC                  9.7688        1.2263        7.9661
    c                   1.3429        0.3153        4.2587
    p1                  0.1071        0.0034       31.2116
    p2                  0.5152        0.0055       93.0565
    p3                  0.3622        0.0053       68.0381
    p4                  0.0143        0.0013       10.8946
    p5                  0.0009        0.0003        2.6469
------------------------------------------------------------------------
log-likelihood    =   -8607.8894
runtime (seconds) =       0.0807
```

(The default `Model()` parameters `RC = 11.7257`, `c = 2.45569` are Rust's
Table X values for bus groups 1–3; groups 1–4 give the estimates above —
the same numbers the MATLAB scripts quote as starting values
`[9.7686; 1.3428]`.)

A library-style session:

```julia
include("src/Zurcher.jl"); using .Zurcher

m = Model()                                  # Rust's Table X parameters
ev, pk, Γ, info = solve(m)                   # poly-algorithm fixed point
data = readbusdata(m)                        # Rust's bus data
res = estim_nfxp(data, Model(RC=0, c=0))     # full MLE
print_estimates(res)

pp, pp_K, pp_R = eqb(res.m, pk)              # equilibrium mileage distribution
sim = simdata(100, 119, res.m, pk)           # simulate a panel
```

## 5. What makes the Julia version fast

The MATLAB code is already well engineered (it follows Rust's GAUSS
implementation); this port keeps its algorithms *exactly* and removes
constant-factor overhead. Measured on the bus-data full MLE (8,156
observations, 7 parameters, `n = 175`): **≈ 0.06–0.08 s end-to-end**, and
0.39 s for a 595,000-observation simulated panel. Concretely:

1. **No transition matrices in the hot path.** The keep-transition matrix is
   banded (J ≈ 5 diagonals) and the replace-transition has identical rows.
   MATLAB builds sparse `P{1}`, `P{2}` and multiplies them on every Bellman
   application; here `keep_mul!` is an O(n·J) loop and the replacement
   continuation value is a *single dot product* reused for all states
   (`replace_expectation`). No allocation, no sparse-matvec overhead.

2. **State-aggregated likelihood.** The log-likelihood depends on the data
   only through per-state keep/replace counts and mileage-increment counts
   (`ChoiceCounts`), so each evaluation is O(n) = 175 work instead of
   O(N) = 8,156 (MATLAB indexes the full data vector every call). The
   per-observation loop survives only in the BHHH outer product, where it is
   genuinely needed.

3. **One sparse LU, used twice.** The NK step factorizes `I − Γ′`; the
   gradient needs `(I − Γ′)⁻¹·∂Γ/∂θ` at the same point. The factorization is
   computed once and solved against K right-hand sides. The NPL policy
   evaluation similarly uses an LU instead of MATLAB's explicit `inv()`.

4. **Warm starts without globals.** MATLAB carries `ev0` across likelihood
   calls via `global`; here an explicit `NFXPCache` does the same job —
   thread-safe, testable, and visible in the function signature.

5. **Type-stable, allocation-free inner loops.** The Bellman operator is a
   concrete struct with preallocated buffers; `apply!` allocates nothing.
   Immutable `Model` structs replace dynamic MATLAB structs (field access
   compiles to a load, not a hash lookup).

6. **Exact boundary derivatives.** `∂Γ/∂π_j` is computed exactly including
   the clamped top-of-grid rows (the MATLAB version zeroes the last rows of
   `dbellman` as an approximation). All gradients are verified against
   central finite differences in the test suite.

## 6. Estimators implemented

| Estimator | Function | Idea |
|-----------|----------|------|
| **NFXP** (partial MLE) | `estim_nfxp(data, m; est_p=false)` | mileage process from frequencies, then MLE of (RC, c) with the fixed point re-solved inside the likelihood |
| **NFXP** (full MLE) | `estim_nfxp(data, m; est_p=true)` | adds the transition probabilities to the second step, scores include the mileage process |
| **MPEC** (Su–Judd) | `estim_mpec(data, m)` | no inner loop: the 175 expected values become decision variables and the Bellman equation becomes 175 equality constraints, handed to Ipopt with JuMP's AD. Finds the same optimum as NFXP (tested to 1e-3) in ~30 iterations |
| **NPL / CCP** (Aguirregabiria–Mira) | `estim_npl(data, m; pk0=…)` | start from nonparametric CCPs; alternate pseudo-likelihood maximization (policy evaluation is one *linear* solve — no fixed point!) with Ψ policy updates. K = 1 is the Hotz–Miller two-step estimator; iterating converges to the NFXP estimates |
| **MSM** (criterion only) | `msm_objective` | match equilibrium mileage-bin fractions, simulated or exact |
| **BBL** (criterion only) | `bbl_objective` | moment inequalities: the estimated policy must beat perturbed policies |

**An MPEC gotcha worth knowing.** With `β = 0.9999` the matrix `I − βΓ′` has
condition number ~`1/(1−β) = 10⁴`, and the *exact* Hessian of the Lagrangian
is indefinite along the resulting ridge in (RC, c, ev)-space. Ipopt's inertia
correction then degrades the Newton direction and the barrier crawls —
thousands of iterations that stall short of the optimum (the "solved to
acceptable level" exit). Switching to the positive-definite L-BFGS
approximation (`hessian_approximation = limited-memory`, set by default in
`estim_mpec`) fixes it: ~30 iterations to the exact NFXP optimum. Two further
safeguards in the port: the log-sum-exp in the Bellman constraints is
anchored on the keep value (`vK + log1p(exp(vR − vK))`) so AD never sees an
overflowing `exp(±2000)`, and the ev variables are warm-started at the exact
fixed point of the starting θ (one cheap poly-algorithm solve).

## 7. References

- Rust (1987), *Econometrica* 55(5): the model and NFXP.
- Rust (2000), "Nested Fixed Point Algorithm Documentation Manual": the
  poly-algorithm, recentering, analytic derivatives, BHHH.
- Aguirregabiria & Mira (2002), *Econometrica* 70(4): NPL.
- Hotz & Miller (1993), *ReStud* 60(3): CCP inversion.
- Bajari, Benkard & Levin (2007), *Econometrica* 75(5): BBL.
- Su & Judd (2012), *Econometrica* 80(5): MPEC.
- Iskhakov, Lee, Rust, Schjerning & Seo (2016), *Econometrica* 84(1):
  NFXP vs MPEC comparison.
