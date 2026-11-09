# Uncertainty budget as a first-class capability

**Status**: implemented and certified (mission `IN144-N11`) ·
**Package**: `tensorphysics.uq` ·
**Acceptance evidence**: `artifacts/uq/acceptance_evidence.json` ·
**Reproduce**: `python scripts/run_uq_acceptance.py` (~45 s)

This page documents the workflow-level UQ contract that the Tensor ecosystem can
chain on. It answers three questions in a machine-adjudicated way:

1. *which* uncertain inputs matter (first/total/closed second-order Sobol'
   indices, with confidence intervals),
2. *how wide* the resulting band on the quantity of interest is (95% interval
   width `U`),
3. *may we trust that number* (PASS / FAIL / **WITHHELD** with reason codes).

## 1. Where it sits in the gate chain

```
inputs + distributions ──▶ sampler (Scrambled Sobol'/LHS, hash-pinned)
        │                        │
        │                        ▼
        │              solver adapter fan-out  ──▶  cluster shards (hash-merged)
        │                        │
        ▼                        ▼
   PCE surrogate  ◀──── N solver runs ────▶  Saltelli/Jansen pick-freeze
   (sparse LARS,                            (independent second opinion,
    analytic Sobol',                         first/total/second order)
    ε_LOO + bootstrap CI)                            │
        └──────────────┬─────────────────────────────┘
                       ▼
        cross-check  |ΔS| (PCE analytic vs sampled)
                       ▼
        95% output interval U  vs  15% engineering threshold
                       ▼
              PASS / FAIL / WITHHELD (+ reason codes)
                       ▼
   UQ half-width  ──▶  U_input of `tensorphysics.vv.validation` (ASME V&V 20)
```

The last arrow is the point of the whole exercise: the uncertainty budget stops
being a report appendix and becomes an input to the verification gate, next to
the discretisation uncertainty `U_num`:

```
U_val = sqrt(U_num² + U_input²),      |f_sim − f_exp| ≤ U_val
```

`examples/uq_end_to_end.py` runs that chain end to end on the Holtrop-1984
resistance kernel and prints both verdicts.

## 2. Contract and provenance

The UQ workflow never imports a solver. A solver is any black box

```python
evaluate(points: np.ndarray) -> np.ndarray      # (n, d) physical → (n,)
```

wrapped in `CallableAdapter` (or anything with an `evaluate` method).  Because
everything downstream is derived from that call, the evidence has to be pinned:

| pinned object | how |
|---|---|
| uncertain-input specification | `SampleSpec.fingerprint()` = sha256 over marginals + copula |
| sample set | `sample_hash` = sha256(spec fingerprint, rule, seed, n, physical points) |
| solver results | `SampleResults.sample_hash`; a shard reporting another hash raises `SampleHashMismatch` |
| surrogate | `eps_LOO`, `eps_LOO_abs`, `r2`, active terms, truncation degree |
| indices | PCE (analytic) + Saltelli (sampled) + bootstrap CIs, both stored |
| verdict | status + reason codes + the numbers they were derived from |

Cluster dispatch uses the same hash: `scripts/uq_shard_worker.py` rebuilds the
design from `(benchmark, rule, seed, n)`, evaluates its rows, and writes
`values/indices/sample_hash`; the merge refuses shards with a different hash.
The rendered `run_cluster` template is produced by
`tensorphysics.uq.adapter.cluster_shard_command`, e.g.

```
python scripts/uq_shard_worker.py --benchmark holtrop --n 256 --seed 20240 \
    --shard {i}/{n} --out shard_{i}.npz
```

## 3. Method

### 3.1 Inputs

`Marginal` supports `uniform` (Legendre basis on `[-1, 1]`), `normal` and
`lognormal` (probabilists' Hermite basis on the standardised latent).  Every
marginal is parametrised through a standard Gaussian latent `z` and the basis
coordinate `xi` is a measure-preserving map of `z` (`xi = 2Φ(z) − 1` for
uniform, `xi = z` otherwise), so the ANOVA decomposition is identical in both
coordinates.  A correlation matrix is applied on the latent scale
(NORTA / Gaussian copula); the resulting indices are then *latent-coordinate*
indices and the verdict says so (`UQ-N01-CORRELATED`).

### 3.2 Polynomial chaos

$$ y(\xi) \approx \sum_{\alpha \in \mathcal{A}} c_\alpha \prod_i \phi_{\alpha_i}(\xi_i) $$

with orthonormal `φ_k` (verified against Gauss–Legendre / normal quadrature to
1e-14 / 1e-8 in the tests).  Coefficients come from a greedy least-angle-style
forward selection (orthogonal matching pursuit on the orthonormal design) whose
stopping rule is the analytic leave-one-out error

$$ \varepsilon_{LOO}^2 = \frac{1}{N}\sum_i \left(\frac{r_i}{1-h_{ii}}\right)^2 ,
\qquad h_{ii} = x_i^\top (X^\top X)^{-1} x_i . $$

`ε_LOO` is reported **normalised by `std(y)`**, so `0.1` means "one tenth of an
output standard deviation" and the surrogate explains ≥ 99% of the variance.

### 3.3 Indices, two routes

From the coefficients (orthonormality makes this algebraic):

```
V       = Σ_{α≠0} c_α²
S_u     = (1/V) Σ_{supp(α)=u} c_α²
S_T,i   = (1/V) Σ_{α_i>0}        c_α²
```

From model runs (pick-freeze with `A`, `B` and `A_B^u`, `V` = pooled variance):

```
S_i      = 1 − E[(f(B) − f(A_B^i))²] / (2V)          (Jansen)
S_i^alt  = E[f(B)(f(A_B^i) − f(A))] / V              (Saltelli 2010)
S_T,i    = E[(f(A) − f(A_B^i))²] / (2V)              (Jansen)
S_ij     = E[f(B)(f(A_B^ij) − f(A_B^i) − f(A_B^j) + f(A))] / V
```

The two routes are compared index by index. First and total order must agree
within **5%**, or within **3 Monte-Carlo standard errors** when the sampled
route carries bootstrap intervals (an index of 0.03 cannot be resolved to 5%
without an enormous `n_base`; the count of such noise-limited indices is
reported).  Closed second-order indices are scored at **25%**, justified by
measurement: the four-term product estimator has ~3× the sampling variance of
the first-order one at the same `n_base` (see §5, G-function row).

### 3.4 The gate

```
U             = q_{0.975}[QoI] − q_{0.025}[QoI]      (propagated through the surrogate,
relative_width = U / |mean|                           n_mc fresh draws)
PASS      : no blocking reason  and  relative_width ≤ threshold (15%)
FAIL      : no blocking reason  and  relative_width > threshold
WITHHELD  : any blocking reason
```

`U` is a **prediction interval** of the quantity of interest, *not* the
confidence interval of its mean (that one shrinks with `n_mc` and would let a
wide physical band pass by drawing more points); the mean's CI is reported
separately for completeness.

| code | meaning |
|---|---|
| `UQ-W01-NONFINITE` | solver returned NaN/Inf for ≥ 1 sample (indices are listed) |
| `UQ-W02-TOOFEW` | fewer valid samples than `max(32, 10·dim)`, or fewer than 4 per active chaos term |
| `UQ-W03-LOO` | surrogate `ε_LOO > 0.1` |
| `UQ-W04-CROSSCHECK` | chaos and Monte-Carlo indices disagree beyond tolerance |
| `UQ-W05-DEGENERATE` | output variance zero / interval not finite |
| `UQ-W06-HASH` | results carry a different sample-set hash |
| `UQ-W07-NOMC` | cross-check missing while the policy requires it |
| `UQ-W08-NOPCE` | no surrogate, so no certified budget |
| `UQ-N01…N04` | non-blocking notes (correlated inputs, noisy MC, wide index CI, cross-check disabled) |

## 4. Acceptance results

Produced by `scripts/run_uq_acceptance.py`; every number below is read from
`artifacts/uq/acceptance_evidence.json` (no hand transcription).

| id | criterion | measured | limit | verdict |
|---|---|---|---|---|
| A1 | Ishigami: \|S_PCE − S_exact\| for S1, S2, S13, ST1, ST3 | **3.89e-4** | 1e-3 | PASS |
| A2 | Ishigami Monte-Carlo relative error | **1.11 %** | 5 % | PASS |
| A3 | ε_LOO vs degree (7 points, N = 1024) | Spearman **−1.00** | ≤ −0.9 | PASS |
| A4 | ε_LOO vs sample count (4 points, fixed truncation) | Spearman **−1.00** | ≤ −0.9 | PASS |
| A5 | same spec/rule/seed → identical hash and indices | **bit-identical** | exact | PASS |
| A6 | NaN / too few samples / ε_LOO > 0.1 → WITHHELD + codes | **W01 / W02 / W03 observed** | required | PASS |
| A7 | PCE-Sobol vs Saltelli relative deviation | **1.13 %** | 5 % | PASS |
| A8 | kernels untouched, workflow solver-free, shards hash-merged, stale shard refused | see below | required | PASS |
| A9 | G-function family (d = 4, two regimes) | **1.8e-3 / 3.4e-3** | 5e-3 | PASS |
| A10 | ship benchmark: narrow → PASS, wide → FAIL | **5.37 % / 21.5 %** | 15 % | PASS |

Ishigami anchor (degree 8, N = 1024, `ε_LOO = 1.81e-2`, 26 active terms,
R² = 0.99970):

| index | closed form | PCE | \|Δ\| | Saltelli/Jansen | MC rel. error |
|---|---|---|---|---|---|
| S1 | 0.313905 | 0.313820 | 8.5e-5 | 0.317376 | 1.11 % |
| S2 | 0.442411 | 0.442796 | 3.9e-4 | 0.439818 | 0.59 % |
| S13 | 0.243684 | 0.243364 | 3.2e-4 | 0.241651 | 0.83 % |
| ST1 | 0.557589 | 0.557200 | 3.9e-4 | 0.562026 | 0.80 % |
| ST2 | 0.442411 | 0.442815 | 4.0e-4 | 0.440661 | 0.40 % |
| ST3 | 0.243684 | 0.243384 | 3.0e-4 | 0.243736 | 0.02 % |

The campaign node quotes S1 = 0.3139, S2 = 0.4424, ST = 0.5574; the PCE values
differ from those by −8e-5, +4.0e-4 and −2.0e-4, i.e. **within 1e-3 as well**.
The closed forms give 0.313905 / 0.442411 / 0.557589; the 4-decimal campaign
values are rounded copies of them (< 2e-4).

Ship benchmark (Holtrop-1984 kernel called read-only; ±1 % uniform uncertainty
on `L, B, T, Cb, Vs`; QoI = calm-water `R_T`):

| quantity | value |
|---|---|
| reference `R_T` | 404.17 kN |
| surrogate | degree 4, 63 terms, ε_LOO = 1.0e-6, R² = 1.000000 |
| `S_T`: Vs / Cb / B / L / T | 0.742 / 0.110 / 0.067 / 0.050 / 0.031 |
| 95 % band | [393.44, 415.13] kN → `U/|mean| = 5.37 %` → **PASS** |
| direct-solver band (independent 4096 runs) | 5.35 % (agreement 0.02 pp) |
| wide inputs (±4 %) | `U/|mean| = 21.5 %` → **FAIL** |

The ranking is physically expected for a displacement hull at constant speed
regime: speed dominates (∝ V²·C_T with the residual coefficient rising with
Fn), then `Cb`, then beam; draft and length are secondary under ±1 % input
uncertainty.

### Cross-architecture reproducibility

The same acceptance script was executed on the x86_64 worktree (numpy 2.4.6 /
scipy 1.16.0) and on a loongarch64 compute node (numpy 2.0.1 / scipy 1.17.1),
see `artifacts/uq/cross_arch_check.json`:

* identical sample-set hash `5c8659bd…`,
* identical ε_LOO grids (`0.8879, 0.7282, 0.5043, 0.3936, 0.1381, 0.0181, 0.0014`),
* identical index errors (3.89e-4 / 1.11 % / 1.13 %),
* 76/76 UQ tests pass on both architectures.

## 5. Limitations and traps (measured, not guessed)

1. **Two separately seeded Sobol' sequences are not independent.** Row-wise
   pairing of `A` and `B` from two `qmc.Sobol` objects gives |corr| up to 0.9
   and *N-independent* wrong indices (observed: ST1 = 1.09 instead of 0.5576,
   unchanged from N = 2048 to N = 32768).  The design therefore partitions one
   `2·n_base` scrambled set by a random permutation; the tests assert
   |corr| < 0.05 on 4096 rows.
2. **PRESS is optimistic when the active set approaches N.** At degree 6 with
   N = 64 and 30 active terms, `ε_LOO = 0.072` while the true error against
   20 000 fresh points was `0.241`; at N ≥ 512 the two agree within 5%.  Hence
   the `SAMPLES_PER_TERM = 4` guard (`UQ-W02-TOOFEW`).
3. **The sample-size LOO grid must hold the truncation fixed.** With the greedy
   selector the active set adapts to N (fewer terms at larger N), so the LOO
   *increases* with N (0.072 → 0.136 on Ishigami degree 6).  A4 therefore uses
   the full total-degree basis (`method="ols"`), which isolates the estimation
   error and is monotone.
4. **A BLAS bug can silently poison the fit.** On loongarch64 (OpenBLAS
   0.3.27) `A.T @ A` returned NaN for a perfectly finite `(1024, 8)` Gram
   matrix, and `A.T @ y` / `A @ coef` did the same; the forward path then
   stalled at 7 terms with `ε_LOO = 0.25`.  `_gram`/`_projection`/`_matmul` now
   validate every product and fall back to an `einsum` reduction.  This is why
   the same numbers come out on both architectures — do not remove the guards
   as "dead code".
5. **G-function truncation is honest but slow.** The G-function is piecewise
   linear, so its Legendre expansion decays only like `1/k²`; at d = 4, p = 6,
   N = 2048, `ε_LOO = 0.099` (just under the 0.1 gate) while the indices are
   accurate to 1.8e-3.  In the stronger-interaction regime `a = [0, 0.5, 1, 1.5]`
   the surrogate hits `ε_LOO = 0.114` → the verdict correctly **withholds**
   even though the indices are within 3.4e-3 — the gate protects against using
   a surrogate without evidence, not against a lucky answer.
6. **Correlated inputs change the meaning of the indices.** With a copula the
   indices are latent-coordinate indices (the only orthogonal decomposition
   available); the sampled route then refuses to run (`ValueError`) and the
   verdict attaches `UQ-N01-CORRELATED`.
7. **A relative gate is meaningless for a zero-mean QoI.** Ishigami's 95% band
   is 4.3× its mean (3.5 kN-scale mean, 3.7 sd), so the anchor *must* FAIL the
   15% gate; the anchor certifies the indices (A1/A2/A7), the gate is certified
   on the ship benchmark (A10).
8. **"Holton-type benchmark"** is interpreted as the Holtrop-1984 resistance
   kernel shipped in this repository — the only matching reference model here.
   The interpretation is recorded in `benchmarks.py` rather than silently
   relabelled; no analytic indices exist for it, so it is certified by the
   PCE↔Saltelli cross-check plus ε_LOO.
9. **Sparse regression is OMP, not a full LARS path.** The selector is
   orthogonal matching pursuit with a PRESS stopping rule: it takes one term
   per step, which is the "least-angle-style forward selection" used here, but
   it does **not** compute the full LARS path with the shrinkage step, and no
   L1/compressed-sensing solver is used.  On the anchors it recovers the
   analytic indices (A1), but for very anisotropic spectra a full LARS path or
   an L1 penalised fit could be cheaper in solver runs.
10. **Determinism is scoped.** Identical `(spec, rule, seed, n)` reproduces
   bit-identical points *for a given NumPy/SciPy build*; the OpenBLAS fallback
   keeps the arithmetic identical across the two architectures tested, but a
   different `qmc` implementation could change the realisation.  The sample
   hash makes such a change visible instead of silent.

## 6. Usage

```python
from tensorphysics.uq import (
    Marginal, SampleSpec, run_uq_study, holtrop_type_benchmark,
)

spec = SampleSpec((
    Marginal.uniform("L", 179.0, 181.0),
    Marginal.uniform("Cb", 0.615, 0.625),
    Marginal.normal("Vs", 14.0, 0.05),
))

study = run_uq_study(my_solver, spec, n_samples=512, degree=4,
                     n_base_mc=1 << 15, bootstrap_mc=100,
                     threshold=0.15, shards=4)          # shards → cluster fan-out

study.status                     # "PASS" | "FAIL" | "WITHHELD"
study.verdict.reason_codes       # ("UQ-W03-LOO", ...)
study.pce_sobol.first            # S_i
study.pce_sobol.total            # S_T,i
study.mc.first                   # independent Saltelli/Jansen check
study.budget.relative_width      # U / |mean|
study.as_dict()                  # full JSON evidence (hashes, numbers, timings)
```

## 7. Files

| path | role |
|---|---|
| `src/tensorphysics/uq/sample_spec.py` | marginals, copula, coordinate maps |
| `src/tensorphysics/uq/sampler.py` | Scrambled Sobol'/LHS/MC, sample hash |
| `src/tensorphysics/uq/adapter.py` | black-box adapter, shard fan-out, hash merge |
| `src/tensorphysics/uq/pce.py` | orthonormal bases, sparse PC, PRESS, bootstrap |
| `src/tensorphysics/uq/sobol_mc.py` | Saltelli/Jansen estimators (1st/2nd/total) |
| `src/tensorphysics/uq/verdict.py` | budget, cross-check, three-state gate, codes |
| `src/tensorphysics/uq/workflow.py` | `run_uq_study` end-to-end chain |
| `src/tensorphysics/uq/benchmarks.py` | Ishigami, Sobol' G, Holtrop-1984 benchmark |
| `tests/test_uq_*.py` | 76 tests (anchors, hashes, shards, gate branches) |
| `examples/uq_end_to_end.py` | UQ → ASME V&V 20 chain on `R_T` |
| `scripts/run_uq_acceptance.py` | A1–A10 acceptance run + figures + JSON |
| `scripts/uq_shard_worker.py` | cluster shard worker (evaluate / merge) |
## 8. IN145-N08 增量（2026-11-09）

节点 IN145-N08 把上面的管线做成**出证前验收门**，并补上四项此前缺失的可信度证据
（详见 [`IN145_N08_UQ_PRE_EVIDENCE_GATE.md`](IN145_N08_UQ_PRE_EVIDENCE_GATE.md)，
工件 `artifacts/in145_n08/acceptance.json`）：

| 新增部件 | 作用 |
|---|---|
| `uq.refinement_study` | 三轴收敛性研究（固定截断 / 固定 N 升阶 / 自适应选择）+ 独立 holdout 误差；给出 `loo_optimism = holdout/eps_LOO` |
| `uq.coverage_study` | 95% bootstrap 区间覆盖率标定（Wilson 区间 + 精确二项检验），零指数用"分辨到零"判据 |
| `uq.injection_suite` | 14 例反证矩阵：任何注入都不得产出 PASS，且必须报出对应原因码 |
| `uq.engineering_error_gate` / `uq.pre_evidence` | digest 寻址的事前证据记录；claim 无证据即 WITHHELD；UQ 半宽作为 ASME `U_input` 链入 `vv.validation` |

**已修复缺陷**：`uq.saltelli_indices` 的一阶（Saltelli 2010）与闭式二阶估计量此前
不具平移不变性——乘子使用原始输出值，输出偏置 c 以 `c·O(σ/√N)` 泄漏。修复前，
Ishigami（`n_base=2^14`，闭式 `S13=0.2437`）在偏置 1e4 下给出 `S13=12.38`；
修复后三种偏置的估计逐位一致。船体阻力 O(1e3) kN 正落在该区间。
回归测试：`tests/test_uq_shift_invariance.py`。
