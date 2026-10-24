# Performance work: NCF scoring and the C++ matching path

Measured on this machine (Intel Core Ultra 9 275HX, 24 threads, torch 2.3.1+cpu,
Windows 11). Every number below is wall clock from the real run loop, not a
microbenchmark, unless the line says otherwise.

## 1. Profile (N=5,000, 20 epochs)

`cProfile` under multi-threaded torch attributes most of the wall clock to
`threading.wait`, so the per-stage split was measured directly instead
(`data/perf/stage_timer.py`, wrapping each call site with `perf_counter`).

Baseline, before any change — `run()` total **138.2 s**:

| stage | total | per call | share |
|---|---|---|---|
| `top_k_recommendations` | 81.31 s | 2032.7 ms | 58.8% |
| `train_epoch` | 27.72 s | 1385.8 ms | 20.1% |
| `count_blocking_pairs` | 21.05 s | 1052.3 ms | 15.2% |
| `mean_admissible_utility` | 4.52 s | 215.2 ms | 3.3% |
| `deferred_acceptance` | 1.10 s | 52.2 ms | 0.8% |
| `build_full_info_pref_lists` | 0.94 s | — | 0.7% |
| `generate_swipes` | 0.49 s | 24.4 ms | 0.4% |

**Top-5 bottlenecks by cumulative time:** NCF scoring (58.8%), NCF training
(20.1%), `count_blocking_pairs` (15.2%), `mean_admissible_utility` (3.3%),
`deferred_acceptance` (0.8%).

`_hypergamic` did *not* appear as a single opaque frame. The two C++ functions
that dominate are separate bound symbols, so the split was visible without
instrumenting the extension.

## 2. What was already done

The three Python optimizations the brief anticipated were **already in the
code** on `main` before this session:

- scoring is batched (one forward per block of rows, not per user);
- `torch.topk` is used, not a sort;
- the model and Adam optimizer are built once in `Simulation.__init__` and
  reused across epochs.

So the remaining cost was elsewhere: inside the forward pass itself, and in two
serial O(N²) C++ loops.

## 3. Changes

### 3.1 Scoring fast path (`src/python/ncf.py`)

Two exact algebraic rewrites of the scoring forward, in `NCFModel.score_block`:

1. **First layer hoisted.** `W1 @ [a; b] + c = a @ W1[:, :D]ᵀ + b @ W1[:, D:]ᵀ + c`
   because the first MLP layer is bilinear. The 128×128 per-pair matmul (16,384
   MACs, the largest single term) becomes two per-entity projections built once
   per call. Per pair: ~10.3k MACs instead of ~26.7k.
2. **GMF folded through the output layer.**
   `Σ_d Wout[d]·gu[d]·gi[d] = Σ_d (Wout[d]·gu[d])·gi[d]`, so the (U, C, D)
   elementwise tensor plus its linear becomes one BLAS matmul over the candidate
   axis.

At the loop's shape (6 rows × 5,000 candidates): **3.3 ms → 1.4 ms per pass
(2.4×)**. Output agrees with `forward` to ≤1.5e-8 absolute on a logit scale of
order 0.1, and the top-k is identical.

### 3.2 `mean_admissible_utility` (C++, `bindings.cpp`)

Was a serial O(N²) loop — 200 ms per call, called every epoch. Parallelised
over rows. Each row accumulates in the same ascending candidate order either
way, so the result is bitwise identical, not merely close. **215 ms → 14 ms
(15×)**.

### 3.3 `count_blocking_pairs` (C++, `matching.cpp`)

Was a serial O(N²) scan. Three changes:

- parallelised over acceptors with an integer `reduction(+ : blocking)`, which
  is exact regardless of summation order;
- hoisted `rank_of(p, partner_prop[p])` out of the acceptor loop — it depends on
  the proposer only, and was being recomputed once per acceptor who ranked him
  above her partner;
- replaced the `rank_of(ai, p)` binary search with the loop index, which *is*
  that rank by construction (`RankIndex` stores each list's original positions).

**1052 ms → 641 ms (1.6×)**.

### 3.4 Also measured, and rejected

- **Adam `foreach=True`**: honestly A/B'd; within noise (see §6).
- **Larger training batch** (3.6× faster at 32768): changes the SGD trajectory.
  Measured at 8 epochs, it moves **200/200** top-1 recommendation lists. It is a
  model parameter, not a performance knob, so it was left alone.
- **`SparseAdam`**: 0.95× — slower, and a different optimizer.
- **`torch.compile`**: unavailable (`Dynamo is not supported on Python 3.12+`
  for this build).
- **Thread count**: swept 4/8/12/16/24. 24 is the best or tied throughout; the
  pass is memory-bound on activations, not thread-starved.
- **SIMD / cache blocking**: not attempted, per the brief.

## 4. Re-profile (N=5,000, 20 epochs)

`run()` total **138.2 s → 83.2 s**, a **1.66× speedup** on the whole loop.

An alternating A/B of the scoring function alone, at the same workload, gives
**1.39×** on the whole loop (43.9 s → 31.7 s median over 8 epochs, legacy
function reconstructed from git HEAD and patched onto the trainer so both arms
run identical code elsewhere). At N=10,000 the same A/B gives **1.91×** — the
scoring pass is a larger share of the loop there.

| stage | before | after | |
|---|---|---|---|
| `train_epoch` | 27.7 s | 33.7 s | untouched code; see note |
| `top_k_recommendations` | 81.3 s | 33.1 s | 2.5× |
| `count_blocking_pairs` | 21.1 s | 12.8 s | 1.6× |
| `mean_admissible_utility` | 4.5 s | 0.29 s | 15× |
| `build_full_info_pref_lists` | 0.94 s | 0.93 s | — |

The scoring pass fell from 58.8% to 39.8% of the loop and training rose to 40.6%
— the work simply moved. `train_epoch` was not modified by any of these changes
(the training path still calls the original `forward`), so its 27.7 → 33.7 s
appearance is drift between two separately-timed runs on a machine that was also
running benchmarks, not a regression. The staged totals are what to trust.

### Equivalence

`data/perf/compare_runs.py` on before/after CSVs at N=5,000, 20 epochs: all
**30 columns bitwise identical**, including `n_blocking` and `bpr`. Also
bit-identical run-to-run after the changes (determinism preserved), which is what
the OpenMP loops needed to be checked for.

Tests: **5/5 C++ suites, 251/251 Python tests** pass. Two new tests pin
`score_block` against `forward` across four `mlp_layers` shapes and check that
scoring does not alias the projection tensors.

## 5. Scale test (N=10,000, 50 epochs)

Target: under 3 minutes. Measured **810.5 s = 13.5 minutes**, clean (nothing else
running). Over target by **4.5×**.

There is no clean pre-optimization 50-epoch baseline: the only "before" run at
this size (826 s) was taken while other benchmarks held the same 24 threads, so
comparing it to 810.5 s would read as a null result. The relative effect is
measured instead, by an alternating A/B inside one process with the legacy
scoring function reconstructed from git HEAD:

| N=10,000, 6 epochs, alternating | median |
|---|---|
| legacy scoring | 83.6 s |
| optimized scoring | 43.7 s |
| **speedup** | **1.91×** |

At N=10,000, 20 epochs, the stage split is `top_k_recommendations` 68.7%,
`train_epoch` 29.2%, everything else under 1%. Above N=5,000 there is no
full-information benchmark, so `count_blocking_pairs` is not called and `bpr` is
NaN — which is why that stage is absent here and present at N=5,000.

**Per-epoch cost grows with run length on this machine.** 6 epochs run at 7.3
s/epoch; 20 epochs at 12.5 s/epoch; 50 epochs at 16.2 s/epoch. The scoring call
itself goes from 3.6 s to 4.3 s over the same span, with identical shapes. That is
sustained-load clock behaviour on a 24-thread laptop part, not an algorithmic
effect, and it means any long-run wall clock measured on this box is
pessimistic relative to a short benchmark of the same code.

## 6. The N=100,000 target

Measured at the real shapes (`data/perf/target_shape.py`), not extrapolated:

| N | pairs/pass | throughput | scoring per epoch |
|---|---|---|---|
| 5,000 | 15,000 | 10.25 Mpairs/s | 1.2 s |
| 10,000 | 30,000 | 9.31 Mpairs/s | 5.4 s |
| 20,000 | 60,000 | 7.84 Mpairs/s | 25.5 s |
| 50,000 | 150,000 | 7.24 Mpairs/s | 172.5 s |
| **100,000** | 250,000 | **6.47 Mpairs/s** | **772.5 s** |

At N=100,000 an epoch is 5 Gpairs of scoring. The target is 600 s for 100
epochs, i.e. **6 s/epoch ≈ 0.8 Gpairs/s** — about **120× this machine's measured
throughput**. Scoring alone would take **21.5 hours** for 100 epochs, before
training is counted; adding training (~320 s/epoch under a crude scaling) puts
the full run near **30 hours**.

Getting there would need every pair scored in ~0.2 ns, i.e. roughly 50× this
CPU's batch-5 BLAS throughput. That is not reachable by the kind of change the
brief asks for (allocation, boundary crossings, batching); it needs float16 or a
GPU, and even then the 50× gap is not obviously closable.

**The structural option, stated as a modeling decision rather than a fix:**
cap the candidate pool per agent (score a sampled shortlist instead of the whole
opposite side). That makes scoring linear in N instead of quadratic. It also
changes the market being studied — every `bpr`, `algorithmic_loss` and exposure
number would be about a different, thinner market — so it is not something to
be slipped in as an optimization.

## 7. Reproduction

```
python data/perf/stage_timer.py data/perf/n5000_e20.yaml   # per-stage wall clock
python data/perf/compare_runs.py before.csv after.csv      # column-by-column diff
python data/perf/target_shape.py                           # throughput at real N
python data/perf/ab_scaling.py 2                           # alternating A/B + scaling
```

Configs for the two workloads are in `data/perf/n5000_e20.yaml` and
`data/perf/n10000_e50.yaml` (copies of `experiments/base.yaml` with `n` and
`n_epochs` changed).
