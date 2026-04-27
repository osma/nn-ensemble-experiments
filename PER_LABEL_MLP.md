# Experiments: `torch_per_label_mlp` Variants

This document tracks small, controlled variations of `torch_per_label_mlp` for benchmarking.
Each variant changes **one aspect** of the model (usually stage 2) while keeping:

- The same data loading, preprocessing, and metrics (NDCG@k, F1@k)
- The same early-stopping policy (train subset NDCG@1000; patience/min-epochs unchanged)
- Roughly the same model size (no dramatic increases in parameters)

The overall goal is to find a `torch_per_label_mlp_*` variant that improves the **overall scoreboard**
criterion: **average of all three test metrics**, averaged across the **three datasets**.

---

## Baseline: `torch_per_label_mlp`

Baseline model intent (as implemented in `benchmarks/torch_per_label_mlp.py`):

Two-stage model:

- **Stage 1**: `torch_per_label` (per-label linear ensemble in logit space) trained with its usual
  early stopping. Produces base logits `base_logits[b, l]`.
- **Stage 2**: Train a residual MLP on *active labels only* using features:
  - Channels 0..2: log1p-preprocessed base predictors `x[b, m, l]`
  - Channel 3: stage-1 logits `base_logits[b, l]`

Stage-2 produces a per-label adjustment signal `delta_active[b, l_active]` which is:
- tanh-bounded (`DELTA_CLAMP`)
- centered per sample over active labels (mean subtraction)
- scaled by a bounded learnable scalar gate `gate ∈ (0, DELTA_GATE_MAX)`

Baseline application form (for active labels):
- `out_active = base_active * (1 + gate * delta_active)`

(Non-active labels remain exactly at the stage-1 logits.)

---

## Variant 1: `torch_per_label_mlp_additive_delta`

**What changes:** apply the residual as an **additive** logit-space delta instead of multiplicative reweighting.

- Baseline (multiplicative):
  - `out_active = base_active * (1 + gate * delta_active)`
- Variant (additive):
  - `out_active = base_active + gate * delta_active`

**Hypothesis:** multiplicative updates can be unstable when `base_active` has large magnitude and can be
ineffective when `base_active ≈ 0`. An additive residual is the standard residual form and may yield
more predictable optimization and better generalization.

### Implementation notes
- Implemented in: `benchmarks/torch_per_label_mlp_additive_delta.py`
- Keep the same `delta_active` computation (tanh bound + centering) and the same scalar gate.
- Only change the final combination rule for active labels.

### Results

Run:
- `./regenerate_scoreboard.sh --models torch_per_label_mlp_additive_delta`
- Device: CPU
- Early-stop criterion (stage 2): train subset NDCG@1000

Best epoch selected by early stopping (stage 2):
- `yso-fi`: epoch **1**
  - Test: NDCG@10 **0.697128**, NDCG@1000 **0.805955**, F1@5 **0.534772**
- `yso-en`: epoch **2**
  - Test: NDCG@10 **0.650038**, NDCG@1000 **0.762893**, F1@5 **0.466016**
- `koko`: epoch **1**
  - Test: NDCG@10 **0.358676**, NDCG@1000 **0.460274**, F1@5 **0.264883**

Comparison vs baseline `torch_per_label_mlp` (from current committed `SCOREBOARD.md`):

- `yso-fi`:
  - baseline MLP: NDCG@10 0.697123, NDCG@1000 0.806150, F1@5 0.534772
  - additive:     NDCG@10 0.697128, NDCG@1000 0.805955, F1@5 0.534772
  - Δ (additive - baseline): +0.000005, -0.000195, +0.000000

- `yso-en`:
  - baseline MLP: NDCG@10 0.650776, NDCG@1000 0.764912, F1@5 0.467200
  - additive:     NDCG@10 0.650038, NDCG@1000 0.762893, F1@5 0.466016
  - Δ (additive - baseline): -0.000738, -0.002019, -0.001184

- `koko`:
  - baseline MLP: NDCG@10 0.361286, NDCG@1000 0.474052, F1@5 0.266288
  - additive:     NDCG@10 0.358676, NDCG@1000 0.460274, F1@5 0.264883
  - Δ (additive - baseline): -0.002610, -0.013778, -0.001405

Macro summary:
- Essentially a **wash** on `yso-fi` (within ~2e-4 for NDCG@1000, identical F1@5).
- Clear **regression** on `yso-en` and especially `koko`, dominated by the drop in NDCG@1000 on `koko`.

### Analysis

1. The additive update did *not* deliver the intended stability/generalization benefit in these runs.
   On `koko`, the drop in NDCG@1000 is large relative to typical per-run jitter seen in the scoreboard,
   suggesting the additive residual is harming long-tail ranking quality.

2. Based on the debug stats from the run:
   - The gate remains near its initialization (~0.02) and grows slowly, so the variant is still in a
     “small residual” regime.
   - Despite the small gate, the residual head’s raw delta distribution grows quickly on `yso-en`/`koko`
     (std increases orders of magnitude by epoch 2–4), indicating the MLP can still learn fairly
     aggressive (bounded) deltas even when the gate is small.

3. Likely explanation for the regression:
   - In the baseline multiplicative form, the residual scales with `base_active` magnitude, so it
     naturally has less effect on near-zero logits and behaves more like a reweighting.
   - In the additive form, the model can introduce a delta that is *not* tied to `base_active`, which may
     make it easier to perturb low-confidence labels and degrade NDCG@1000 (where many more labels matter).

4. Next steps (controlled):
   - If we revisit additive deltas, try smaller `DELTA_CLAMP` (e.g. 0.2) or a lower `DELTA_GATE_MAX`
     to constrain the additive pathway more strongly.
   - Alternatively, keep additive but reintroduce a dependence on base magnitude (e.g. add `base_active`
     as an explicit multiplicative feature to the delta, or scale `delta_active` by `tanh(base_active)`).

Verdict:
- Keep as a documented variant, but **not a candidate “best”**: it does not outperform the baseline MLP
  and appears harmful on `koko` (NDCG@1000).

---

## Variant 2: `torch_per_label_mlp_gate_per_label`

**What changes:** replace the **single scalar gate** with a **per-active-label gate vector**.

- Baseline:
  - `gate = gate_max * sigmoid(raw_gate)`  (scalar)
- Variant:
  - `gate[l] = gate_max * sigmoid(raw_gate[l])`  for `l` in active labels

Apply elementwise, e.g.:
- multiplicative form: `out_active = base_active * (1 + gate * delta_active)`
- additive form (if combined in a later experiment): `out_active = base_active + gate * delta_active`

(For controlled comparison in this variant, keep the baseline multiplicative form.)

**Hypothesis:** only a subset of labels benefit from cross-label adjustments; for others, stage 2
should remain effectively off. A per-label gate allows selective capacity while keeping changes
small and bounded.

### Implementation notes
- Adds `n_active` parameters (small compared to MLP weights).
- Keep the same gate bounds (`DELTA_GATE_MAX`) and initialization policy, but applied per label.
- Implemented in: `benchmarks/torch_per_label_mlp_gate_per_label.py`

### Results

Run:
- `./regenerate_scoreboard.sh --models torch_per_label_mlp_gate_per_label`
- Device: CPU
- Early-stop criterion (stage 2): train subset NDCG@1000 (patience=2, min_epochs=2)

Best epoch selected by early stopping (stage 2):
- `yso-fi`: epoch **6**
  - Test: NDCG@10 **0.697123**, NDCG@1000 **0.806150**, F1@5 **0.534772**
- `yso-en`: epoch **1**
  - Test: NDCG@10 **0.650776**, NDCG@1000 **0.764912**, F1@5 **0.467200**
- `koko`: epoch **3**
  - Test: NDCG@10 **0.361536**, NDCG@1000 **0.465728**, F1@5 **0.266017**

Comparison vs baseline `torch_per_label_mlp` (from current committed `SCOREBOARD.md`):

- `yso-fi`:
  - baseline MLP: NDCG@10 0.697123, NDCG@1000 0.806150, F1@5 0.534772
  - gate-per-label: NDCG@10 0.697123, NDCG@1000 0.806150, F1@5 0.534772
  - Δ: +0.000000, +0.000000, +0.000000

- `yso-en`:
  - baseline MLP: NDCG@10 0.650776, NDCG@1000 0.764912, F1@5 0.467200
  - gate-per-label: NDCG@10 0.650776, NDCG@1000 0.764912, F1@5 0.467200
  - Δ: +0.000000, +0.000000, +0.000000

- `koko`:
  - baseline MLP: NDCG@10 0.361286, NDCG@1000 0.474052, F1@5 0.266288
  - gate-per-label: NDCG@10 0.361536, NDCG@1000 0.465728, F1@5 0.266017
  - Δ: +0.000250, -0.008324, -0.000271

Macro summary:
- This variant is effectively **identical** to the baseline `torch_per_label_mlp` on `yso-fi` and `yso-en`.
- On `koko`, it is a **regression** driven by a notable drop in **NDCG@1000**.

### Analysis

1. **Per-label gates did not “turn on” meaningfully on yso-*.**
   In the debug output, `gate_vec` stayed extremely close to its initialization (~0.02) for both `yso-fi`
   and `yso-en` across epochs (std on the order of 1e-8 to 1e-7). That implies stage 2 remained almost
   uniformly “on at the same small level” for all active labels, behaving very similarly to the scalar-gate
   baseline.

2. **The residual head learned extremely small deltas on yso-*; net effect remained negligible.**
   For `yso-fi` and `yso-en`, `delta(tanh-bounded, centered)` standard deviations were tiny (1e-4-ish),
   and `gated_delta(mult)` standard deviations were even smaller (1e-6-ish). With the multiplicative
   combination rule, this results in almost no change to base logits/rankings—consistent with the identical
   scoreboard metrics.

3. **On `koko`, the residual “turned on” aggressively and appears to harm long-tail ranking.**
   Unlike yso-*, `koko` quickly exhibited large residual activations and deltas:
   - `delta` std grew from ~0.002 (epoch 1) to ~0.26–0.43 (epoch 3–5), hitting the tanh clamp range.
   - Hidden activations (`h1`) and parameter magnitudes (`fc2.w`) grew rapidly.
   Even though `gate_vec` means remained near ~0.02, it also developed more spread and drift.
   The net result is a measurable drop in NDCG@1000, suggesting the residual perturbed many low-ranked labels
   (exactly what NDCG@1000 is sensitive to).

4. **Why this likely happens: “capacity where it’s least safe.”**
   The per-label gate adds degrees of freedom, but (as implemented) it’s *not conditioned on per-label
   reliability signals* and is not regularized beyond the sigmoid bound. For `koko` (with far fewer active
   labels and different base-model sparsity patterns), the residual head can learn broad cross-label patterns
   that overfit and degrade the long tail.

Verdict:
- Not a clear improvement over the baseline `torch_per_label_mlp`.
- Safe/no-op on yso-* in this run, but **harmful on `koko` (NDCG@1000 regression)**.
- Keep as a documented variant; if revisited, consider adding either:
  - stronger regularization on `raw_delta_gate` (e.g. L2 toward its init), or
  - a data-dependent gate (see Variant 3), or
  - tighter `DELTA_CLAMP` / lower `DELTA_GATE_MAX` specifically for `koko`.

---

## Variant 3: `torch_per_label_mlp_gate_per_sample`

**What changes:** make the gate **data-dependent** (per sample) instead of a single learned scalar.

Concept (as implemented in `benchmarks/torch_per_label_mlp_gate_per_sample.py`):

- Pool stage-2 features over active labels:
  - `p[b, c] = mean_l feats[b, c, l_active]`
- Compute a per-sample bounded gate via a tiny head:
  - `gate[b] = gate_max * sigmoid(linear(p[b]))`
- Apply multiplicatively on active labels:
  - `out_active = base_active * (1 + gate[b] * delta_active)`

**Hypothesis:** cross-label “rules” may only be reliable for certain samples/documents. A per-sample
gate can reduce harm by turning the residual down when signals are weak or noisy, protecting stage-1
performance.

### Implementation notes
- Implemented in: `benchmarks/torch_per_label_mlp_gate_per_sample.py`
- Gate head is a single `nn.Linear(n_channels=4 -> 1)` with:
  - weights initialized to 0
  - bias initialized so `sigmoid(bias) = DELTA_GATE_INIT / DELTA_GATE_MAX`
- Gate head is trained with **no weight decay**; residual MLP uses AdamW weight decay.
- Keeps baseline safety constraints: tanh-bounded residual (`DELTA_CLAMP`) and per-sample centering of `delta_active`.

### Results

Run:
- `./regenerate_scoreboard.sh --models torch_per_label_mlp_gate_per_sample`
- Device: CPU
- Early-stop criterion (stage 2): train subset NDCG@1000 (patience=2, min_epochs=2)

Best epoch selected by early stopping (stage 2):
- `yso-fi`: epoch **5**
  - Test: NDCG@10 **0.697123**, NDCG@1000 **0.806152**, F1@5 **0.534772**
- `yso-en`: epoch **1**
  - Test: NDCG@10 **0.650776**, NDCG@1000 **0.764916**, F1@5 **0.467200**
- `koko`: epoch **1**
  - Test: NDCG@10 **0.361286**, NDCG@1000 **0.474100**, F1@5 **0.266266**

Comparison vs baseline `torch_per_label_mlp` (from current committed `SCOREBOARD.md`):

- `yso-fi`:
  - baseline MLP: NDCG@10 0.697123, NDCG@1000 0.806150, F1@5 0.534772
  - gate-per-sample: NDCG@10 0.697123, NDCG@1000 0.806152, F1@5 0.534772
  - Δ: +0.000000, +0.000002, +0.000000

- `yso-en`:
  - baseline MLP: NDCG@10 0.650776, NDCG@1000 0.764912, F1@5 0.467200
  - gate-per-sample: NDCG@10 0.650776, NDCG@1000 0.764916, F1@5 0.467200
  - Δ: +0.000000, +0.000004, +0.000000

- `koko`:
  - baseline MLP: NDCG@10 0.361286, NDCG@1000 0.474052, F1@5 0.266288
  - gate-per-sample: NDCG@10 0.361286, NDCG@1000 0.474100, F1@5 0.266266
  - Δ: +0.000000, +0.000048, -0.000022

Macro summary:
- Effectively a **wash** relative to `torch_per_label_mlp` on all datasets in this run.
- Tiny positive movement in NDCG@1000 on all three datasets (on the order of 1e-6 to 1e-4), with a tiny F1@5 decrease on `koko`.
- Overall: this does **not** appear to be a meaningful improvement beyond normal run-to-run/caching precision, but it also does **not** show the `koko` NDCG@1000 regression seen in the gate-per-label variant.

### Analysis

1. The gate is *effectively constant* (not meaningfully data-dependent) in this run.
   The epoch logs show:
   - `yso-fi`: `gate mean ≈ 0.02`, `std ≈ 0` for most epochs (std only reaches ~1e-9 by epoch 6–7).
   - `yso-en`: `gate mean ≈ 0.02`, `std = 0` (to printed precision) throughout.
   - `koko`: `gate mean ≈ 0.02004 → 0.02133` by epoch 3, but still near-constant across the debug subset (tiny std).
   This implies the linear gate head is learning mostly a **global bias shift**, not per-sample variation.

2. Stage-2 updates remain extremely small on yso-* (net behavior ≈ no-op).
   On `yso-fi` and `yso-en`, `gated_delta(mult term)` std stays in the ~1e-6 to 1e-5 range.
   With the multiplicative combination rule, this produces almost no perturbation to rankings—consistent with
   the scoreboard metrics being identical to baseline up to 1e-6.

3. `koko` still shows “residual wants to turn on”, but early stopping protects quality.
   For `koko`, by epoch 2–3:
   - `delta` std jumps from ~0.002 to ~0.05 then ~0.14 (hitting/approaching the tanh clamp),
   - hidden activations and fc2 weights grow quickly,
   - and test NDCG@1000 drops (epoch 2: 0.468868; epoch 3: 0.463755) compared to epoch 1.
   Early stopping (best epoch=1) effectively selects the **safe near-no-op snapshot**, preventing the degradation.

4. Why the “per-sample gate” hypothesis wasn’t realized here:
   - The pooling `mean_l feats[b, c, l]` can be dominated by many near-zero features (especially with sparse-ish inputs),
     making it hard for a linear head to produce a wide dynamic range.
   - Weight decay is disabled for the gate head, but the residual head can still create large deltas; the easiest way for
     optimization to remain stable is to keep the gate near its initialization.
   - With `DELTA_GATE_MAX=0.2` and init at `0.02`, the model is biased toward “small residual”; it may require a stronger
     learning signal or different pooling (e.g. mean of |feats|, max, or a learned pooling) to become meaningfully conditional.

Verdict:
- Safe and essentially equivalent to baseline `torch_per_label_mlp` in these runs.
- Not a clear candidate for “best”, but a useful confirmation that making the gate data-dependent (with this simple linear pooling head)
  does **not** automatically improve results—and that `koko` remains sensitive to stage-2 capacity after the first epoch.

---

## Variant 4: `torch_per_label_mlp_layernorm_feats`

**What changes:** add **normalization** to stage-2 features before the MLP.

Selected option (implemented):
- `LayerNorm` over the **flattened feature vector per sample** (shape `(C * L_active,)`),
  applied right after `Flatten(feats)` and before `fc1`.

**Hypothesis:** stage-2 inputs mix log1p base scores and stage-1 logits, which can have different
scales and distributions. Normalizing features can stabilize optimization, reduce sensitivity to
learning rate, and mitigate “early peak then degrade” behavior.

### Implementation notes
- Implemented in: `benchmarks/torch_per_label_mlp_layernorm_feats.py`
- `LayerNorm(..., elementwise_affine=True)` (affine parameters enabled).
- If `n_active == 0`, the residual head uses `nn.Identity()` as a safe no-op in place of `LayerNorm`
  (since LayerNorm requires a positive normalized shape).
- No other changes vs the baseline `torch_per_label_mlp`.

### Results

Run:
- `./regenerate_scoreboard.sh --models torch_per_label_mlp_layernorm_feats`
- Device: CPU
- Early-stop criterion (stage 2): train subset NDCG@1000 (patience=2, min_epochs=2)

Best epoch selected by early stopping (stage 2):
- `yso-fi`: epoch **1**
  - Test: NDCG@10 **0.697001**, NDCG@1000 **0.805845**, F1@5 **0.534772**
- `yso-en`: epoch **1**
  - Test: NDCG@10 **0.650211**, NDCG@1000 **0.763108**, F1@5 **0.466447**
- `koko`: epoch **1**
  - Test: NDCG@10 **0.361464**, NDCG@1000 **0.461302**, F1@5 **0.266139**

Comparison vs baseline `torch_per_label_mlp` (from current committed `SCOREBOARD.md`):

- `yso-fi`:
  - baseline MLP: NDCG@10 0.697123, NDCG@1000 0.806150, F1@5 0.534772
  - layernorm-feats: NDCG@10 0.697001, NDCG@1000 0.805845, F1@5 0.534772
  - Δ (layernorm - baseline): -0.000122, -0.000305, +0.000000

- `yso-en`:
  - baseline MLP: NDCG@10 0.650776, NDCG@1000 0.764912, F1@5 0.467200
  - layernorm-feats: NDCG@10 0.650211, NDCG@1000 0.763108, F1@5 0.466447
  - Δ (layernorm - baseline): -0.000565, -0.001804, -0.000753

- `koko`:
  - baseline MLP: NDCG@10 0.361286, NDCG@1000 0.474052, F1@5 0.266288
  - layernorm-feats: NDCG@10 0.361464, NDCG@1000 0.461302, F1@5 0.266139
  - Δ (layernorm - baseline): +0.000178, -0.012750, -0.000149

Macro summary:
- Overall a **regression** vs baseline, dominated by **NDCG@1000 drops** on `koko` and `yso-en`.
- `yso-fi` is essentially unchanged/slightly worse on NDCG, identical F1@5.

### Analysis

1. The intended “stabilize stage-2 optimization” effect did not translate into better test ranking.
   In all three datasets, the best snapshot by early stopping was **epoch 1** of stage 2, and metrics
   were already slightly below the baseline MLP for `yso-*`, with a large regression in `koko` NDCG@1000.

2. LayerNorm appears to make it easier for the residual to produce large deltas very quickly.
   The debug logs show a pattern consistent with “fast ramp-up” of the residual head:
   - `yso-fi`: delta std grows from ~0.003 (epoch 1) to ~0.036 (epoch 2) to ~0.116 (epoch 3).
   - `yso-en`: delta std grows from ~0.009 (epoch 1) to ~0.091 (epoch 2) to ~0.185 (epoch 3).
   - `koko`: delta std is already large at epoch 1 (~0.217) and remains high (~0.23–0.24).
   Even with a small gate (~0.02), these deltas are large enough to perturb many ranks once multiplied by
   `base_active` in the multiplicative form, which is consistent with degraded NDCG@1000.

3. The regression is most pronounced on `koko` long-tail ranking (NDCG@1000).
   `koko` drops from **0.474052 → 0.461302** at the stage-2 best epoch.
   This matches a repeated theme in other variants: stage-2 capacity tends to hurt the long tail on `koko`
   unless it remains a near-no-op.

4. Why LayerNorm-over-flattened-features may be counterproductive here:
   - It removes per-feature scale information across the entire `(C * L_active)` vector. With many active
     labels, this can amplify small but widespread signals and reduce the “natural” dominance of strong
     base logits for a subset of labels.
   - It also couples all active labels through a shared normalization statistic per sample, potentially
     introducing cross-label interactions *before* the MLP itself—exactly where we are trying to be cautious.
   - Given inputs include stage-1 logits plus log1p predictor features, a global LN may encourage the MLP
     to use relative deviations across the whole active set, which can overfit and degrade NDCG@1000.

5. Practical takeaway / next iteration:
   - Keep this as a documented variant, but it is **not a “best candidate”** given the measured regressions.
   - If normalization is revisited, prefer more local/structured options that don’t mix all labels:
     - normalize per-channel across labels (e.g. `LayerNorm` over the `L_active` dimension per channel),
       or
     - standardize only the stage-1 logit channel, or
     - reduce `DELTA_CLAMP` / `DELTA_GATE_MAX` specifically for `koko`.

Verdict:
- **Not recommended** as a replacement for `torch_per_label_mlp` due to consistent regressions in
  NDCG@1000 (especially `koko`).

---

## Variant 5: `torch_per_label_mlp_rank_bottleneck`

**What changes:** introduce a **low-rank bottleneck** in the stage-2 output projection to regularize
cross-label adjustment structure.

Replace:
- `fc2: hidden_dim -> n_active`

with:
- `fc2a: hidden_dim -> r`
- `fc2b: r -> n_active`

where `r` is small (e.g. 32).

**Hypothesis:** the residual adjustment across labels may lie in a low-dimensional subspace. A rank
bottleneck can reduce overfitting, encourage shared structure, and improve generalization while
keeping the model size in the same ballpark.

### Implementation notes
- Implemented in: `benchmarks/torch_per_label_mlp_rank_bottleneck.py`
- This run used `RANK_BOTTLENECK=32`.
- Initialization policy matches the baseline safety goal: `fc2b` is zero-initialized so the stage-2
  residual starts as (near) a no-op, then can “turn on” through training.

### Results

Run:
- `./regenerate_scoreboard.sh --models torch_per_label_mlp_rank_bottleneck`
- Device: CPU
- Early-stop criterion (stage 2): train subset NDCG@1000 (patience=2, min_epochs=2)

Best epoch selected by early stopping (stage 2):
- `yso-fi`: epoch **4**
  - Test: NDCG@10 **0.697123**, NDCG@1000 **0.806153**, F1@5 **0.534772**
- `yso-en`: epoch **4**
  - Test: NDCG@10 **0.650776**, NDCG@1000 **0.764786**, F1@5 **0.467200**
- `koko`: epoch **4**
  - Test: NDCG@10 **0.360948**, NDCG@1000 **0.470921**, F1@5 **0.266199**

Comparison vs baseline `torch_per_label_mlp` (from current committed `SCOREBOARD.md`):

- `yso-fi`:
  - baseline MLP: NDCG@10 0.697123, NDCG@1000 0.806150, F1@5 0.534772
  - rank-bottleneck: NDCG@10 0.697123, NDCG@1000 0.806153, F1@5 0.534772
  - Δ (rank - baseline): +0.000000, +0.000003, +0.000000

- `yso-en`:
  - baseline MLP: NDCG@10 0.650776, NDCG@1000 0.764912, F1@5 0.467200
  - rank-bottleneck: NDCG@10 0.650776, NDCG@1000 0.764786, F1@5 0.467200
  - Δ (rank - baseline): +0.000000, -0.000126, +0.000000

- `koko`:
  - baseline MLP: NDCG@10 0.361286, NDCG@1000 0.474052, F1@5 0.266288
  - rank-bottleneck: NDCG@10 0.360948, NDCG@1000 0.470921, F1@5 0.266199
  - Δ (rank - baseline): -0.000338, -0.003131, -0.000089

Macro summary:
- Essentially a **wash** on `yso-fi` (slight +3e-6 in NDCG@1000).
- Slight **regression** on `yso-en` (−1.26e-4 in NDCG@1000; other metrics unchanged to 6 decimals).
- Clearer **regression** on `koko`, dominated by the drop in **NDCG@1000** (−0.003131).

### Analysis

1. **The bottleneck did not produce a meaningful generalization gain.**
   The measured changes on `yso-*` are at or below typical scoreboard-level jitter (1e-4-ish), and
   `koko` regresses in the long tail (NDCG@1000).

2. **Stage 2 remains mostly “near-no-op” on yso-* (which is consistent with identical metrics).**
   Debug stats show `delta(tanh-bounded, centered)` std stays very small on yso-fi/yso-en:
   - `yso-fi`: std grows from ~4e-5 → ~2.6e-4 over epochs 1→6
   - `yso-en`: std grows from ~8e-5 → ~1.1e-3 over epochs 1→6
   With `gate ≈ 0.02`, the effective multiplicative term `gate * delta` is tiny (std in the ~1e-6 → 2e-5 range),
   so rankings barely change. This also suggests the rank bottleneck is not being “used” strongly on yso-*.

3. **On `koko`, stage 2 “turns on” and still harms NDCG@1000 despite the low-rank constraint.**
   The bottleneck reduces degrees of freedom, but the residual can still become large:
   - `koko` delta std jumps from ~3.6e-4 (epoch 1) → ~0.0047 (epoch 3) → ~0.0356 (epoch 4),
     and continues increasing afterward.
   - Correspondingly, `gate * delta` grows to std ~7.4e-4 by epoch 4.
   Even with a rank-32 output subspace, this is enough to perturb many low-ranked labels and reduce
   NDCG@1000.

4. **Early stopping picked epoch 4 for all datasets, but that doesn’t imply “stage 2 helps.”**
   On yso-*, the selected snapshot is effectively equivalent to base performance.
   On `koko`, epoch 4 is already worse than the baseline MLP row, indicating that (for this dataset)
   the rank bottleneck alone is not sufficient to keep stage 2 safely beneficial.

5. **Practical takeaway / next iteration ideas (controlled):**
   - If we keep exploring the bottleneck idea, consider pairing it with *stronger safety constraints*
     specifically for `koko`, e.g. lower `DELTA_CLAMP` (0.2) and/or lower `DELTA_GATE_MAX` (0.1),
     or add a small explicit penalty on `fc2b`/delta magnitude.
   - Alternatively, keep the bottleneck but switch to the **per-sample gate** (Variant 3) to make it
     easier for early stopping to select a “residual off” snapshot on `koko`.

Verdict:
- Keep as a documented variant.
- Not a clear improvement over baseline `torch_per_label_mlp`; shows a small-to-moderate `koko`
  long-tail regression (NDCG@1000).

---

## Variant 6: `torch_per_label_mlp_remove_centering`

**What changes:** remove the per-sample centering constraint on `delta_active`.

- Baseline:
  - `delta_active = delta_active - mean(delta_active over active labels)`
- Variant:
  - no centering

**Hypothesis:** centering forces the residual to be “redistributive” among active labels only. This
may be too restrictive: in some cases, a uniform shift (or broad boost/suppression) of active logits
could help separate active vs inactive labels. Removing centering tests whether the constraint is
helpful regularization or an unnecessary limitation.

### Implementation notes
- Keep tanh bounding and gating identical.
- Only remove the mean-subtraction step.
- Combination rule remains the baseline multiplicative form:
  - `out_active = base_active * (1 + gate * delta_active)`

### Results

Run:
- `./regenerate_scoreboard.sh --models torch_per_label_mlp_remove_centering`
- Device: CPU
- Early-stop criterion (stage 2): train subset NDCG@1000 (patience=2, min_epochs=2)

Best epoch selected by early stopping (stage 2):
- `yso-fi`: epoch **1**
  - Test: NDCG@10 **0.697123**, NDCG@1000 **0.793181**, F1@5 **0.534772**
- `yso-en`: epoch **1**
  - Test: NDCG@10 **0.650776**, NDCG@1000 **0.737115**, F1@5 **0.467200**
- `koko`: epoch **1**
  - Test: NDCG@10 **0.361142**, NDCG@1000 **0.410901**, F1@5 **0.266119**

Comparison vs baseline `torch_per_label_mlp` (from current committed `SCOREBOARD.md`):

- `yso-fi`:
  - baseline MLP: NDCG@10 0.697123, NDCG@1000 0.806150, F1@5 0.534772
  - remove-centering: NDCG@10 0.697123, NDCG@1000 0.793181, F1@5 0.534772
  - Δ (remove-centering - baseline): +0.000000, **-0.012969**, +0.000000

- `yso-en`:
  - baseline MLP: NDCG@10 0.650776, NDCG@1000 0.764912, F1@5 0.467200
  - remove-centering: NDCG@10 0.650776, NDCG@1000 0.737115, F1@5 0.467200
  - Δ (remove-centering - baseline): +0.000000, **-0.027797**, +0.000000

- `koko`:
  - baseline MLP: NDCG@10 0.361286, NDCG@1000 0.474052, F1@5 0.266288
  - remove-centering: NDCG@10 0.361142, NDCG@1000 0.410901, F1@5 0.266119
  - Δ (remove-centering - baseline): -0.000144, **-0.063151**, -0.000169

Macro summary:
- **Severe regression** in long-tail ranking quality (**NDCG@1000**) on *all* datasets.
- NDCG@10 and F1@5 remain essentially unchanged (to 6 decimals), but that is misleading: the variant
  catastrophically harms the long tail without materially affecting the very top of the ranking.

### Analysis

1. The no-centering variant collapsed to a degenerate residual: `delta_active ≈ +DELTA_CLAMP` everywhere.
   In the epoch logs on all datasets, the debug stats show:
   - `delta(tanh-bounded, NOT centered): mean=0.5 std≈0 min=0.5 max=0.5`
   With `DELTA_CLAMP=0.5`, this means the tanh-bounded residual saturates at the positive clamp for
   (almost) every (sample, active-label) pair. In other words, stage 2 did *not* learn meaningful
   per-label redistributions; it learned a near-constant positive boost signal.

2. In the multiplicative combination rule, a constant positive delta is equivalent to scaling active logits upward.
   Since:
   - `out_active = base_active * (1 + gate * delta_active)`
   and here `delta_active ≈ 0.5`, `gate ≈ 0.02`, we get:
   - `1 + gate * delta_active ≈ 1 + 0.01 = 1.01`
   So stage 2 effectively applies a uniform ~+1% scaling to *all* active-label logits.

3. Uniform scaling of logits should preserve ranking order within the active set, so why does NDCG@1000 collapse?
   The key is that `active_idx` includes labels that are active in **either** train truth **or any base prediction**.
   For yso-fi/yso-en this is ~80–86% of all labels; for koko it is ~32% (~19k labels). A uniform rescaling applied
   to a huge portion of the label space changes the relative ordering between:
   - labels that were previously just below the top-k cutoff vs just above it,
   - active vs inactive-label blocks (inactive labels remain unscaled at base logits),
   - and, critically for NDCG@1000, the tail of the ranking where small perturbations affect many positions.
   Even when the transform is monotone within the active block, it can reshuffle many cross-block comparisons.

4. The empirical outcome matches that expectation: NDCG@10 and F1@5 barely move, but NDCG@1000 drops sharply.
   NDCG@10 depends on the top of the ranking and is relatively robust to small global scaling effects (especially if
   the top-ranked labels are already mostly within the active block). NDCG@1000 is far more sensitive because it
   evaluates a much larger portion of the label ranking, where the cross-block perturbation affects many labels.

5. Why removing centering likely causes this failure mode:
   - With centering, the residual is forced to have zero mean per sample, preventing an easy “push everything up”
     solution; the MLP must learn redistributive patterns to improve the early-stop metric.
   - Without centering, the optimization can exploit the tanh clamp + multiplicative form by driving the residual
     to a constant sign (here positive), which is a low-effort direction in parameter space and can increase/decrease
     the BCE loss without learning any cross-label structure.
   - In this run, it appears the model rapidly found that saturated constant solution and then effectively stopped
     changing metrics (train subset NDCG@1000 stayed flat across epochs).

Verdict:
- This variant is **not viable** in its current form; removing centering produces a degenerate residual and causes a
  large NDCG@1000 regression across all datasets.
- If revisited, it would need additional constraints to prevent constant shifts, such as:
  - reintroducing centering, or
  - adding an explicit penalty on `mean(delta_active)` per sample, or
  - switching to an additive residual form *plus* a constraint/regularizer on the mean delta, or
  - redefining `active_idx` more conservatively (e.g. truth-only) to reduce cross-block effects.
