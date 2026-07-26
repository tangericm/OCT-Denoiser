# Measured Findings

Everything below was measured on the Maestro2/Maestro3 acquisitions in this
project. Claims that were made and later refuted are recorded as such rather
than deleted, because knowing which reasoning failed is worth as much as the
result.

Reproduce with the scripts under `src/octdenoiser/experiments/`.

---

## 1. The supervision defect

The original method feeds two Gaussian spectral sub-bands and regresses them
onto the **full-band** reconstruction. The full band *contains* both sub-bands,
so the target's noise is correlated with the input's and the network is rewarded
for passing noise through. This is bias, not variance.

Measured with lateral structure removed, so speckle is isolated rather than
swamped by shared anatomy:

| Pairing | Speckle correlation |
|---|---|
| Sub-band input vs **its own full-band target** | **+0.138** |
| Sub-band input vs complementary sub-band | +0.003 |

Noise2Noise requires that correlation near zero. The leaking target is ~40×
worse than a clean pairing.

**A related suspicion was void.** "Gaussian windows overlap in the tails" was
listed as a defect. At the tuned σ=0.05, gap=0.60 the measured support overlap
is exactly **0.0000** — the windows are already disjoint.

---

## 2. View constructions compared

Speckle correlation with structure removed; axial PSF relative to full band;
signal mismatch = relative difference in mean depth profile between the paired
views.

| Construction | Speckle corr | Axial PSF | Signal mismatch | Frames available |
|---|---|---|---|---|
| Adjacent positions, 1 step (11.7 µm) | **+0.017** | 1.00× | — | ~7,800 |
| Repeat frames (Maestro2) | +0.008 | 1.00× | ~0 | 200 |
| Contiguous Gaussian sub-bands | +0.003 | **2.43×** | **0.269** | ~8,850 |
| Complementary random masks | **+0.257** | 1.01× | 0.058 | ~8,850 |

**The existing bandgap already decorrelates speckle almost perfectly** (+0.003).
Its costs are elsewhere: the leaking target, a 2.43× axial resolution penalty,
and a 0.269 chromatic signal mismatch with a systematic depth-dependent slope
(the two bands sit at different k, and scattering is wavelength-dependent — a
Noise2Noise bias term, not noise).

### Refuted: complementary random masks

Proposed on the expectation that disjoint k-support would decorrelate speckle
better than overlapping Gaussians. It does the opposite, by **75×**.

The reason is physical: speckle decorrelates because two views sample
*different* spectral regions. Contiguous sub-bands do exactly that; interleaved
masks spread across the *same* range — which is precisely the property that
preserves the PSF. **Decorrelation and resolution are in direct tension and
cannot both be obtained from a single frame.**

Retained in `physics/masks.py` as a diagnostic and for the ablation table.

---

## 3. The OCTA volumes are interleaved repeat pairs

`M3_*_NxNx2` volumes store `[p0r0, p0r1, p1r0, p1r1, ...]` — 512 positions × 2
repeats = 1024 files, matching the file count exactly.

Established by parity: even-start pairs share **4–6× more speckle** than
odd-start pairs across all three volumes tested. Nothing but same-position
acquisition explains that. An earlier reading of these as sequential oversampled
positions was wrong — it averaged separation-1 correlation over both parities,
halving the very signal that distinguishes them.

**Counterintuitively, position pairing beats repeat pairing for denoising:**

| Pairing | Speckle corr | Structure corr |
|---|---|---|
| Repeats (2k, 2k+1) | 0.105 | 0.963 |
| **Positions (i, i+2)** | **0.017** | 0.958 |

Noise2Noise wants structure shared and speckle *not* shared. Repeats revisit the
same scatterers, so their speckle only decorrelates as far as motion carries it.
Structure correlation is essentially equal, so position pairing wins on the axis
that matters. Both arms are exposed via `pair_mode` for the ablation.

---

## 4. Trained comparison

1500 steps, single seed, scored against registered 50-frame Maestro2 averages —
a target matching no scheme's training objective, so it cannot flatter any of
them.

| Scheme | PSNR | SSIM | vs do-nothing |
|---|---|---|---|
| noisy input (floor) | 9.57 | 0.044 | — |
| A bandgap→fullband *(original method)* | 20.13 | 0.336 | +10.57 |
| B complementary sub-band | 20.28 | **0.448** | +10.71 |
| **C frame pair, position** | **20.84** | 0.445 | **+11.27** |
| D frame pair, repeat | 19.97 | 0.398 | +10.41 |

All four genuinely denoise. **B and C are effectively tied at the top and both
beat the original method**, which sits second-from-last on SSIM.

The pair statistics measured *beforehand* predicted the speckle-removal ordering
after training: less shared speckle in the pair → more removed by the trained
model. That is the main methodological result — cheap measurements on the data
predicted expensive training outcomes.

---

## 5. Networks vs simply averaging

Reference built from frames 25–49; all inputs drawn from frames 0–24, so no
frame is shared between input and target.

| Method | PSNR | SSIM |
|---|---|---|
| average 1 frame | 9.67 | 0.032 |
| average 4 frames | 14.25 | 0.093 |
| average 8 frames | 15.62 | 0.142 |
| average 16 frames | 16.58 | 0.200 |
| **net A** (1 frame) | **16.22** | 0.236 |
| **net B** (1 frame) | 16.04 | **0.287** |
| net C (1 frame) | 15.68 | 0.257 |
| net D (1 frame) | 15.39 | 0.228 |

**A single network pass on one frame is worth roughly 8–16 averaged frames**,
and every network beats 16-frame averaging on SSIM. The networks earn their
complexity.

---

## 6. Where the sharpness comes from

Fine-scale component of each output, correlated against the reference's real
detail versus against the input's speckle.

| Scheme | vs reference ↑ | vs input speckle ↓ | Fine energy | Real:speckle |
|---|---|---|---|---|
| noisy input | 0.047 | 1.000 | 1.000 | — |
| A | 0.047 | 0.468 | 0.046 | 0.10 |
| B | 0.058 | 0.371 | 0.010 | 0.16 |
| **C** | **0.089** | **0.305** | 0.010 | **0.29** |
| D | 0.082 | 0.483 | **0.049** | 0.17 |

**D looks sharpest but the sharpness is mostly retained speckle** — the most
fine-scale energy and the highest correlation with the input's speckle. C
recovers the most real structure per unit speckle passed through.

**A's fine detail correlates with the reference at 0.047 — identical to raw
noise.** The original method's fine-scale content is no more reference-like than
the unprocessed frame.

**Denoising strength is tunable** via pair separation, which sets shared speckle:
repeat 0.105 → 1 position 0.017 → 2 positions 0.005 → 4 positions 0.001, while
structure correlation falls only 0.963 → 0.951.

---

## 7. Detector noise calibration

`var(µ) = read_var + gain·µ + rin·µ²`, fitted per acquisition from `back*.raw`
by non-negative least squares.

| Parameter | Range across 10 acquisitions |
|---|---|
| read_var | 4.44 – 4.88 ADU² (read noise 2.1 – 2.2 ADU) |
| gain | 0.0000 – 0.0613 ADU/e⁻ |
| rin | 3.3e−4 – 3.3e−3 |
| R² | 0.9908 – 0.9992 |

The quadratic RIN term is not optional: measured `var/mean` climbs from 0.24 to
4.5 across the intensity range, and a linear fit returns a **negative**
read-noise intercept, which is physically impossible.

**Refuted: pooled fitting.** The gain spread looked like shot/RIN degeneracy, so
a shared-gain fit was tried. It collapsed mean R² from 0.995 to **0.323**, three
acquisitions going worse than a constant. Detector gain is an adjustable
acquisition setting on this instrument — the Maestro2 folder names say so
outright (`gain165`, `gain167`). Per-acquisition calibration is correct.

---

## 8. Near-clean references

Registered and averaged Maestro2 repeat stacks, the only near-clean reference
this dataset admits.

| Stack | Corr before → after | Effective looks (of 50) |
|---|---|---|
| 6 mm gain165 | 0.427 → 0.558 | **36.2** |
| 6 mm YM | 0.422 → 0.425 | 31.5 |
| 6 mm ET | 0.296 → 0.343 | 25.6 |
| 3 mm anterior | 0.159 → 0.161 | **3.0 — unusable** |

Three of four are usable, cutting speckle contrast from ~0.53 to ~0.09. The
anterior stack is not.

Two guards were required. **Per-stack ROI**: the anterior scan spans rows
~80–970 while the retina stacks concentrate in ~150–360. **Reject worsening
shifts**: on low-correlation stacks a wide search finds spurious phase-
correlation peaks — the YM stack produced mean shifts of dz=22, dx=20 px and
drove one frame from 0.716 to 0.177. Zero shift is always available and always
at least as good.

---

## 9. Metric caveats — read before trusting any number here

**PSNR and SSIM reward blur.** B scores well on both while being visibly the
most over-smoothed output; its residual map is full of anatomy. This is the
failure mode that makes raw SNR untrustworthy, demonstrated on real output.

**`residual_leak` reversed ordering between datasets** (C 0.117 vs B 0.060 on
M3; C 0.188 vs B 0.231 on Maestro2). A metric that flips under domain change
should not carry ranking weight.

**`fine vs reference` is near-zero for everything, including 16-frame
averaging.** The 25-frame reference still carries ~20% residual speckle, and
that residual is random — nothing can predict it. The metric was largely
measuring correlation with the reference's own noise.

**`vs_identity` is a weak floor.** The identity pays 2σ² — its own noise plus the
held-out view's — so a ruinous σ=12 blur still scored 0.888 against it. Only a
degenerate constant output exceeds 1.0.

**The held-out-mask metric needs enough frames.** On one 64×48 frame it swapped
two near-tied models (true MSE 5% apart, scored 0.25% apart). Over 32 frames,
ranking matched a clean reference exactly (Spearman 1.0).

**Every reference-scored number here predates the alignment fix.** Sections 4,
5 and 6 compared a prediction made from frame *i* against an average registered
to frame 0 — the registration shifts were computed and then thrown away. These
stacks carry real non-monotonic motion (one frame reached dz=22 px), so those
PSNR/SSIM values charge each model for eye movement on top of denoising, and are
understated in absolute terms. The penalty is not a constant that cancels in a
ranking: misalignment costs a *sharp* output more than a blurred one, the same
direction these metrics are already biased in, so the small gaps (B vs C at
0.56 dB) are the ones least safe to trust. Fixed in `run_fair_eval.Reference`;
re-scoring needs no retraining because the scripts save checkpoints.

**The correction has now been measured, and it was not small.** Mean |shift|
across the three Maestro2 stacks is 21.5, 12.2 and 2.6 px. Re-scoring the
architecture sweep moved every trained model up by 2.6-4.3 dB and **reversed its
ranking** — see section 10. The size of the bias was not the surprise; the fact
that it was non-uniform, and therefore changed the ordering rather than shifting
it, is. Sections 4, 5 and 6 have NOT yet been re-scored and should be read as
provisional until they are.

**Corrected numbers.** An earlier repeat-frame correlation of 0.975 was inflated
by a DC artefact in a reconstruction that skipped k-linearisation; the correct
value is ~0.50. Gains of 6.25/4.46 came from the discredited linear PTC fit; the
correct values are ~0.04.

---

## 10. Architecture sweep, re-scored — the alignment bug inverted the result

Supervision held fixed at frame-pair position, 1200 steps, single seed. Scored
first against the unregistered average, then re-scored from the saved
checkpoints against the aligned one (section 9). No retraining.

| Architecture | PSNR now | was | Δ | SSIM now | was | Params | Latency | Frames |
|---|---|---|---|---|---|---|---|---|
| noisy input | 9.839 | 9.566 | +0.273 | 0.0712 | 0.0442 | — | — | — |
| **nafnet** | **25.323** | 21.042 | +4.281 | **0.5327** | 0.4620 | 6.82M | **10.7 ms** | 1 |
| restormer | 25.317 | 20.993 | +4.324 | 0.5150 | 0.4441 | 6.53M | 50.3 ms | 1 |
| aniso_resunet | 25.117 | 20.927 | +4.190 | 0.5278 | 0.4615 | 9.92M | 12.2 ms | 1 |
| resunet_pseudo3d | 24.818 | 20.832 | +3.986 | 0.5125 | 0.4476 | 7.23M | 11.8 ms | 1 |
| deform_fusion | 24.428 | **21.857** | +2.571 | 0.5251 | 0.4875 | 1.09M | 33.5 ms | **5** |

**deform_fusion went from first to last.** It led the unaligned table by 0.8 dB;
aligned, it sits 0.9 dB below nafnet and below the baseline as well.

The delta column measures the bias predicted in section 9 rather than assuming
it. Misalignment costs a SHARP output more than a blurred one, so the blurriest
model is penalised least: deform_fusion gained only +2.571 dB from alignment
while every single-frame model gained 3.99-4.32. It had been winning by being
smooth enough to escape the penalty. The qualitative panels agree — in the zoom
it alone has lost the speckle texture and choroidal detail.

The noisy-input floor barely moved (+0.273 dB) for the complementary reason:
raw speckle matches the reference at no alignment, so registering it changes
little. **That asymmetry is why the defect stayed invisible** — it was near-zero
against the floor and ~4 dB against every trained model.

**Corrected reading.** nafnet and restormer tie on PSNR (0.006 dB apart, inside
noise), but nafnet wins SSIM by 0.018 and runs 4.7x faster. nafnet beats the
baseline by 0.505 dB and +0.0202 SSIM, past the ~0.2 dB resolution threshold for
this protocol; before the fix that same gap read as a marginal 0.227 dB. It does
so with fewer parameters and the lowest latency measured.

Not settled by this run: single seed at 1200 steps. deform_fusion's row remains
not like-for-like in either direction — it consumes 5 frames against 1.

`ffc_resunet` did not train: its Fourier branch cast only the FFT input to
float32 while the conv between the transforms still ran under autocast and
returned bfloat16, which `torch.complex` rejects. Fixed, with every registered
backbone now covered by a bf16 forward-and-backward test.

---

## 11. Open / incomplete

- **Controlled B-vs-C comparison**, 3 seeds × 6000 steps. B's three seeds gave
  PSNR 19.987 / 20.071 / 20.201 — mean 20.09, sd 0.11. The single-seed gap that
  prompted the run was 0.56 dB, five times that spread, so it survives unless
  C's own spread is much wider. C is still training.
- **Re-score the remaining checkpoints.** The architecture sweep is done (§10).
  Sections 4, 5 and 6 and the controlled comparison still carry the unaligned
  numbers; all of them save `.pt` files, so each is an eval pass via
  `experiments/rescore_checkpoints.py`, not a retrain.
- **Confirm the sweep at more than one seed.** §10 is a single seed at 1200
  steps. The nafnet-over-baseline gap (0.505 dB) clears the ~0.2 dB threshold,
  but nafnet vs restormer (0.006 dB) is not a result.
- **README results table** has not been regenerated since the SNR estimator fix.
  Because `max ≥ p99.99`, the published ΔSNR was **understated**; re-running
  should move those numbers up.
- **Registration is not wired into a CLI** — it exists as a library plus
  experiment scripts.
- **mypy ratchet**: 13 of 52 modules excluded, listed largest-first in
  `pyproject.toml`. Two entries are real latent bugs, not missing annotations.

---

## Reliability note

Seven claims made from reasoning were overturned by measurement: the 0.975
repeat correlation, the mask decorrelation advantage, the pooled-gain
hypothesis, a "slow-axis blur" that measured a quantity the network never
computes, a registration regression that compared linear-domain against
log-domain correlation, a prediction that scheme C would fail, and
`deform_fusion`'s multi-frame lead — which survived exactly as long as the
reference went unregistered.

The seventh is the sharpest instance of the pattern, because the reasoning was
not merely unsupported but *self-consistent and wrong*: the multi-frame model
was expected to win on the grounds that it sees more information, it did win on
the metric, and the metric was broken in the one direction that rewards the
thing that model actually does. A plausible mechanism plus a confirming number
is still not a result if the number was never checked.

The pattern is consistent — reasoning ahead of measurement failed; measuring
first held. Treat any claim here without a number attached as provisional.
