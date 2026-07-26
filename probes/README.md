# probes/ — archived measurement scripts

**This directory is an archive, not maintained code.** These scripts produced the
measured numbers in [`../docs/FINDINGS.md`](../docs/FINDINGS.md). They are
committed so that every claim in that document has a reproducer.

They were written ad-hoc during the study and are preserved **exactly as they
ran**. Tidying them would risk changing what they computed, which would defeat
the point of keeping them. They are excluded from ruff and mypy for that reason
(see the comment in `pyproject.toml`).

Do not add new work here. New measurements belong in
`src/octdenoiser/experiments/` and are held to the normal gates.

## What produced what

| FINDINGS section | Script | Notes |
|---|---|---|
| §1 supervision defect, §2 view constructions | `probe_speckle.py`, `probe_decorrelation.py` | speckle correlation with structure removed |
| §2 axial PSF, mask refutation | `phase1_measure.py` | also the mask PSF table |
| §3 OCTA interleaving | `probe_parity.py`, `probe_repeats.py`, `probe_neighbors.py` | parity test establishing repeat-pair layout |
| §5 networks vs averaging | `probe_averaging_baseline.py` | K-frame averaging ladder, `KS = (1,2,4,8,16)` |
| §6 where sharpness comes from | `probe_sharpness.py` | fine-scale vs reference / vs input speckle |
| §7 detector noise calibration | `probe_noise.py`, `phase1_pooled.py` | PTC fit; pooled-fit refutation |
| §8 near-clean references | `phase1_reference.py`, `probe_reg_debug.py` | registration + averaging |
| bias budget, Wiener/coherence | `probe_bias.py`, `probe_bias_mtf.py`, `probe_wiener.py` | separation-vs-bias tradeoff |
| figures | `make_figures.py` | |

## Known limitation — these predate the alignment fix

`probe_averaging_baseline.py` (§5) and `probe_sharpness.py` (§6) load the **v1**
reference cache, which has no `shifts` array, so no per-frame alignment was
possible. Both therefore score against an average in frame 0's coordinates.

Mean |shift| on these stacks is 21.5 / 12.2 / 2.6 px, and correcting it moved
every trained model by 2.6–4.3 dB elsewhere in the study — enough to reverse one
ranking outright (FINDINGS §10). **§5 and §6 are provisional until these two
scripts are ported onto `Reference.aligned_to`.**

Both are eval-only: they load `runs/supervision_ablation/*.pt` and forward-pass.
Re-running them costs a forward pass plus, for §5, re-registering a 50-frame
stack on CPU — minutes, not a retrain.
