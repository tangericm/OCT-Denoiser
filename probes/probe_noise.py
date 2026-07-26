"""Read-only: proper photon-transfer-curve from BACKGROUND frames (no sample),
plus repeat-noise check on the Maestro2 50-frame line scans."""
import os
import numpy as np

ROOT = r"C:\Users\erict\OneDrive\Desktop\Projects\OCT Data"


def load(path, pixels, alines):
    a = np.fromfile(path, dtype=np.uint16)
    return a.reshape((pixels, alines), order="F").astype(np.float64)


def ptc(stack, label, nb=14):
    """stack [N,H,W] of repeated identical-condition frames -> var vs mean."""
    m = stack.mean(axis=0).ravel()
    v = stack.var(axis=0, ddof=1).ravel()
    o = np.argsort(m); m, v = m[o], v[o]
    edges = np.linspace(0, m.size, nb + 1).astype(int)
    print(f"\n  [{label}]  N={stack.shape[0]} frames")
    print(f"  {'mean':>10} {'variance':>12} {'var/mean':>10}")
    mm, vv = [], []
    for k in range(nb):
        sl = slice(edges[k], edges[k + 1])
        if sl.stop > sl.start:
            a, b = m[sl].mean(), v[sl].mean()
            mm.append(a); vv.append(b)
            print(f"  {a:>10.1f} {b:>12.1f} {b/max(a,1e-9):>10.3f}")
    mm, vv = np.array(mm), np.array(vv)
    A = np.vstack([mm, np.ones_like(mm)]).T
    g, r = np.linalg.lstsq(A, vv, rcond=None)[0]
    print(f"  fit: var = {g:.4f}*mean + {r:.1f}   (gain={g:.4f} ADU/e-, read_var={r:.1f})")
    return g, r


print("=" * 72)
print("PROPER PTC — background frames, no sample. Mean varies across k via the")
print("source spectrum, variance is temporal across repeats. Uncontaminated.")
print("=" * 72)
for folder, alines in [("M3_Macula_6x6mm_512x512x2", 512),
                       ("M3_Disc_9x9mm_512x512x2", 512)]:
    d = os.path.join(ROOT, "Maestro3", folder)
    backs = sorted(f for f in os.listdir(d) if f.startswith("back"))
    B = np.stack([load(os.path.join(d, f), 2048, alines) for f in backs])
    ptc(B, folder)

print()
print("=" * 72)
print("MAESTRO2 50-FRAME REPEATS — same line position, real repeats")
print("=" * 72)
d2 = os.path.join(ROOT, "Maestro2", "Line_6mm_2048Aline_135degCW_50frame_gain165")
S = np.stack([load(os.path.join(d2, f"bscan{i:06d}.raw"), 2048, 2048) for i in range(12)])
print(f"  loaded {S.shape}")
ptc(S, "50frame repeats (raw spectra, frames 0-11)")
print("  NOTE: this conflates detector noise with speckle decorrelation from")
print("  eye motion across frames -- it is an UPPER bound on detector noise.")

# Adjacent-frame difference is the tightest detector-noise estimate available.
dif = (S[1] - S[0])
print(f"\n  adjacent-frame diff: std={dif.std():.2f} -> per-frame noise std ~ {dif.std()/np.sqrt(2):.2f}")
print(f"  frame mean level={S[0].mean():.1f}")

print()
print("=" * 72)
print("SPECKLE DECORRELATION ACROSS THE 50 REPEATS (after DC removal)")
print("=" * 72)


def bscan(spec, crop=(0, 1024)):
    s = spec - spec.mean(axis=1, keepdims=True)
    return np.abs(np.fft.ifft(s, axis=0))[crop[0]:crop[1], :]


b0 = bscan(S[0])
for j in (1, 2, 4, 8, 11):
    bj = bscan(S[j])
    a = (b0 - b0.mean()).ravel(); c = (bj - bj.mean()).ravel()
    print(f"  linear-magnitude corr(frame0, frame{j:2d}) = "
          f"{float(a @ c / (np.linalg.norm(a)*np.linalg.norm(c))):+.4f}")
