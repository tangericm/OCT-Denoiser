"""Read-only probe: determine repeat structure + noise stats in the OCT data."""
import os
import numpy as np

ROOT = r"C:\Users\erict\OneDrive\Desktop\Projects\OCT Data"


def load(path, pixels, alines):
    a = np.fromfile(path, dtype=np.uint16)
    assert a.size == pixels * alines, f"{path}: {a.size} != {pixels*alines}"
    return a.reshape((pixels, alines), order="F").astype(np.float32)


def corr(a, b):
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def quick_bscan(spec, crop=(0, 1024)):
    """DC-subtract -> IFFT -> log magnitude. No k-resample (structure only)."""
    s = spec - spec.mean(axis=1, keepdims=True)
    d = np.abs(np.fft.ifft(s, axis=0))[crop[0]:crop[1], :]
    return np.log10(d + 1e-6).astype(np.float32)


print("=" * 70)
print("A) REPEAT STRUCTURE — M3_Macula_6x6mm_512x512x2 (2048 x 512)")
print("=" * 70)
d = os.path.join(ROOT, "Maestro3", "M3_Macula_6x6mm_512x512x2")
idx = [0, 1, 2, 3, 512, 513]
fr = {i: load(os.path.join(d, f"bscan{i:03d}000.raw"), 2048, 512) for i in idx}
bs = {i: quick_bscan(v) for i, v in fr.items()}
for a, b, label in [(0, 1, "0-1  consecutive"),
                    (1, 2, "1-2  consecutive"),
                    (0, 2, "0-2  skip one"),
                    (0, 512, "0-512 half-volume"),
                    (0, 513, "0-513 half+1")]:
    print(f"  {label:20s}  raw_corr={corr(fr[a], fr[b]):+.4f}   bscan_corr={corr(bs[a], bs[b]):+.4f}")

print()
print("=" * 70)
print("B) MAESTRO2 50-FRAME LINE REPEATS (2048 x 2048)")
print("=" * 70)
d2 = os.path.join(ROOT, "Maestro2", "Line_6mm_2048Aline_135degCW_50frame_gain165")
f2 = {i: load(os.path.join(d2, f"bscan{i:06d}.raw"), 2048, 2048) for i in (0, 1, 2, 25, 49)}
b2 = {i: quick_bscan(v) for i, v in f2.items()}
for a, b in [(0, 1), (0, 2), (0, 25), (0, 49)]:
    print(f"  frame {a:2d} vs {b:2d}   raw_corr={corr(f2[a], f2[b]):+.4f}   bscan_corr={corr(b2[a], b2[b]):+.4f}")

print()
print("=" * 70)
print("C) DETECTOR NOISE MODEL — background frames (photon transfer curve)")
print("=" * 70)
backs = sorted(f for f in os.listdir(d) if f.startswith("back"))
print(f"  found {len(backs)} background frames: {backs}")
B = np.stack([load(os.path.join(d, f), 2048, 512) for f in backs])
print(f"  stack {B.shape}  mean={B.mean():.2f}  std_overall={B.std():.2f}")
print(f"  temporal std (per-pixel, averaged) = {B.std(axis=0).mean():.3f}")
print(f"  spatial  std of temporal mean      = {B.mean(axis=0).std():.3f}   <- fixed pattern")

fpn = load(os.path.join(d, "FPN.raw"), 2048, 512)
print(f"  FPN.raw mean={fpn.mean():.2f} std={fpn.std():.2f}  corr(FPN, back_mean)={corr(fpn, B.mean(axis=0)):+.4f}")

print()
print("  -- variance-vs-mean on SAMPLE frames (shot-noise slope = gain) --")
S = np.stack([fr[i] for i in (0, 1, 2, 3)])
m = S.mean(axis=0).ravel()
v = S.var(axis=0, ddof=1).ravel()
order = np.argsort(m)
m, v = m[order], v[order]
nb = 12
edges = np.linspace(0, m.size, nb + 1).astype(int)
print(f"  {'mean_bin':>12} {'variance':>12}")
for k in range(nb):
    sl = slice(edges[k], edges[k + 1])
    if sl.stop > sl.start:
        print(f"  {m[sl].mean():>12.1f} {v[sl].mean():>12.1f}")
lo, hi = m[: m.size // 4], v[: m.size // 4]
A = np.vstack([lo, np.ones_like(lo)]).T
slope, icept = np.linalg.lstsq(A, hi, rcond=None)[0]
print(f"\n  linear fit (low quartile): var = {slope:.4f}*mean + {icept:.1f}")
print(f"    -> gain ~ {slope:.4f} ADU/e-,  read-noise var ~ {icept:.1f} ADU^2")
