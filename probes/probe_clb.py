import numpy as np

p = r"C:\Users\erict\OneDrive\Desktop\Projects\OCT Data\Maestro3\000003_3DOCT-1_FUNDUS.CLB"
with open(p, "rb") as f:
    head = f.read(64)
    raw = f.read()
a = np.frombuffer(raw, dtype=np.float32)
print(f"file bytes after 64-byte header: {len(raw)}   float32 count: {a.size}")
print(f"first 8 : {a[:8]}")
print(f"last 8  : {a[-8:]}")
r = a[:2048]
print(f"\nresampling[:2048]  min={r.min():.6f}  max={r.max():.6f}  monotonic={bool(np.all(np.diff(r) > 0))}")
print(f"  mean step={np.diff(r).mean():.8f}  step min={np.diff(r).min():.8f} max={np.diff(r).max():.8f}")
d = r - np.linspace(r[0], r[-1], 2048)
print(f"  max deviation from linear: {np.abs(d).max():.6f}")
print(f"\nheader first 32 bytes: {head[:32]!r}")
