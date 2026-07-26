"""Evaluate the saved scheme-A checkpoint with the fixed multi-channel input."""
import json
import os

import torch

from octdenoiser.experiments.run_supervision_ablation import evaluate
from octdenoiser.networks import create_model

ROOT = r"C:\Users\erict\OneDrive\Desktop\Projects\OCT Data\Maestro3"
OUT = "runs/supervision_ablation"
CKPT = os.path.join(OUT, "A_bandgap_fullband.pt")

ck = torch.load(CKPT, map_location="cpu", weights_only=True)
model = create_model("resunet_pseudo3d", base=32, in_ch=ck["in_ch"]).cuda()
model.load_state_dict(ck["model"], strict=True)
print(f"loaded {CKPT}  in_ch={ck['in_ch']}")

m = evaluate(model, ROOT, "cuda", "A_bandgap_fullband", n_frames=12)
m.update(in_ch=ck["in_ch"], steps=1500, checkpoint=CKPT, final_loss=0.96356)
print(json.dumps(m, indent=2))

res_path = os.path.join(OUT, "results.json")
with open(res_path, encoding="utf-8") as f:
    results = json.load(f)
results["A_bandgap_fullband"] = m
with open(res_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print(f"\nupdated {res_path}")

print()
print("=" * 84)
print(f"{'scheme':<22}{'in_ch':>7}{'speckle_ratio':>15}{'structure_keep':>16}{'residual_leak':>15}")
print("=" * 84)
for name in ("A_bandgap_fullband", "B_complementary", "C_pair_position", "D_pair_repeat"):
    r = results.get(name, {})
    if "error" in r or not r:
        print(f"{name:<22}  (missing)")
        continue
    print(f"{name:<22}{r['in_ch']:>7}{r['speckle_ratio']:>15.4f}"
          f"{r['structure_keep']:>16.4f}{r['residual_leak']:>15.4f}")
