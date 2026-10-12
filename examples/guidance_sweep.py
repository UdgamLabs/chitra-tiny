"""The same prompt and seed at several guidance values, side by side, to see what guidance does.
    python examples/guidance_sweep.py "a cottage in a snowy forest" --seed 0    -> examples/out/guidance_sweep.png"""
import argparse
import os
import sys

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from chitra_tiny import ChitraTiny  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("prompt", nargs="?", default="a cottage in a snowy forest")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--guidance", type=float, nargs="+", default=[1, 3, 5, 7, 10])
ap.add_argument("--steps", type=int, default=50)
a = ap.parse_args()

pipe = ChitraTiny.from_pretrained("UdgamLabs/Chitra-Tiny")
strip = Image.new("RGB", (256 * len(a.guidance), 280), "white")
draw = ImageDraw.Draw(strip)
for i, g in enumerate(a.guidance):
    strip.paste(pipe(a.prompt, guidance=g, steps=a.steps, seed=a.seed)[0], (256 * i, 0))
    draw.text((256 * i + 8, 262), f"guidance {g:g}", fill="black")
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(out, exist_ok=True)
strip.save(os.path.join(out, "guidance_sweep.png"))
print(f"saved {os.path.join(out, 'guidance_sweep.png')}")
