"""Regenerates every image on the walkthrough page (docs/assets/img/) from the released weights.
    python examples/make_walkthrough_images.py            # ~2 min on an Apple-silicon Mac
Seed 0 unless stated, guidance 7, 50 steps. Nothing is cherry-picked: each prompt is run once with the seeds listed."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from chitra_tiny import ChitraTiny  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "assets", "img")
GALLERY = [
    "a lighthouse on a rocky coast at sunset", "an oil painting of a vase of flowers",
    "a mountain landscape with snow and pine trees", "a sailing ship on a calm sea",
    "a black and white photo of a city street", "a watercolor painting of a bird on a branch",
    "a red car parked in front of a house", "a bridge over a river in autumn",
    "a castle on a hill under a cloudy sky", "a bowl of soup next to a spoon",
    "a steam locomotive at a railway station", "a field of sunflowers on a summer day",
]
GUIDANCE_PROMPT, GUIDANCE = "a cottage in a snowy forest", [1, 3, 5, 7, 10]
SEED_PROMPT, SEEDS = "an old stone church in a green valley", [0, 1, 2, 3]
WEAK = [
    ("counting", "three red apples on a table"), ("text", "a shop sign that says OPEN"),
    ("position", "a cup to the left of a laptop"), ("faces", "a close-up photo of a smiling woman"),
    ("rare object", "a frisbee on the grass"),
]


def save(im, name, meta, **kw):
    im.save(os.path.join(OUT, name), quality=92)
    meta.append({"file": name, **kw})


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    pipe = ChitraTiny.from_pretrained(os.environ.get("CHITRA_MODEL", "UdgamLabs/Chitra-Tiny"))
    meta = []
    for i, p in enumerate(GALLERY):
        save(pipe(p, seed=0)[0], f"gallery_{i:02d}.jpg", meta, section="gallery", prompt=p, seed=0, guidance=7)
    for g in GUIDANCE:
        save(pipe(GUIDANCE_PROMPT, guidance=g, seed=0)[0], f"guidance_{g:02d}.jpg", meta, section="guidance", prompt=GUIDANCE_PROMPT, seed=0, guidance=g)
    for s in SEEDS:
        save(pipe(SEED_PROMPT, seed=s)[0], f"seed_{s}.jpg", meta, section="seeds", prompt=SEED_PROMPT, seed=s, guidance=7)
    for i, (k, p) in enumerate(WEAK):
        save(pipe(p, seed=0)[0], f"weak_{i}.jpg", meta, section="weak", kind=k, prompt=p, seed=0, guidance=7)
    json.dump({"device": str(pipe.device), "steps": 50, "images": meta}, open(os.path.join(OUT, "images.json"), "w"), indent=1)
    print(f"wrote {len(meta)} images to {OUT} on {pipe.device}")
