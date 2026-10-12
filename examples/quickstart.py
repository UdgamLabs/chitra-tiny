"""Four images for one prompt, saved as a 2x2 grid.
    python examples/quickstart.py "a sailing ship on a calm sea"     -> examples/out/quickstart.png"""
import os
import sys

from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from chitra_tiny import ChitraTiny  # noqa: E402

prompt = sys.argv[1] if len(sys.argv) > 1 else "a sailing ship on a calm sea"
pipe = ChitraTiny.from_pretrained("UdgamLabs/Chitra-Tiny")   # downloads 0.85 GB once, plus the text encoder and decoder
images = pipe(prompt, n=4, guidance=7, steps=50, seed=0)      # four different pictures from one seed

grid = Image.new("RGB", (512, 512))
for i, im in enumerate(images):
    grid.paste(im, ((i % 2) * 256, (i // 2) * 256))
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(out, exist_ok=True)
grid.save(os.path.join(out, "quickstart.png"))
print(f"saved {os.path.join(out, 'quickstart.png')} ({pipe.device})")
