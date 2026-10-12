# Examples

Run from the repository root after `pip install -r requirements.txt`. Each script downloads the released weights from [UdgamLabs/Chitra-Tiny](https://huggingface.co/UdgamLabs/Chitra-Tiny) on first use (no login) and writes to `examples/out/`.

| Script | Command | Output |
|---|---|---|
| `quickstart.py` | `python examples/quickstart.py "a sailing ship on a calm sea"` | `out/quickstart.png`: four images, seed 0, guidance 7 |
| `guidance_sweep.py` | `python examples/guidance_sweep.py "a cottage in a snowy forest" --guidance 1 3 5 7 10` | `out/guidance_sweep.png`: one image per guidance value |
| `make_walkthrough_images.py` | `python examples/make_walkthrough_images.py` | rewrites `docs/assets/img/` (26 images) and `images.json` |

All three were run on 12 October 2026 against the released weights on an Apple-silicon Mac (MPS). On another device the same seed gives a near-identical image.
