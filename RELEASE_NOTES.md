# Chitra Tiny 1.0.0

First public release, 12 October 2026. The weights are on [Hugging Face](https://huggingface.co/UdgamLabs/Chitra-Tiny). This repository adds the code, runnable examples, evaluation details, a visual walkthrough and checksums.

- **One model:** a 212M-parameter diffusion transformer (16 layers, width 768) for 256×256 English text-to-image, trained from scratch with rectified flow on about 4.7 million public-domain and CC0 images.
- **Uses two frozen open models**, downloaded on first use: google/flan-t5-base (text encoder, Apache-2.0) and stabilityai/sd-vae-ft-mse (image decoder, MIT).
- **Format:** one fp32 safetensors file (0.85 GB, EMA weights) plus `config.json`, and a standalone PyTorch pipeline (`chitra_tiny.py`) with no dependence on diffusers pipelines.
- **Runs on** NVIDIA GPUs, Apple silicon (MPS) and CPU; about 3 seconds per image on an Apple-silicon Mac.
- **Measured:** FID-5k 22.1 on held-out in-distribution images, CLIP score 26.2 (real images 27.5), GenEval 0.219 at guidance 7 (Stable Diffusion 1.5: 0.426 in the same harness). See [EVALUATION.md](docs/EVALUATION.md).
- **Weak at:** less common objects, several objects, counting, positions, people and faces, and text in images. No safety filter of its own.
- **Licence:** Apache-2.0.

## In this repository

| Path | Contents |
|---|---|
| `README.md` | Overview, quick start, how it works, results and limits |
| `chitra_tiny.py`, `app.py`, `LICENSE` | Byte-identical to the Hugging Face copies |
| `examples/` | Quickstart grid, guidance sweep, and the script that regenerates the walkthrough images |
| `docs/` | [Evaluation details](docs/EVALUATION.md) and the [visual walkthrough](https://udgamlabs.github.io/chitra-tiny/) |
| `SHA256SUMS` | SHA-256 of every published Hugging Face file except the model card and `.gitattributes` |
