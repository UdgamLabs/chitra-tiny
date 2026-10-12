# Chitra Tiny evaluation

Every number here was measured by us, on the released weights (the EMA copy in `model.safetensors`). All runs use the sampler in [`chitra_tiny.py`](../chitra_tiny.py): 50 Euler steps of the rectified flow, with classifier-free guidance against the empty prompt.

## Summary

| Test | Chitra Tiny | Reference |
|---|---|---|
| FID-5k, held-out (lower is better) | **22.1** | n/a (in-distribution; see below) |
| CLIP score, ViT-L/14 (higher is better) | **26.2** | real held-out images: 27.5 |
| GenEval overall, guidance 7 | **0.219** | Stable Diffusion 1.5: 0.426 |
| GenEval overall, guidance 4 | 0.190 | |

## FID and CLIP

- **Images:** 5,000 held-out images from the same public-domain and CC0 collections as the training set. They were set aside before training and never trained on. One generated image per held-out caption, at guidance 4.
- **FID:** Inception features as in clean-fid, at 256×256. Because the held-out images come from the same collections as the training images, this measures how well the model reproduces that kind of image. It is **not comparable** to published FID numbers on COCO or ImageNet.
- **CLIP score:** OpenCLIP ViT-L/14, 100 × cosine similarity between each image and its caption. Real held-out images score 27.5 with their own captions, so 26.2 means the generated images match their captions nearly as well as the real photos do.

## GenEval

[GenEval](https://github.com/djghosh13/geneval) asks for objects from the 80 everyday COCO categories, in six task types, and checks the images with an object detector. We ran the official code and prompts: 553 prompts × 4 images (seeds 0–3), 50 steps.

We validated our harness on Stable Diffusion 1.5 first: it scored **0.426**, against **0.43** published by the GenEval authors.

| Task | Chitra Tiny, guidance 7 | Chitra Tiny, guidance 4 | Stable Diffusion 1.5 (guidance 9) |
|---|---|---|---|
| Single object | 0.60 | 0.56 | 0.96 |
| Two objects | 0.10 | 0.07 | 0.37 |
| Counting | 0.15 | 0.10 | 0.38 |
| Colours | 0.39 | 0.35 | 0.77 |
| Position | 0.01 | 0.02 | 0.04 |
| Colour attribution | 0.07 | 0.05 | 0.05 |
| **Overall** | **0.219** | **0.190** | **0.426** |

Stable Diffusion 1.5 has about 860M parameters in its image model (four times Chitra Tiny) and was trained on far more images.

**Why the gap?** Accuracy on single objects tracks how often an object appears in the training captions. At guidance 4, single-object accuracy for the most common third of the COCO categories is 0.85; for the middle third, 0.51; for the rarest third, 0.33. Public-domain collections have many landscapes, buildings, paintings and vehicles, and few frisbees, toasters or hair driers. Data coverage and model size, more than the sampler, set the score.

**What did not help:** rewriting GenEval prompts into the descriptive caption style of the training data ("The image shows …") lowered the score at both guidance values (0.168 at guidance 4, 0.177 at guidance 7). Raising guidance from 4 to 7 helped (0.190 → 0.219), so 7 is the default.

## Speed

About 3 seconds per image (50 steps) on an Apple-silicon Mac (MPS), including the text encoder and decoder; faster on a recent NVIDIA GPU. On a CPU, expect tens of seconds per image. `steps=25` roughly halves the time.

## Reproducing

- The images on the [walkthrough page](https://udgamlabs.github.io/chitra-tiny/) come from [`examples/make_walkthrough_images.py`](../examples/make_walkthrough_images.py).
- The noise comes from a CPU random generator with the given seed, so every device starts from the same noise. Different devices then give near-identical, but not bit-identical, images.
