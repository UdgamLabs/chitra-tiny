"""Chitra Tiny: a 212M-parameter text-to-image model (256x256) by UdgamLabs. Standalone inference code (Apache-2.0).

    from chitra_tiny import ChitraTiny
    pipe = ChitraTiny.from_pretrained("UdgamLabs/Chitra-Tiny")       # or a local folder
    images = pipe("a lighthouse on a rocky coast at sunset", n=4)   # list of PIL images

Model: diffusion transformer (patch 2 on 32x32x4 latents) trained from scratch with rectified flow. Text encoder:
google/flan-t5-base (Apache-2.0). Image decoder: stabilityai/sd-vae-ft-mse (MIT). Both are downloaded on first use.
"""
import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------------------------------------------- model
def _sincos_2d(d, n):
    y, x = torch.meshgrid(torch.arange(n, dtype=torch.float32), torch.arange(n, dtype=torch.float32), indexing="ij")
    q = d // 4
    f = torch.exp(-math.log(10000.0) * torch.arange(q, dtype=torch.float32) / q)
    return torch.cat([torch.sin(x.flatten()[:, None] * f), torch.cos(x.flatten()[:, None] * f),
                      torch.sin(y.flatten()[:, None] * f), torch.cos(y.flatten()[:, None] * f)], dim=1)


def _timestep_embedding(t, dim=256):
    half = dim // 2
    f = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    a = t.float()[:, None] * 1000.0 * f[None]
    return torch.cat([a.cos(), a.sin()], dim=-1)


def _modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class _Attention(nn.Module):
    def __init__(self, d, h, cross=False):
        super().__init__()
        self.h, self.cross = h, cross
        if cross:
            self.q = nn.Linear(d, d); self.kv = nn.Linear(d, 2 * d)
        else:
            self.qkv = nn.Linear(d, 3 * d)
        self.q_norm = nn.RMSNorm(d // h, eps=1e-6); self.k_norm = nn.RMSNorm(d // h, eps=1e-6)
        self.proj = nn.Linear(d, d)

    def forward(self, x, ctx=None, mask=None):
        B, N, D = x.shape
        if self.cross:
            q = self.q(x); k, v = self.kv(ctx).chunk(2, dim=-1)
        else:
            q, k, v = self.qkv(x).chunk(3, dim=-1)
        sh = lambda t: t.view(B, -1, self.h, D // self.h).transpose(1, 2)  # noqa: E731
        q, k, v = self.q_norm(sh(q)), self.k_norm(sh(k)), sh(v)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=None if mask is None else mask[:, None, None, :])
        return self.proj(o.transpose(1, 2).reshape(B, N, D))


class _Block(nn.Module):
    def __init__(self, d, h, mlp_ratio):
        super().__init__()
        self.n1 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6); self.attn = _Attention(d, h)
        self.n2 = nn.LayerNorm(d, elementwise_affine=True, eps=1e-6); self.xattn = _Attention(d, h, cross=True)
        self.n3 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        hid = int(d * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(d, hid), nn.GELU(approximate="tanh"), nn.Linear(hid, d))
        self.ada = nn.Linear(d, 6 * d)

    def forward(self, x, c, cmask, temb):
        sh1, sc1, g1, sh2, sc2, g2 = self.ada(F.silu(temb)).chunk(6, dim=-1)
        x = x + g1.unsqueeze(1) * self.attn(_modulate(self.n1(x), sh1, sc1))
        x = x + self.xattn(self.n2(x), c, cmask)
        return x + g2.unsqueeze(1) * self.mlp(_modulate(self.n3(x), sh2, sc2))


class ChitraTinyDiT(nn.Module):
    """Predicts the flow velocity v = noise - x0 for latents x_t = (1 - t) x0 + t noise (t=1 is pure noise)."""

    def __init__(self, width=768, depth=16, heads=12, patch=2, in_ch=4, grid=32, txt_dim=768, mlp_ratio=4.0, **_):
        super().__init__()
        self.patch, self.in_ch, self.n_side = patch, in_ch, grid // patch
        d = width
        self.patch_in = nn.Linear(in_ch * patch * patch, d)
        self.register_buffer("pos", _sincos_2d(d, self.n_side)[None], persistent=False)
        self.txt_in = nn.Sequential(nn.Linear(txt_dim, d), nn.GELU(approximate="tanh"), nn.Linear(d, d))
        self.t_in = nn.Sequential(nn.Linear(256, d), nn.SiLU(), nn.Linear(d, d))
        self.pool_in = nn.Linear(d, d)
        self.blocks = nn.ModuleList(_Block(d, heads, mlp_ratio) for _ in range(depth))
        self.norm_out = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.ada_out = nn.Linear(d, 2 * d)
        self.out = nn.Linear(d, in_ch * patch * patch)

    def forward(self, x, t, txt, txt_mask):
        B, C, H, W = x.shape; p = self.patch
        h = x.view(B, C, H // p, p, W // p, p).permute(0, 2, 4, 1, 3, 5).reshape(B, (H // p) * (W // p), C * p * p)
        h = self.patch_in(h) + self.pos
        c = self.txt_in(txt)
        m = txt_mask.to(c.dtype).unsqueeze(-1)
        temb = self.t_in(_timestep_embedding(t)) + self.pool_in((c * m).sum(1) / m.sum(1).clamp(min=1.0))
        for blk in self.blocks:
            h = blk(h, c, txt_mask, temb)
        sh, sc = self.ada_out(F.silu(temb)).chunk(2, dim=-1)
        v = self.out(_modulate(self.norm_out(h), sh, sc))
        n = self.n_side
        return v.view(B, n, n, C, p, p).permute(0, 3, 1, 4, 2, 5).reshape(B, C, n * p, n * p)


# ------------------------------------------------------------------------------------------------------------- pipeline
class ChitraTiny:
    def __init__(self, model, config, device):
        from diffusers import AutoencoderKL
        from transformers import AutoTokenizer, T5EncoderModel
        self.model, self.cfg, self.device = model, config, device
        self.tok = AutoTokenizer.from_pretrained(config["text_encoder"])
        self.te = T5EncoderModel.from_pretrained(config["text_encoder"]).to(device).eval()
        self.vae = AutoencoderKL.from_pretrained(config["vae"]).to(device).eval()

    @classmethod
    def from_pretrained(cls, path_or_repo="UdgamLabs/Chitra-Tiny", device=None):
        from safetensors.torch import load_file
        if not os.path.isdir(path_or_repo):
            from huggingface_hub import snapshot_download
            path_or_repo = snapshot_download(path_or_repo, allow_patterns=["config.json", "model.safetensors"])
        device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        cfg = json.load(open(os.path.join(path_or_repo, "config.json")))
        model = ChitraTinyDiT(**cfg["model"])
        model.load_state_dict(load_file(os.path.join(path_or_repo, "model.safetensors")))
        return cls(model.to(device).eval(), cfg, device)

    @torch.no_grad()
    def _encode(self, texts):
        e = self.tok(texts, max_length=self.cfg["max_text_tokens"], truncation=True, padding="max_length", return_tensors="pt")
        ids, mask = e.input_ids.to(self.device), e.attention_mask.to(self.device)
        return self.te(input_ids=ids, attention_mask=mask).last_hidden_state.float(), mask.bool()

    @torch.no_grad()
    def __call__(self, prompt, n=1, guidance=None, steps=None, seed=0, negative_prompt=""):
        """Returns a list of n PIL images (256x256). guidance: classifier-free guidance scale (default from config)."""
        from PIL import Image
        guidance = self.cfg["default_guidance"] if guidance is None else guidance
        steps = self.cfg["default_steps"] if steps is None else steps
        txt, m = self._encode([prompt] * n); nt, nm = self._encode([negative_prompt] * n)
        g = self.cfg["latent_grid"]
        x = torch.randn((n, self.cfg["latent_channels"], g, g), generator=torch.Generator().manual_seed(seed)).to(self.device)
        ts = torch.linspace(1.0, 0.0, steps + 1, device=self.device)
        for i in range(steps):  # Euler integration of the flow from noise (t=1) to image (t=0)
            v = self.model(torch.cat([x, x]), ts[i].expand(2 * n), torch.cat([txt, nt]), torch.cat([m, nm]))
            vc, vu = v.chunk(2)
            x = x + (ts[i + 1] - ts[i]) * (vu + guidance * (vc - vu))
        img = self.vae.decode(x / self.cfg["vae_scale"]).sample.clamp(-1, 1)
        img = (img * 127.5 + 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()
        return [Image.fromarray(a) for a in img]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Generate images with Chitra Tiny")
    ap.add_argument("prompt"); ap.add_argument("--n", type=int, default=4); ap.add_argument("--guidance", type=float)
    ap.add_argument("--steps", type=int); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="UdgamLabs/Chitra-Tiny"); ap.add_argument("--out", default="chitra")
    a = ap.parse_args()
    pipe = ChitraTiny.from_pretrained(a.model)
    for i, im in enumerate(pipe(a.prompt, n=a.n, guidance=a.guidance, steps=a.steps, seed=a.seed)):
        im.save(f"{a.out}_{i}.png"); print(f"{a.out}_{i}.png")
