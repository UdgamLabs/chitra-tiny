"""Chitra Tiny local playground. Run: python app.py, then open http://127.0.0.1:7860
Needs: torch, transformers, diffusers, safetensors, huggingface_hub, gradio."""
import os
import time

import gradio as gr

from chitra_tiny import ChitraTiny

HERE = os.path.dirname(os.path.abspath(__file__))
pipe = ChitraTiny.from_pretrained(HERE if os.path.exists(os.path.join(HERE, "model.safetensors")) else "UdgamLabs/Chitra-Tiny")


def run(prompt, guidance, steps, seed, n, negative):
    t0 = time.time()
    images = pipe(prompt, n=int(n), guidance=guidance, steps=int(steps), seed=int(seed), negative_prompt=negative or "")
    return images, f"{int(n)} image(s), {int(steps)} steps, guidance {guidance}, seed {int(seed)}: {time.time() - t0:.1f} s on {pipe.device}"


with gr.Blocks(title="Chitra Tiny") as demo:
    gr.Markdown("## Chitra Tiny\nA 212M-parameter text-to-image model trained from scratch by UdgamLabs. 256×256 images. "
                "Best at photos, paintings, landscapes, buildings, vehicles and food.")
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt", value="a lighthouse on a rocky coast at sunset", lines=2)
            negative = gr.Textbox(label="Negative prompt (optional)")
            guidance = gr.Slider(1, 12, value=7, step=0.5, label="Guidance")
            steps = gr.Slider(10, 100, value=50, step=5, label="Steps")
            seed = gr.Number(value=0, precision=0, label="Seed")
            n = gr.Slider(1, 8, value=4, step=1, label="Images")
            btn = gr.Button("Generate", variant="primary")
        with gr.Column():
            gallery = gr.Gallery(label="Results", columns=2, object_fit="contain")
            info = gr.Markdown()
    btn.click(run, [prompt, guidance, steps, seed, n, negative], [gallery, info])

if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1", server_port=int(os.environ.get("PORT", 7860)))
