#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import json
import types

import faiss
import numpy as np
import gradio as gr
from PIL import Image

import torch
import torch.nn.functional as F

from eval_test_checkpoint import load_lora_model


def load_demo_resources():
    args = types.SimpleNamespace(
        # checkpoint
        mode="lora",
        ckpt_path="./outputs_hard_negative/best_hardneg_lora.pt",

        # fallback
        model_name="ViT-B-32",
        pretrained="/root/autodl-tmp/model/open_clip_model.safetensors",

        # FAISS
        index_path="./demo_faiss/faiss_test.index",
        metadata_path="./demo_faiss/metadata_test.json",

        # system
        device="cuda" if torch.cuda.is_available() else "cpu",
        amp=True,
    )

    print("=" * 80)
    print("Loading demo resources")
    print("=" * 80)
    print(f"Checkpoint: {args.ckpt_path}")
    print(f"FAISS index: {args.index_path}")
    print(f"Metadata: {args.metadata_path}")
    print(f"Device: {args.device}")
    print("=" * 80)

    model, tokenizer, preprocess, model_name, pretrained = load_lora_model(args)

    index = faiss.read_index(args.index_path)

    with open(args.metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    image_paths = metadata["image_paths"]
    image_filenames = metadata["image_filenames"]

    print(f"Loaded FAISS index with {index.ntotal} vectors.")
    print(f"Loaded {len(image_paths)} image paths.")

    return args, model, tokenizer, index, image_paths, image_filenames


@torch.no_grad()
def encode_query_text(query: str, model, tokenizer, device: str, amp: bool):
    tokens = tokenizer([query]).to(device)

    model.eval()

    if amp and device.startswith("cuda"):
        with torch.cuda.amp.autocast():
            text_feature = model.encode_text(tokens)
    else:
        text_feature = model.encode_text(tokens)

    text_feature = F.normalize(text_feature.float(), dim=-1)

    return text_feature.cpu().numpy().astype("float32")


def search_images(query: str, top_k: int):
    query = query.strip()

    if len(query) == 0:
        return [], "Please input a non-empty text query."

    top_k = int(top_k)

    query_feature = encode_query_text(
        query=query,
        model=MODEL,
        tokenizer=TOKENIZER,
        device=ARGS.device,
        amp=ARGS.amp,
    )

    scores, indices = INDEX.search(query_feature, top_k)

    scores = scores[0]
    indices = indices[0]

    gallery_items = []
    lines = []

    for rank, (idx, score) in enumerate(zip(indices, scores), start=1):
        img_path = IMAGE_PATHS[int(idx)]
        filename = IMAGE_FILENAMES[int(idx)]

        if os.path.exists(img_path):
            img = Image.open(img_path).convert("RGB")
            caption = f"#{rank} | {filename} | score={float(score):.4f}"
            gallery_items.append((img, caption))
            lines.append(caption)
        else:
            lines.append(f"#{rank} | Missing image: {img_path}")

    return gallery_items, "\n".join(lines)


ARGS, MODEL, TOKENIZER, INDEX, IMAGE_PATHS, IMAGE_FILENAMES = load_demo_resources()


demo = gr.Interface(
    fn=search_images,
    inputs=[
        gr.Textbox(
            label="Text Query",
            placeholder="Example: a man wearing an orange hat and glasses",
            lines=2,
        ),
        gr.Slider(
            minimum=1,
            maximum=20,
            value=10,
            step=1,
            label="Top-K",
        ),
    ],
    outputs=[
        gr.Gallery(label="Retrieved Images", columns=5, height="auto"),
        gr.Textbox(label="Ranking Details", lines=12),
    ],
    title="CLIP Image-Text Retrieval Demo",
    description=(
        "Text-to-Image retrieval demo based on OpenCLIP + LoRA + Hard Negative training. "
        "Image embeddings are indexed by FAISS."
    ),
    examples=[
        ["a man wearing an orange hat and glasses", 10],
        ["a dog running through the grass", 10],
        ["a child in a pink dress climbing stairs", 10],
        ["two men working with construction equipment", 10],
        ["people riding bicycles on a street", 10],
    ],
)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )

