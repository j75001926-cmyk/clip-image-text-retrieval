#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import ast
import json
import types
from pathlib import Path
from typing import List

import faiss
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from eval_test_checkpoint import load_lora_model


ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_caption_list(raw_value):
    if isinstance(raw_value, list):
        return [str(x) for x in raw_value]

    if isinstance(raw_value, str):
        try:
            value = ast.literal_eval(raw_value)
            if isinstance(value, list):
                return [str(x) for x in value]
        except Exception:
            pass

        try:
            value = json.loads(raw_value)
            if isinstance(value, list):
                return [str(x) for x in value]
        except Exception:
            pass

    return []


def load_image_paths(csv_path: str, img_dir: str, split: str = "test", skip_missing: bool = False):
    df = pd.read_csv(csv_path)

    if split != "all":
        df = df[df["split"] == split].reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    img_dir = Path(img_dir)

    image_paths = []
    image_filenames = []

    for _, row in df.iterrows():
        filename = str(row["filename"])
        img_path = img_dir / filename

        if not img_path.exists():
            if skip_missing:
                print(f"[Warning] Missing image skipped: {img_path}")
                continue
            raise FileNotFoundError(f"Image not found: {img_path}")

        image_paths.append(str(img_path))
        image_filenames.append(filename)

    print("=" * 80)
    print("Image data for FAISS index")
    print("=" * 80)
    print(f"Split: {split}")
    print(f"Num images: {len(image_paths)}")
    print("=" * 80)

    return image_paths, image_filenames


class ImageOnlyDataset(Dataset):
    def __init__(self, image_paths: List[str], preprocess):
        self.image_paths = image_paths
        self.preprocess = preprocess

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.preprocess(image)
        return image


@torch.no_grad()
def encode_images(model, preprocess, image_paths, batch_size, num_workers, device, amp):
    dataset = ImageOnlyDataset(image_paths, preprocess)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
    )

    all_features = []

    model.eval()

    for images in tqdm(loader, desc="Encoding images for FAISS"):
        images = images.to(device, non_blocking=True)

        if amp and device.startswith("cuda"):
            with torch.cuda.amp.autocast():
                feats = model.encode_image(images)
        else:
            feats = model.encode_image(images)

        feats = F.normalize(feats.float(), dim=-1)
        all_features.append(feats.cpu())

    features = torch.cat(all_features, dim=0)
    return features.numpy().astype("float32")


def build_faiss_index(image_features: np.ndarray):
    """
    因为 CLIP embedding 已经 L2 normalize，
    用 IndexFlatIP 做内积搜索，等价于 cosine similarity。
    """
    dim = image_features.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(image_features)

    print("=" * 80)
    print("FAISS index built")
    print("=" * 80)
    print(f"Index type: IndexFlatIP")
    print(f"Num vectors: {index.ntotal}")
    print(f"Dim: {dim}")
    print("=" * 80)

    return index


def main():
    args = types.SimpleNamespace(
        # data
        csv_path="/root/autodl-tmp/flickr_annotations_30k.csv",
        img_dir="/root/autodl-tmp/flickr30k-images",

        # demo 图库范围：
        # "test" 更快；"all" 可以检索全 Flickr30k
        split="test",

        # final checkpoint
        mode="lora",
        ckpt_path="./outputs_hard_negative/best_hardneg_lora.pt",

        # fallback
        model_name="ViT-B-32",
        pretrained="/root/autodl-tmp/model/open_clip_model.safetensors",

        # system
        device="cuda" if torch.cuda.is_available() else "cpu",
        amp=True,
        batch_size=64,
        num_workers=4,
        skip_missing=False,

        # output
        output_dir="./demo_faiss",
    )

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("Build FAISS Index for Image-Text Retrieval Demo")
    print("=" * 80)
    print(f"Split: {args.split}")
    print(f"Checkpoint: {args.ckpt_path}")
    print(f"Device: {args.device}")
    print("=" * 80)

    image_paths, image_filenames = load_image_paths(
        csv_path=args.csv_path,
        img_dir=args.img_dir,
        split=args.split,
        skip_missing=args.skip_missing,
    )

    model, tokenizer, preprocess, model_name, pretrained = load_lora_model(args)

    image_features = encode_images(
        model=model,
        preprocess=preprocess,
        image_paths=image_paths,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        amp=args.amp,
    )

    index = build_faiss_index(image_features)

    index_path = os.path.join(args.output_dir, f"faiss_{args.split}.index")
    meta_path = os.path.join(args.output_dir, f"metadata_{args.split}.json")
    feat_path = os.path.join(args.output_dir, f"image_features_{args.split}.npy")

    faiss.write_index(index, index_path)
    np.save(feat_path, image_features)

    metadata = {
        "split": args.split,
        "model_name": model_name,
        "pretrained": pretrained,
        "ckpt_path": args.ckpt_path,
        "image_paths": image_paths,
        "image_filenames": image_filenames,
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("\nSaved files:")
    print(f"FAISS index: {index_path}")
    print(f"Metadata:    {meta_path}")
    print(f"Features:    {feat_path}")


if __name__ == "__main__":
    main()


# In[ ]:




