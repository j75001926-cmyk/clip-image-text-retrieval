#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import ast
import json
import types
from pathlib import Path
from typing import List, Tuple, Dict

import pandas as pd
from PIL import Image, ImageFile
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

import open_clip
LOCAL_WEIGHT_PATH = "/root/autodl-tmp/model/open_clip_model.safetensors"

ImageFile.LOAD_TRUNCATED_IMAGES = True


# =========================
# Data utils
# =========================

def parse_caption_list(raw_value) -> List[str]:
    """
    raw 字段格式类似：
    ["caption1", "caption2", "caption3", "caption4", "caption5"]
    读取 CSV 后通常是字符串，所以需要转回 list。
    """
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

    raise ValueError(f"Cannot parse caption list from raw={raw_value}")


def build_eval_data(
    csv_path: str,
    img_dir: str,
    split: str,
    skip_missing: bool = False
) -> Tuple[List[str], List[str], torch.LongTensor, List[str]]:
    """
    返回：
    image_paths:      每张唯一图片的路径，长度 N
    captions:         所有 caption，长度约 5N
    caption_to_image: 第 j 条 caption 对应第几张 image，长度约 5N
    image_filenames:  图片文件名，长度 N
    """
    df = pd.read_csv(csv_path)

    print("=" * 80)
    print("CSV Info")
    print("=" * 80)
    print("Columns:", df.columns.tolist())
    print("Split counts:")
    print(df["split"].value_counts())
    print("=" * 80)

    df = df[df["split"] == split].reset_index(drop=True)

    if len(df) == 0:
        raise ValueError(f"No data found for split={split}")

    img_dir = Path(img_dir)

    image_paths = []
    image_filenames = []
    captions = []
    caption_to_image = []

    image_idx = 0

    for _, row in df.iterrows():
        filename = str(row["filename"])
        img_path = img_dir / filename

        if not img_path.exists():
            if skip_missing:
                print(f"[Warning] Missing image skipped: {img_path}")
                continue
            raise FileNotFoundError(
                f"Image not found: {img_path}\n"
                f"Please check --img_dir"
            )

        caps = parse_caption_list(row["raw"])

        image_paths.append(str(img_path))
        image_filenames.append(filename)

        for cap in caps:
            captions.append(cap)
            caption_to_image.append(image_idx)

        image_idx += 1

    caption_to_image = torch.LongTensor(caption_to_image)

    print("\n" + "=" * 80)
    print("Evaluation Data")
    print("=" * 80)
    print(f"Split: {split}")
    print(f"Num images: {len(image_paths)}")
    print(f"Num captions: {len(captions)}")
    print(f"Captions per image: {len(captions) / max(len(image_paths), 1):.2f}")
    print("=" * 80)

    print("\nSanity check: first image and captions")
    print("Image 0:", image_filenames[0])
    first_caption_indices = torch.where(caption_to_image == 0)[0].tolist()
    for idx in first_caption_indices:
        print(f"caption[{idx}]: {captions[idx]}")
    print("=" * 80)

    return image_paths, captions, caption_to_image, image_filenames


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


# =========================
# Feature extraction
# =========================

@torch.no_grad()
def encode_images(
    model,
    preprocess,
    image_paths: List[str],
    batch_size: int,
    num_workers: int,
    device: str,
    amp: bool
) -> torch.Tensor:
    dataset = ImageOnlyDataset(image_paths, preprocess)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    features_list = []
    model.eval()

    for images in tqdm(loader, desc="Encoding images"):
        images = images.to(device, non_blocking=True)

        if amp and device.startswith("cuda"):
            with torch.cuda.amp.autocast():
                feats = model.encode_image(images)
        else:
            feats = model.encode_image(images)

        feats = feats / feats.norm(dim=-1, keepdim=True)
        features_list.append(feats.cpu())

    image_features = torch.cat(features_list, dim=0)
    return image_features


@torch.no_grad()
def encode_texts(
    model,
    tokenizer,
    captions: List[str],
    batch_size: int,
    device: str,
    amp: bool
) -> torch.Tensor:
    features_list = []
    model.eval()

    for start in tqdm(range(0, len(captions), batch_size), desc="Encoding texts"):
        batch_caps = captions[start:start + batch_size]
        tokens = tokenizer(batch_caps).to(device)

        if amp and device.startswith("cuda"):
            with torch.cuda.amp.autocast():
                feats = model.encode_text(tokens)
        else:
            feats = model.encode_text(tokens)

        feats = feats / feats.norm(dim=-1, keepdim=True)
        features_list.append(feats.cpu())

    text_features = torch.cat(features_list, dim=0)
    return text_features


# =========================
# Evaluation
# =========================

@torch.no_grad()
def evaluate_t2i(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    caption_to_image: torch.LongTensor,
    ks=(1, 5, 10),
    device: str = "cuda",
    chunk_size: int = 1024
) -> Dict[str, float]:
    """
    Text-to-Image:
    每条 caption 检索所有 image。
    如果对应 image 出现在 Top-K，算命中。
    """
    image_features = image_features.to(device)
    text_features = text_features.to(device)
    caption_to_image = caption_to_image.to(device)

    num_captions = text_features.size(0)
    max_k = max(ks)

    correct = {k: 0 for k in ks}

    for start in tqdm(range(0, num_captions, chunk_size), desc="Evaluating T2I"):
        end = min(start + chunk_size, num_captions)

        sims = text_features[start:end] @ image_features.T
        topk = sims.topk(k=max_k, dim=1).indices

        gt = caption_to_image[start:end].unsqueeze(1)
        match = topk.eq(gt)

        for k in ks:
            correct[k] += match[:, :k].any(dim=1).sum().item()

    results = {}
    for k in ks:
        results[f"t2i_R@{k}"] = correct[k] / num_captions * 100.0

    return results


@torch.no_grad()
def evaluate_i2t(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    caption_to_image: torch.LongTensor,
    ks=(1, 5, 10),
    device: str = "cuda",
    chunk_size: int = 512
) -> Dict[str, float]:
    """
    Image-to-Text:
    每张 image 检索所有 caption。
    如果该 image 对应的任意 caption 出现在 Top-K，算命中。
    """
    image_features = image_features.to(device)
    text_features = text_features.to(device)
    caption_to_image = caption_to_image.to(device)

    num_images = image_features.size(0)
    max_k = max(ks)

    correct = {k: 0 for k in ks}

    for start in tqdm(range(0, num_images, chunk_size), desc="Evaluating I2T"):
        end = min(start + chunk_size, num_images)

        sims = image_features[start:end] @ text_features.T
        topk_caption_indices = sims.topk(k=max_k, dim=1).indices

        # topk_caption_indices: [B, K]
        # caption_to_image[topk_caption_indices]: [B, K]
        # 如果其中等于当前 image index，就说明命中。
        topk_image_indices = caption_to_image[topk_caption_indices]
        gt_image_indices = torch.arange(start, end, device=device).unsqueeze(1)

        match = topk_image_indices.eq(gt_image_indices)

        for k in ks:
            correct[k] += match[:, :k].any(dim=1).sum().item()

    results = {}
    for k in ks:
        results[f"i2t_R@{k}"] = correct[k] / num_images * 100.0

    return results


def print_results(results: Dict[str, float]):
    print("\n" + "=" * 80)
    print("CLIP Zero-shot Baseline Results")
    print("=" * 80)

    print("Text-to-Image Retrieval")
    print(f"R@1 :  {results['t2i_R@1']:.2f}%")
    print(f"R@5 :  {results['t2i_R@5']:.2f}%")
    print(f"R@10:  {results['t2i_R@10']:.2f}%")

    print("\nImage-to-Text Retrieval")
    print(f"R@1 :  {results['i2t_R@1']:.2f}%")
    print(f"R@5 :  {results['i2t_R@5']:.2f}%")
    print(f"R@10:  {results['i2t_R@10']:.2f}%")

    print("=" * 80)


def save_results(
    results: Dict[str, float],
    save_dir: str,
    split: str,
    model_name: str,
    pretrained: str,
    num_images: int,
    num_captions: int
):
    os.makedirs(save_dir, exist_ok=True)

    row = {
        "method": "CLIP Baseline",
        "split": split,
        "model_name": model_name,
        "pretrained": pretrained,
        "num_images": num_images,
        "num_captions": num_captions,
        **results
    }

    df = pd.DataFrame([row])
    save_path = os.path.join(save_dir, f"baseline_{split}_{model_name}_{pretrained}.csv")
    df.to_csv(save_path, index=False)

    print(f"\nSaved results to: {save_path}")


# =========================
# Main
# =========================

def main():
    # 直接硬编码参数，无需命令行
    args = types.SimpleNamespace(
        csv_path="/root/autodl-tmp/flickr_annotations_30k.csv",
        img_dir="/root/autodl-tmp/flickr30k-images",
        split="test",
        model_name="ViT-B-32",
        pretrained="laion2b_s34b_b79k",
        image_batch_size=64,
        text_batch_size=128,
        num_workers=4,
        device="cuda" if torch.cuda.is_available() else "cpu",
        amp=True,
        save_dir="./outputs_baseline",
        skip_missing=False
    )

    print("=" * 80)
    print("CLIP Image-Text Retrieval Baseline")
    print("=" * 80)
    print(f"CSV path: {args.csv_path}")
    print(f"Image dir: {args.img_dir}")
    print(f"Split: {args.split}")
    print(f"Model: {args.model_name}")
    print(f"Pretrained: {args.pretrained}")
    print(f"Device: {args.device}")
    print(f"AMP: {args.amp}")
    print("=" * 80)

    image_paths, captions, caption_to_image, image_filenames = build_eval_data(
        csv_path=args.csv_path,
        img_dir=args.img_dir,
        split=args.split,
        skip_missing=args.skip_missing
    )

    print("\nLoading OpenCLIP model...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model_name,
        pretrained=args.pretrained,
        device=args.device
    )
    tokenizer = open_clip.get_tokenizer(args.model_name)
    model.eval()

    print("\nExtracting image features...")
    image_features = encode_images(
        model=model,
        preprocess=preprocess,
        image_paths=image_paths,
        batch_size=args.image_batch_size,
        num_workers=args.num_workers,
        device=args.device,
        amp=args.amp
    )

    print("\nExtracting text features...")
    text_features = encode_texts(
        model=model,
        tokenizer=tokenizer,
        captions=captions,
        batch_size=args.text_batch_size,
        device=args.device,
        amp=args.amp
    )

    print("\nFeature shapes:")
    print(f"image_features: {tuple(image_features.shape)}")
    print(f"text_features:  {tuple(text_features.shape)}")
    print(f"caption_to_image: {tuple(caption_to_image.shape)}")

    if len(captions) != len(caption_to_image):
        raise RuntimeError("len(captions) must equal len(caption_to_image).")

    if image_features.size(0) != len(image_paths):
        raise RuntimeError("image_features number does not match image_paths.")

    if text_features.size(0) != len(captions):
        raise RuntimeError("text_features number does not match captions.")

    print("\nEvaluating...")
    t2i_results = evaluate_t2i(
        image_features=image_features,
        text_features=text_features,
        caption_to_image=caption_to_image,
        ks=(1, 5, 10),
        device=args.device
    )

    i2t_results = evaluate_i2t(
        image_features=image_features,
        text_features=text_features,
        caption_to_image=caption_to_image,
        ks=(1, 5, 10),
        device=args.device
    )

    results = {}
    results.update(t2i_results)
    results.update(i2t_results)

    print_results(results)

    save_results(
        results=results,
        save_dir=args.save_dir,
        split=args.split,
        model_name=args.model_name,
        pretrained=args.pretrained,
        num_images=len(image_paths),
        num_captions=len(captions)
    )


if __name__ == "__main__":
    main()


# In[ ]:




