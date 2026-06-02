#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import ast
import json
import math
import random
import types
from pathlib import Path
from typing import List, Tuple, Dict

import pandas as pd
from PIL import Image, ImageFile
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import open_clip


ImageFile.LOAD_TRUNCATED_IMAGES = True


# =========================
# Reproducibility
# =========================

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# Data utils
# =========================

def parse_caption_list(raw_value) -> List[str]:
    """
    raw 字段格式：
    ["caption1", "caption2", "caption3", "caption4", "caption5"]
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


def build_rows(
    csv_path: str,
    img_dir: str,
    split: str,
    skip_missing: bool = False
):
    """
    每一行是一张图片和它的 5 条 captions。
    用于 train / val / test 的统一构建。
    """
    df = pd.read_csv(csv_path)

    df = df[df["split"] == split].reset_index(drop=True)

    if len(df) == 0:
        raise ValueError(f"No data found for split={split}")

    img_dir = Path(img_dir)
    rows = []

    for _, row in df.iterrows():
        filename = str(row["filename"])
        img_path = img_dir / filename

        if not img_path.exists():
            if skip_missing:
                print(f"[Warning] Missing image skipped: {img_path}")
                continue
            raise FileNotFoundError(
                f"Image not found: {img_path}\n"
                f"Please check img_dir."
            )

        captions = parse_caption_list(row["raw"])

        rows.append({
            "filename": filename,
            "image_path": str(img_path),
            "captions": captions,
        })

    return rows


class FlickrTrainDataset(Dataset):
    """
    训练集 Dataset。

    注意：
    - len(dataset) = 训练图片数量，比如 29000
    - 每次 __getitem__ 返回一张图片 + 这张图片随机选的一条 caption
    - 这样每个 batch 里一般不会出现同一张图重复多次，避免 false negative 更严重
    """
    def __init__(self, csv_path: str, img_dir: str, split: str, transform, skip_missing=False):
        assert split == "train", "FlickrTrainDataset should only use split='train'."

        self.rows = build_rows(
            csv_path=csv_path,
            img_dir=img_dir,
            split=split,
            skip_missing=skip_missing
        )
        self.transform = transform

        print("=" * 80)
        print("Train Dataset")
        print("=" * 80)
        print(f"Split: {split}")
        print(f"Num train images: {len(self.rows)}")
        print("Each image randomly samples one caption per epoch/iteration.")
        print("=" * 80)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        image = self.transform(image)

        caption = random.choice(row["captions"])

        return image, caption


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


def build_eval_data(
    csv_path: str,
    img_dir: str,
    split: str,
    skip_missing: bool = False
) -> Tuple[List[str], List[str], torch.LongTensor, List[str]]:
    """
    用于 val / test 评估。

    返回：
    image_paths:      每张唯一图片路径，长度 N
    captions:         所有 caption，长度 5N
    caption_to_image: 第 j 条 caption 对应第几张 image，长度 5N
    image_filenames:  图片文件名，长度 N
    """
    assert split in ["val", "test"], "Evaluation split should be 'val' or 'test'."

    rows = build_rows(
        csv_path=csv_path,
        img_dir=img_dir,
        split=split,
        skip_missing=skip_missing
    )

    image_paths = []
    image_filenames = []
    captions = []
    caption_to_image = []

    for image_idx, row in enumerate(rows):
        image_paths.append(row["image_path"])
        image_filenames.append(row["filename"])

        for cap in row["captions"]:
            captions.append(cap)
            caption_to_image.append(image_idx)

    caption_to_image = torch.LongTensor(caption_to_image)

    print("=" * 80)
    print("Eval Dataset")
    print("=" * 80)
    print(f"Split: {split}")
    print(f"Num images: {len(image_paths)}")
    print(f"Num captions: {len(captions)}")
    print(f"Captions per image: {len(captions) / max(len(image_paths), 1):.2f}")
    print("=" * 80)

    print("Sanity check for eval data:")
    print("Image 0:", image_filenames[0])
    first_caption_indices = torch.where(caption_to_image == 0)[0].tolist()
    for idx in first_caption_indices:
        print(f"caption[{idx}]: {captions[idx]}")
    print("=" * 80)

    return image_paths, captions, caption_to_image, image_filenames


# =========================
# Model
# =========================

class ResidualAdapter(nn.Module):
    """
    小型残差 Adapter。

    初始化时最后一层为 0，所以刚开始：
    output ≈ input

    这样不会一上来破坏 CLIP 原本的 embedding 空间。
    """
    def __init__(self, embed_dim: int = 512, bottleneck_dim: int = 128, residual_scale: float = 0.2):
        super().__init__()

        self.residual_scale = residual_scale

        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, embed_dim),
        )

        # 关键：零初始化最后一层，让 finetune 初始状态接近 baseline
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return x + self.residual_scale * self.net(x)


class FinetuneCLIP(nn.Module):
    """
    冻结 CLIP 主干，只训练 image/text adapter。
    """
    def __init__(
        self,
        clip_model,
        embed_dim: int = 512,
        bottleneck_dim: int = 128,
        residual_scale: float = 0.2,
    ):
        super().__init__()

        self.clip = clip_model

        # 冻结 CLIP 主干
        for p in self.clip.parameters():
            p.requires_grad = False

        self.image_adapter = ResidualAdapter(
            embed_dim=embed_dim,
            bottleneck_dim=bottleneck_dim,
            residual_scale=residual_scale
        )

        self.text_adapter = ResidualAdapter(
            embed_dim=embed_dim,
            bottleneck_dim=bottleneck_dim,
            residual_scale=residual_scale
        )

    def encode_image(self, images):
        with torch.no_grad():
            image_features = self.clip.encode_image(images)

        image_features = image_features.float()
        image_features = self.image_adapter(image_features)
        image_features = F.normalize(image_features, dim=-1)

        return image_features

    def encode_text(self, texts):
        with torch.no_grad():
            text_features = self.clip.encode_text(texts)

        text_features = text_features.float()
        text_features = self.text_adapter(text_features)
        text_features = F.normalize(text_features, dim=-1)

        return text_features

    def forward(self, images, texts):
        image_features = self.encode_image(images)
        text_features = self.encode_text(texts)
        return image_features, text_features


def contrastive_loss(image_features, text_features, logit_scale: float = 100.0):
    """
    CLIP-style bidirectional InfoNCE loss.

    image_features: [B, D]
    text_features:  [B, D]
    """
    logits = logit_scale * image_features @ text_features.T
    labels = torch.arange(logits.size(0), device=logits.device)

    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)

    loss = (loss_i2t + loss_t2i) / 2.0

    return loss


# =========================
# Feature extraction for eval
# =========================

@torch.no_grad()
def encode_eval_images(
    model: FinetuneCLIP,
    preprocess,
    image_paths: List[str],
    batch_size: int,
    num_workers: int,
    device: str,
    amp: bool,
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

    for images in tqdm(loader, desc="Encoding val images"):
        images = images.to(device, non_blocking=True)

        if amp and device.startswith("cuda"):
            with torch.cuda.amp.autocast():
                feats = model.encode_image(images)
        else:
            feats = model.encode_image(images)

        features_list.append(feats.cpu())

    return torch.cat(features_list, dim=0)


@torch.no_grad()
def encode_eval_texts(
    model: FinetuneCLIP,
    tokenizer,
    captions: List[str],
    batch_size: int,
    device: str,
    amp: bool,
) -> torch.Tensor:
    features_list = []
    model.eval()

    for start in tqdm(range(0, len(captions), batch_size), desc="Encoding val texts"):
        batch_caps = captions[start:start + batch_size]
        tokens = tokenizer(batch_caps).to(device)

        if amp and device.startswith("cuda"):
            with torch.cuda.amp.autocast():
                feats = model.encode_text(tokens)
        else:
            feats = model.encode_text(tokens)

        features_list.append(feats.cpu())

    return torch.cat(features_list, dim=0)


# =========================
# Recall evaluation
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
    image_features = image_features.to(device)
    text_features = text_features.to(device)
    caption_to_image = caption_to_image.to(device)

    num_captions = text_features.size(0)
    max_k = max(ks)

    correct = {k: 0 for k in ks}

    for start in tqdm(range(0, num_captions, chunk_size), desc="Evaluating val T2I"):
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
    image_features = image_features.to(device)
    text_features = text_features.to(device)
    caption_to_image = caption_to_image.to(device)

    num_images = image_features.size(0)
    max_k = max(ks)

    correct = {k: 0 for k in ks}

    for start in tqdm(range(0, num_images, chunk_size), desc="Evaluating val I2T"):
        end = min(start + chunk_size, num_images)

        sims = image_features[start:end] @ text_features.T
        topk_caption_indices = sims.topk(k=max_k, dim=1).indices

        topk_image_indices = caption_to_image[topk_caption_indices]
        gt_image_indices = torch.arange(start, end, device=device).unsqueeze(1)

        match = topk_image_indices.eq(gt_image_indices)

        for k in ks:
            correct[k] += match[:, :k].any(dim=1).sum().item()

    results = {}
    for k in ks:
        results[f"i2t_R@{k}"] = correct[k] / num_images * 100.0

    return results


@torch.no_grad()
def evaluate_retrieval(
    model: FinetuneCLIP,
    tokenizer,
    preprocess,
    image_paths: List[str],
    captions: List[str],
    caption_to_image: torch.LongTensor,
    args,
) -> Dict[str, float]:
    image_features = encode_eval_images(
        model=model,
        preprocess=preprocess,
        image_paths=image_paths,
        batch_size=args.eval_image_batch_size,
        num_workers=args.num_workers,
        device=args.device,
        amp=args.amp,
    )

    text_features = encode_eval_texts(
        model=model,
        tokenizer=tokenizer,
        captions=captions,
        batch_size=args.eval_text_batch_size,
        device=args.device,
        amp=args.amp,
    )

    print("Eval feature shapes:")
    print(f"image_features: {tuple(image_features.shape)}")
    print(f"text_features:  {tuple(text_features.shape)}")

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

    return results


def mean_recall(results: Dict[str, float]) -> float:
    keys = [
        "t2i_R@1", "t2i_R@5", "t2i_R@10",
        "i2t_R@1", "i2t_R@5", "i2t_R@10",
    ]
    return sum(results[k] for k in keys) / len(keys)


def print_eval_results(results: Dict[str, float], title: str = "Validation Results"):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    print("Text-to-Image Retrieval")
    print(f"R@1 :  {results['t2i_R@1']:.2f}%")
    print(f"R@5 :  {results['t2i_R@5']:.2f}%")
    print(f"R@10:  {results['t2i_R@10']:.2f}%")

    print("\nImage-to-Text Retrieval")
    print(f"R@1 :  {results['i2t_R@1']:.2f}%")
    print(f"R@5 :  {results['i2t_R@5']:.2f}%")
    print(f"R@10:  {results['i2t_R@10']:.2f}%")

    print(f"\nMean Recall: {mean_recall(results):.2f}%")
    print("=" * 80)


# =========================
# LR scheduler
# =========================

def build_warmup_cosine_scheduler(optimizer, total_steps: int, warmup_steps: int):
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))

        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# =========================
# Save
# =========================

def save_checkpoint(
    model: FinetuneCLIP,
    optimizer,
    epoch: int,
    best_score: float,
    val_results: Dict[str, float],
    args,
    save_path: str,
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "best_score": best_score,
        "val_results": val_results,

        # 只保存 adapter，比保存整个 CLIP 小很多
        "image_adapter": model.image_adapter.state_dict(),
        "text_adapter": model.text_adapter.state_dict(),

        "optimizer": optimizer.state_dict(),

        "config": {
            "model_name": args.model_name,
            "pretrained": args.pretrained,
            "embed_dim": args.embed_dim,
            "bottleneck_dim": args.bottleneck_dim,
            "residual_scale": args.residual_scale,
            "train_split": args.train_split,
            "val_split": args.val_split,
        }
    }

    torch.save(checkpoint, save_path)
    print(f"Saved best checkpoint to: {save_path}")


def save_val_log(log_rows: List[Dict], save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df = pd.DataFrame(log_rows)
    df.to_csv(save_path, index=False)
    print(f"Saved val log to: {save_path}")


# =========================
# Train
# =========================

def train_one_epoch(
    model: FinetuneCLIP,
    tokenizer,
    train_loader,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    args,
):
    model.train()

    total_loss = 0.0
    total_samples = 0

    progress = tqdm(train_loader, desc=f"Training Epoch {epoch}")

    for images, captions in progress:
        images = images.to(args.device, non_blocking=True)
        texts = tokenizer(list(captions)).to(args.device)

        optimizer.zero_grad(set_to_none=True)

        if args.amp and args.device.startswith("cuda"):
            with torch.cuda.amp.autocast():
                image_features, text_features = model(images, texts)
                loss = contrastive_loss(
                    image_features=image_features,
                    text_features=text_features,
                    logit_scale=args.logit_scale
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            image_features, text_features = model(images, texts)
            loss = contrastive_loss(
                image_features=image_features,
                text_features=text_features,
                logit_scale=args.logit_scale
            )

            loss.backward()
            optimizer.step()

        scheduler.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        progress.set_postfix({
            "loss": f"{loss.item():.4f}",
            "lr": f"{scheduler.get_last_lr()[0]:.2e}"
        })

    avg_loss = total_loss / max(1, total_samples)

    return avg_loss


# =========================
# Main
# =========================

def main():
    args = types.SimpleNamespace(
        # 路径
        csv_path="/root/autodl-tmp/flickr_annotations_30k.csv",
        img_dir="/root/autodl-tmp/flickr30k-images",

        # 重点：训练集和验证集严格分开
        train_split="train",
        val_split="val",

        # 模型
        model_name="ViT-B-32",
        pretrained="/root/autodl-tmp/model/open_clip_model.safetensors",
        embed_dim=512,

        # Adapter 参数
        bottleneck_dim=128,
        residual_scale=0.2,

        # 训练参数
        epochs=10,
        train_batch_size=128,
        lr=1e-4,
        weight_decay=1e-2,
        warmup_ratio=0.05,
        logit_scale=100.0,

        # 验证参数
        eval_image_batch_size=64,
        eval_text_batch_size=128,

        # 系统参数
        num_workers=4,
        device="cuda" if torch.cuda.is_available() else "cpu",
        amp=True,
        seed=42,
        skip_missing=False,

        # 输出
        output_dir="./outputs_finetune",
        best_ckpt_name="best_finetune_adapter.pt",
        val_log_name="val_log.csv",
    )

    set_seed(args.seed)

    # 防止误用
    assert args.train_split == "train", "train_split must be 'train'."
    assert args.val_split == "val", "val_split must be 'val'."

    print("=" * 80)
    print("CLIP Finetune Training")
    print("=" * 80)
    print(f"CSV path: {args.csv_path}")
    print(f"Image dir: {args.img_dir}")
    print(f"Train split: {args.train_split}")
    print(f"Val split: {args.val_split}")
    print("Test split is NOT used in this training script.")
    print(f"Model: {args.model_name}")
    print(f"Pretrained: {args.pretrained}")
    print(f"Device: {args.device}")
    print(f"AMP: {args.amp}")
    print("=" * 80)

    print("\nLoading OpenCLIP model...")
    clip_model, train_preprocess, val_preprocess = open_clip.create_model_and_transforms(
        args.model_name,
        pretrained=args.pretrained,
        device=args.device
    )
    tokenizer = open_clip.get_tokenizer(args.model_name)

    model = FinetuneCLIP(
        clip_model=clip_model,
        embed_dim=args.embed_dim,
        bottleneck_dim=args.bottleneck_dim,
        residual_scale=args.residual_scale,
    ).to(args.device)

    train_dataset = FlickrTrainDataset(
        csv_path=args.csv_path,
        img_dir=args.img_dir,
        split=args.train_split,
        transform=train_preprocess,
        skip_missing=args.skip_missing,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # 验证集只在这里构建，不参与训练
    val_image_paths, val_captions, val_caption_to_image, _ = build_eval_data(
        csv_path=args.csv_path,
        img_dir=args.img_dir,
        split=args.val_split,
        skip_missing=args.skip_missing,
    )

    trainable_params = (
        list(model.image_adapter.parameters()) +
        list(model.text_adapter.parameters())
    )

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(args.warmup_ratio * total_steps)

    scheduler = build_warmup_cosine_scheduler(
        optimizer=optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps
    )

    scaler = torch.cuda.amp.GradScaler(
        enabled=args.amp and args.device.startswith("cuda")
    )

    best_score = -1.0
    log_rows = []

    best_ckpt_path = os.path.join(args.output_dir, args.best_ckpt_name)
    val_log_path = os.path.join(args.output_dir, args.val_log_name)

    print("\nStart training...")
    print(f"Total train steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps}")

    for epoch in range(1, args.epochs + 1):
        avg_loss = train_one_epoch(
            model=model,
            tokenizer=tokenizer,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            args=args,
        )

        print(f"\nEpoch {epoch} finished. Train loss: {avg_loss:.4f}")

        # 每个 epoch 只在 val 上评估
        val_results = evaluate_retrieval(
            model=model,
            tokenizer=tokenizer,
            preprocess=val_preprocess,
            image_paths=val_image_paths,
            captions=val_captions,
            caption_to_image=val_caption_to_image,
            args=args,
        )

        val_score = mean_recall(val_results)

        print_eval_results(
            val_results,
            title=f"Validation Results - Epoch {epoch}"
        )

        row = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "val_mean_recall": val_score,
            **val_results
        }
        log_rows.append(row)

        save_val_log(log_rows, val_log_path)

        if val_score > best_score:
            best_score = val_score

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_score=best_score,
                val_results=val_results,
                args=args,
                save_path=best_ckpt_path,
            )

            print(f"New best val mean recall: {best_score:.2f}%")
        else:
            print(f"No improvement. Best val mean recall: {best_score:.2f}%")

    print("\n" + "=" * 80)
    print("Training finished.")
    print(f"Best val mean recall: {best_score:.2f}%")
    print(f"Best checkpoint: {best_ckpt_path}")
    print("Reminder: test split was NOT used. Evaluate test only after selecting best checkpoint.")
    print("=" * 80)


if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:




