#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
# Seed
# =========================

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# Data
# =========================

def parse_caption_list(raw_value) -> List[str]:
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


def build_retrieval_data(
    csv_path: str,
    img_dir: str,
    split: str,
    skip_missing: bool = False
):
    """
    可用于 train/val/test 的检索格式数据。

    返回：
    rows:             每张图一行，含 image_path 和 captions
    image_paths:      唯一图片路径，长度 N
    captions:         所有 caption，长度约 5N
    caption_to_image: 每条 caption 对应的 image index
    image_to_captions: 每张 image 对应的 caption indices
    """
    rows = build_rows(
        csv_path=csv_path,
        img_dir=img_dir,
        split=split,
        skip_missing=skip_missing
    )

    image_paths = []
    captions = []
    caption_to_image = []
    image_to_captions = []

    cap_idx = 0

    for image_idx, row in enumerate(rows):
        image_paths.append(row["image_path"])

        cur_cap_indices = []
        for cap in row["captions"]:
            captions.append(cap)
            caption_to_image.append(image_idx)
            cur_cap_indices.append(cap_idx)
            cap_idx += 1

        image_to_captions.append(cur_cap_indices)

    caption_to_image = torch.LongTensor(caption_to_image)

    print("=" * 80)
    print("Retrieval Data")
    print("=" * 80)
    print(f"Split: {split}")
    print(f"Num images: {len(image_paths)}")
    print(f"Num captions: {len(captions)}")
    print(f"Captions per image: {len(captions) / max(len(image_paths), 1):.2f}")
    print("=" * 80)

    print("Sanity check:")
    print("Image 0:", Path(image_paths[0]).name)
    for idx in image_to_captions[0]:
        print(f"caption[{idx}]: {captions[idx]}")
    print("=" * 80)

    return rows, image_paths, captions, caption_to_image, image_to_captions


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


class HardNegativeTrainDataset(Dataset):
    """
    每个样本返回：
    - anchor image
    - positive caption
    - hard negative caption
    - hard negative image

    训练只用 train split。
    """
    def __init__(
        self,
        rows,
        captions: List[str],
        image_to_captions: List[List[int]],
        image_to_hard_captions: torch.LongTensor,
        caption_to_hard_images: torch.LongTensor,
        transform,
    ):
        self.rows = rows
        self.captions = captions
        self.image_to_captions = image_to_captions
        self.image_to_hard_captions = image_to_hard_captions
        self.caption_to_hard_images = caption_to_hard_images
        self.transform = transform

        print("=" * 80)
        print("Hard Negative Train Dataset")
        print("=" * 80)
        print(f"Num train images: {len(self.rows)}")
        print(f"Num captions: {len(self.captions)}")
        print(f"image_to_hard_captions shape: {tuple(self.image_to_hard_captions.shape)}")
        print(f"caption_to_hard_images shape: {tuple(self.caption_to_hard_images.shape)}")
        print("=" * 80)

    def __len__(self):
        return len(self.rows)

    def _load_image(self, image_idx):
        image_path = self.rows[image_idx]["image_path"]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image

    def __getitem__(self, image_idx):
        # anchor image
        image = self._load_image(image_idx)

        # positive caption：从当前图片 5 条 caption 中随机选一条
        pos_caption_idx = random.choice(self.image_to_captions[image_idx])
        pos_caption = self.captions[pos_caption_idx]

        # hard negative caption：和当前图片很像，但不是当前图片的 caption
        hn_caption_pool = self.image_to_hard_captions[image_idx]
        hn_caption_idx = int(hn_caption_pool[random.randint(0, len(hn_caption_pool) - 1)].item())
        hn_caption = self.captions[hn_caption_idx]

        # hard negative image：和当前 positive caption 很像，但不是对应图片
        hn_image_pool = self.caption_to_hard_images[pos_caption_idx]
        hn_image_idx = int(hn_image_pool[random.randint(0, len(hn_image_pool) - 1)].item())
        hn_image = self._load_image(hn_image_idx)

        return image, pos_caption, hn_caption, hn_image


# =========================
# LoRA modules
# =========================

class LoRAProjection(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.05
    ):
        super().__init__()

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        self.dropout = nn.Dropout(dropout)
        self.down = nn.Linear(in_features, r, bias=False)
        self.up = nn.Linear(r, out_features, bias=False)

        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return self.up(self.down(self.dropout(x))) * self.scaling


class LoRAMultiheadAttention(nn.Module):
    def __init__(
        self,
        base_attn: nn.MultiheadAttention,
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.05,
        lora_on_k: bool = False,
    ):
        super().__init__()

        self.base_attn = base_attn

        for p in self.base_attn.parameters():
            p.requires_grad = False

        self.embed_dim = base_attn.embed_dim
        self.num_heads = base_attn.num_heads
        self.dropout_p = base_attn.dropout
        self.batch_first = base_attn.batch_first
        self.head_dim = self.embed_dim // self.num_heads
        self.lora_on_k = lora_on_k

        assert self.head_dim * self.num_heads == self.embed_dim
        assert base_attn._qkv_same_embed_dim

        self.lora_q = LoRAProjection(
            self.embed_dim, self.embed_dim, r=r, alpha=alpha, dropout=dropout
        )
        self.lora_v = LoRAProjection(
            self.embed_dim, self.embed_dim, r=r, alpha=alpha, dropout=dropout
        )
        self.lora_out = LoRAProjection(
            self.embed_dim, self.embed_dim, r=r, alpha=alpha, dropout=dropout
        )

        if lora_on_k:
            self.lora_k = LoRAProjection(
                self.embed_dim, self.embed_dim, r=r, alpha=alpha, dropout=dropout
            )
        else:
            self.lora_k = None

    def _shape(self, x, batch_size, seq_len):
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=None,
        average_attn_weights=True,
        is_causal=False,
    ):
        original_dtype = query.dtype

        if self.batch_first:
            q_input = query
            k_input = key
            v_input = value
        else:
            q_input = query.transpose(0, 1)
            k_input = key.transpose(0, 1)
            v_input = value.transpose(0, 1)

        batch_size, tgt_len, _ = q_input.shape
        src_len = k_input.shape[1]

        in_proj_weight = self.base_attn.in_proj_weight
        in_proj_bias = self.base_attn.in_proj_bias

        w_q, w_k, w_v = in_proj_weight.chunk(3, dim=0)

        if in_proj_bias is not None:
            b_q, b_k, b_v = in_proj_bias.chunk(3, dim=0)
        else:
            b_q = b_k = b_v = None

        q = F.linear(q_input, w_q, b_q) + self.lora_q(q_input)

        k = F.linear(k_input, w_k, b_k)
        if self.lora_k is not None:
            k = k + self.lora_k(k_input)

        v = F.linear(v_input, w_v, b_v) + self.lora_v(v_input)

        q = self._shape(q, batch_size, tgt_len)
        k = self._shape(k, batch_size, src_len)
        v = self._shape(v, batch_size, src_len)

        q = q * (self.head_dim ** -0.5)

        attn_scores = torch.matmul(q, k.transpose(-2, -1))

        if attn_mask is not None:
            attn_mask = attn_mask.to(device=attn_scores.device)

            if attn_mask.dtype == torch.bool:
                if attn_mask.dim() == 2:
                    attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
                elif attn_mask.dim() == 3:
                    attn_mask = attn_mask.view(batch_size, self.num_heads, tgt_len, src_len)
                attn_scores = attn_scores.masked_fill(attn_mask, float("-inf"))
            else:
                attn_mask = attn_mask.to(dtype=attn_scores.dtype)
                if attn_mask.dim() == 2:
                    attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
                elif attn_mask.dim() == 3:
                    attn_mask = attn_mask.view(batch_size, self.num_heads, tgt_len, src_len)
                attn_scores = attn_scores + attn_mask

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.to(device=attn_scores.device)

            if key_padding_mask.dtype == torch.bool:
                mask = key_padding_mask.view(batch_size, 1, 1, src_len)
                attn_scores = attn_scores.masked_fill(mask, float("-inf"))
            else:
                mask = key_padding_mask.view(batch_size, 1, 1, src_len).to(dtype=attn_scores.dtype)
                attn_scores = attn_scores + mask

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = F.dropout(attn_probs, p=self.dropout_p, training=self.training)

        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, tgt_len, self.embed_dim
        )

        attn_output = (
            F.linear(
                attn_output,
                self.base_attn.out_proj.weight,
                self.base_attn.out_proj.bias
            )
            + self.lora_out(attn_output)
        )

        attn_output = attn_output.to(original_dtype)

        if not self.batch_first:
            attn_output = attn_output.transpose(0, 1)

        if need_weights:
            weights = attn_probs
            if average_attn_weights:
                weights = weights.mean(dim=1)
            return attn_output, weights

        return attn_output, None


def freeze_all_params(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = False


def inject_lora_to_attention(
    module: nn.Module,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    lora_on_k: bool = False,
):
    num_replaced = 0

    for name, child in list(module.named_children()):
        if isinstance(child, nn.MultiheadAttention):
            setattr(
                module,
                name,
                LoRAMultiheadAttention(
                    base_attn=child,
                    r=r,
                    alpha=alpha,
                    dropout=dropout,
                    lora_on_k=lora_on_k,
                )
            )
            num_replaced += 1
        else:
            num_replaced += inject_lora_to_attention(
                child,
                r=r,
                alpha=alpha,
                dropout=dropout,
                lora_on_k=lora_on_k,
            )

    return num_replaced


def get_trainable_params(model: nn.Module):
    return [p for p in model.parameters() if p.requires_grad]


def count_params(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_lora_state_dict(model: nn.Module):
    state = {}
    for k, v in model.state_dict().items():
        if "lora_" in k:
            state[k] = v.cpu()
    return state


def load_lora_model_from_checkpoint(args):
    """
    加载 best_lora_clip.pt，作为 hard negative 训练的起点。
    """
    checkpoint = torch.load(args.init_lora_ckpt, map_location="cpu")
    config = checkpoint.get("config", {})

    model_name = config.get("model_name", args.model_name)
    pretrained = config.get("pretrained", args.pretrained)

    lora_r = config.get("lora_r", args.lora_r)
    lora_alpha = config.get("lora_alpha", args.lora_alpha)
    lora_dropout = config.get("lora_dropout", args.lora_dropout)
    lora_on_k = config.get("lora_on_k", args.lora_on_k)

    print("=" * 80)
    print("Loading Initial LoRA Checkpoint")
    print("=" * 80)
    print(f"Init checkpoint: {args.init_lora_ckpt}")
    print(f"Model name: {model_name}")
    print(f"Pretrained: {pretrained}")
    print(f"LoRA r: {lora_r}")
    print(f"LoRA alpha: {lora_alpha}")
    print(f"LoRA dropout: {lora_dropout}")
    print(f"LoRA on k: {lora_on_k}")
    print("=" * 80)

    model, train_preprocess, val_preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        device=args.device,
    )
    tokenizer = open_clip.get_tokenizer(model_name)

    freeze_all_params(model)

    num_replaced = inject_lora_to_attention(
        model,
        r=lora_r,
        alpha=lora_alpha,
        dropout=lora_dropout,
        lora_on_k=lora_on_k,
    )

    if num_replaced == 0:
        raise RuntimeError("No nn.MultiheadAttention modules were found.")

    # 关键：注入的新 LoRA 模块默认在 CPU，需要重新搬到 GPU
    model = model.to(args.device)

    lora_state_dict = checkpoint["lora_state_dict"]

    model_state = model.state_dict()
    missing_keys = [k for k in lora_state_dict.keys() if k not in model_state]
    if len(missing_keys) > 0:
        raise RuntimeError(f"Some LoRA keys are not found in model: {missing_keys[:10]}")

    model.load_state_dict(lora_state_dict, strict=False)

    total_params, trainable_params = count_params(model)

    print(f"Injected LoRA into {num_replaced} attention modules.")
    print(f"Loaded LoRA tensors: {len(lora_state_dict)}")
    print(f"Total params: {total_params / 1e6:.2f}M")
    print(f"Trainable params: {trainable_params / 1e6:.4f}M")
    print(f"Trainable ratio: {trainable_params / total_params * 100:.4f}%")
    print("=" * 80)

    return model, tokenizer, train_preprocess, val_preprocess, {
        "model_name": model_name,
        "pretrained": pretrained,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "lora_on_k": lora_on_k,
    }


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
    amp: bool,
    desc: str = "Encoding images",
) -> torch.Tensor:
    dataset = ImageOnlyDataset(image_paths, preprocess)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
    )

    features_list = []
    model.eval()

    for images in tqdm(loader, desc=desc):
        images = images.to(device, non_blocking=True)

        if amp and device.startswith("cuda"):
            with torch.cuda.amp.autocast():
                feats = model.encode_image(images)
        else:
            feats = model.encode_image(images)

        feats = F.normalize(feats.float(), dim=-1)
        features_list.append(feats.cpu())

    return torch.cat(features_list, dim=0)


@torch.no_grad()
def encode_texts(
    model,
    tokenizer,
    captions: List[str],
    batch_size: int,
    device: str,
    amp: bool,
    desc: str = "Encoding texts",
) -> torch.Tensor:
    features_list = []
    model.eval()

    for start in tqdm(range(0, len(captions), batch_size), desc=desc):
        batch_caps = captions[start:start + batch_size]
        tokens = tokenizer(batch_caps).to(device)

        if amp and device.startswith("cuda"):
            with torch.cuda.amp.autocast():
                feats = model.encode_text(tokens)
        else:
            feats = model.encode_text(tokens)

        feats = F.normalize(feats.float(), dim=-1)
        features_list.append(feats.cpu())

    return torch.cat(features_list, dim=0)


# =========================
# Hard negative mining
# =========================

@torch.no_grad()
def mine_hard_negatives(
    model,
    tokenizer,
    preprocess,
    train_image_paths: List[str],
    train_captions: List[str],
    train_caption_to_image: torch.LongTensor,
    args,
):
    """
    只在 train split 上挖 hard negatives。

    输出：
    image_to_hard_captions: [num_images, hard_topk]
        每张 image 对应若干 hard negative caption indices。

    caption_to_hard_images: [num_captions, hard_topk]
        每条 caption 对应若干 hard negative image indices。
    """
    os.makedirs(args.output_dir, exist_ok=True)

    if os.path.exists(args.hard_negative_cache):
        print(f"Loading cached hard negatives from: {args.hard_negative_cache}")
        cache = torch.load(args.hard_negative_cache, map_location="cpu")
        return cache["image_to_hard_captions"], cache["caption_to_hard_images"]

    print("=" * 80)
    print("Mining Hard Negatives on TRAIN split only")
    print("=" * 80)

    image_features = encode_images(
        model=model,
        preprocess=preprocess,
        image_paths=train_image_paths,
        batch_size=args.mine_image_batch_size,
        num_workers=args.num_workers,
        device=args.device,
        amp=args.amp,
        desc="Mining: encoding train images",
    )

    text_features = encode_texts(
        model=model,
        tokenizer=tokenizer,
        captions=train_captions,
        batch_size=args.mine_text_batch_size,
        device=args.device,
        amp=args.amp,
        desc="Mining: encoding train captions",
    )

    print("Mining feature shapes:")
    print(f"image_features: {tuple(image_features.shape)}")
    print(f"text_features:  {tuple(text_features.shape)}")

    num_images = image_features.size(0)
    num_captions = text_features.size(0)

    image_features = image_features.to(args.device)
    text_features = text_features.to(args.device)
    caption_to_image = train_caption_to_image.to(args.device)

    hard_topk = args.hard_topk
    extra_topk = args.hard_topk + args.extra_topk_for_filter

    # -------------------------
    # Image -> hard negative captions
    # -------------------------
    image_to_hard_captions = []

    for start in tqdm(range(0, num_images, args.mine_sim_chunk_size), desc="Mining image->hard captions"):
        end = min(start + args.mine_sim_chunk_size, num_images)

        sims = image_features[start:end] @ text_features.T

        topk = sims.topk(k=min(extra_topk, num_captions), dim=1).indices

        for local_i, image_idx in enumerate(range(start, end)):
            candidate_caps = topk[local_i]

            # 排除属于当前 image 的 positive captions
            candidate_img_ids = caption_to_image[candidate_caps]
            mask = candidate_img_ids != image_idx
            negatives = candidate_caps[mask][:hard_topk]

            if len(negatives) < hard_topk:
                raise RuntimeError(
                    f"Not enough hard negative captions for image {image_idx}. "
                    f"Increase extra_topk_for_filter."
                )

            image_to_hard_captions.append(negatives.cpu())

    image_to_hard_captions = torch.stack(image_to_hard_captions, dim=0)

    # -------------------------
    # Caption -> hard negative images
    # -------------------------
    caption_to_hard_images = []

    for start in tqdm(range(0, num_captions, args.mine_sim_chunk_size), desc="Mining caption->hard images"):
        end = min(start + args.mine_sim_chunk_size, num_captions)

        sims = text_features[start:end] @ image_features.T

        topk = sims.topk(k=min(extra_topk, num_images), dim=1).indices

        gt_images = caption_to_image[start:end]

        for local_i in range(end - start):
            candidate_imgs = topk[local_i]
            gt_img = gt_images[local_i]

            # 排除真正匹配的 image
            mask = candidate_imgs != gt_img
            negatives = candidate_imgs[mask][:hard_topk]

            if len(negatives) < hard_topk:
                raise RuntimeError(
                    f"Not enough hard negative images for caption {start + local_i}. "
                    f"Increase extra_topk_for_filter."
                )

            caption_to_hard_images.append(negatives.cpu())

    caption_to_hard_images = torch.stack(caption_to_hard_images, dim=0)

    cache = {
        "image_to_hard_captions": image_to_hard_captions,
        "caption_to_hard_images": caption_to_hard_images,
        "hard_topk": hard_topk,
        "source": "train split only",
        "init_lora_ckpt": args.init_lora_ckpt,
    }

    torch.save(cache, args.hard_negative_cache)

    print("=" * 80)
    print("Hard negative mining finished.")
    print(f"image_to_hard_captions: {tuple(image_to_hard_captions.shape)}")
    print(f"caption_to_hard_images: {tuple(caption_to_hard_images.shape)}")
    print(f"Saved cache to: {args.hard_negative_cache}")
    print("=" * 80)

    return image_to_hard_captions, caption_to_hard_images


# =========================
# Loss
# =========================

def clip_contrastive_loss(image_features, text_features, logit_scale: float = 100.0):
    image_features = F.normalize(image_features.float(), dim=-1)
    text_features = F.normalize(text_features.float(), dim=-1)

    logits = logit_scale * image_features @ text_features.T
    labels = torch.arange(logits.size(0), device=logits.device)

    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)

    return (loss_i2t + loss_t2i) / 2.0


def hard_negative_loss(
    image_features,
    text_features,
    hard_text_features,
    hard_image_features,
    logit_scale: float = 100.0,
):
    """
    对每个样本构造两个二分类难负样本任务：

    1. 给定 image，在 positive caption 和 hard negative caption 中选 positive。
    2. 给定 caption，在 positive image 和 hard negative image 中选 positive。
    """
    image_features = F.normalize(image_features.float(), dim=-1)
    text_features = F.normalize(text_features.float(), dim=-1)
    hard_text_features = F.normalize(hard_text_features.float(), dim=-1)
    hard_image_features = F.normalize(hard_image_features.float(), dim=-1)

    # image -> [positive text, hard negative text]
    pos_i2t = (image_features * text_features).sum(dim=-1)
    neg_i2t = (image_features * hard_text_features).sum(dim=-1)
    logits_i2t = torch.stack([pos_i2t, neg_i2t], dim=1) * logit_scale

    # text -> [positive image, hard negative image]
    pos_t2i = (text_features * image_features).sum(dim=-1)
    neg_t2i = (text_features * hard_image_features).sum(dim=-1)
    logits_t2i = torch.stack([pos_t2i, neg_t2i], dim=1) * logit_scale

    labels = torch.zeros(image_features.size(0), dtype=torch.long, device=image_features.device)

    loss_i2t = F.cross_entropy(logits_i2t, labels)
    loss_t2i = F.cross_entropy(logits_t2i, labels)

    return (loss_i2t + loss_t2i) / 2.0


# =========================
# Validation Evaluation
# =========================

@torch.no_grad()
def evaluate_t2i(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    caption_to_image: torch.LongTensor,
    ks=(1, 5, 10),
    device: str = "cuda",
    chunk_size: int = 1024,
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
    chunk_size: int = 512,
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
    model,
    tokenizer,
    preprocess,
    image_paths: List[str],
    captions: List[str],
    caption_to_image: torch.LongTensor,
    args,
) -> Dict[str, float]:
    image_features = encode_images(
        model=model,
        preprocess=preprocess,
        image_paths=image_paths,
        batch_size=args.eval_image_batch_size,
        num_workers=args.num_workers,
        device=args.device,
        amp=args.amp,
        desc="Encoding val images",
    )

    text_features = encode_texts(
        model=model,
        tokenizer=tokenizer,
        captions=captions,
        batch_size=args.eval_text_batch_size,
        device=args.device,
        amp=args.amp,
        desc="Encoding val texts",
    )

    print("Val feature shapes:")
    print(f"image_features: {tuple(image_features.shape)}")
    print(f"text_features:  {tuple(text_features.shape)}")
    print(f"caption_to_image: {tuple(caption_to_image.shape)}")

    t2i_results = evaluate_t2i(
        image_features=image_features,
        text_features=text_features,
        caption_to_image=caption_to_image,
        ks=(1, 5, 10),
        device=args.device,
    )

    i2t_results = evaluate_i2t(
        image_features=image_features,
        text_features=text_features,
        caption_to_image=caption_to_image,
        ks=(1, 5, 10),
        device=args.device,
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


def print_eval_results(results: Dict[str, float], title: str):
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
# Scheduler / Save
# =========================

def build_warmup_cosine_scheduler(optimizer, total_steps: int, warmup_steps: int):
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))

        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(
    model,
    optimizer,
    epoch: int,
    best_score: float,
    val_results: Dict[str, float],
    args,
    lora_config: Dict,
    save_path: str,
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "best_score": best_score,
        "val_results": val_results,

        # 兼容你之前 eval_test_checkpoint.py 的 lora 加载逻辑
        "lora_state_dict": get_lora_state_dict(model),

        "optimizer": optimizer.state_dict(),

        "config": {
            "model_name": lora_config["model_name"],
            "pretrained": lora_config["pretrained"],
            "lora_r": lora_config["lora_r"],
            "lora_alpha": lora_config["lora_alpha"],
            "lora_dropout": lora_config["lora_dropout"],
            "lora_on_k": lora_config["lora_on_k"],

            "train_split": args.train_split,
            "val_split": args.val_split,
            "init_lora_ckpt": args.init_lora_ckpt,
            "hard_negative": True,
            "hard_topk": args.hard_topk,
            "hard_loss_weight": args.hard_loss_weight,
        }
    }

    torch.save(checkpoint, save_path)
    print(f"Saved best hard-negative LoRA checkpoint to: {save_path}")


def save_val_log(log_rows: List[Dict], save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pd.DataFrame(log_rows).to_csv(save_path, index=False)
    print(f"Saved val log to: {save_path}")


# =========================
# Train
# =========================

def train_one_epoch(
    model,
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
    total_clip_loss = 0.0
    total_hn_loss = 0.0
    total_samples = 0

    progress = tqdm(train_loader, desc=f"HardNeg Training Epoch {epoch}")

    for images, pos_captions, hard_captions, hard_images in progress:
        images = images.to(args.device, non_blocking=True)
        hard_images = hard_images.to(args.device, non_blocking=True)

        pos_texts = tokenizer(list(pos_captions)).to(args.device)
        hard_texts = tokenizer(list(hard_captions)).to(args.device)

        optimizer.zero_grad(set_to_none=True)

        if args.amp and args.device.startswith("cuda"):
            with torch.cuda.amp.autocast():
                image_features = model.encode_image(images)
                text_features = model.encode_text(pos_texts)

                hard_text_features = model.encode_text(hard_texts)
                hard_image_features = model.encode_image(hard_images)

                loss_clip = clip_contrastive_loss(
                    image_features=image_features,
                    text_features=text_features,
                    logit_scale=args.logit_scale,
                )

                loss_hn = hard_negative_loss(
                    image_features=image_features,
                    text_features=text_features,
                    hard_text_features=hard_text_features,
                    hard_image_features=hard_image_features,
                    logit_scale=args.logit_scale,
                )

                loss = loss_clip + args.hard_loss_weight * loss_hn

            scaler.scale(loss).backward()

            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(get_trainable_params(model), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

        else:
            image_features = model.encode_image(images)
            text_features = model.encode_text(pos_texts)

            hard_text_features = model.encode_text(hard_texts)
            hard_image_features = model.encode_image(hard_images)

            loss_clip = clip_contrastive_loss(
                image_features=image_features,
                text_features=text_features,
                logit_scale=args.logit_scale,
            )

            loss_hn = hard_negative_loss(
                image_features=image_features,
                text_features=text_features,
                hard_text_features=hard_text_features,
                hard_image_features=hard_image_features,
                logit_scale=args.logit_scale,
            )

            loss = loss_clip + args.hard_loss_weight * loss_hn

            loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(get_trainable_params(model), args.grad_clip)

            optimizer.step()

        scheduler.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_clip_loss += loss_clip.item() * batch_size
        total_hn_loss += loss_hn.item() * batch_size
        total_samples += batch_size

        progress.set_postfix({
            "loss": f"{loss.item():.4f}",
            "clip": f"{loss_clip.item():.4f}",
            "hn": f"{loss_hn.item():.4f}",
            "lr": f"{scheduler.get_last_lr()[0]:.2e}",
        })

    avg_loss = total_loss / max(1, total_samples)
    avg_clip_loss = total_clip_loss / max(1, total_samples)
    avg_hn_loss = total_hn_loss / max(1, total_samples)

    return avg_loss, avg_clip_loss, avg_hn_loss


# =========================
# Main
# =========================

def main():
    args = types.SimpleNamespace(
        # 路径
        csv_path="/root/autodl-tmp/flickr_annotations_30k.csv",
        img_dir="/root/autodl-tmp/flickr30k-images",

        # 初始化：用你的 best LoRA checkpoint
        init_lora_ckpt="./outputs_lora/best_lora_clip.pt",

        # 严格区分
        train_split="train",
        val_split="val",

        # 基础模型兜底配置，会优先用 checkpoint config 里的配置
        model_name="ViT-B-32",
        pretrained="/root/autodl-tmp/model/open_clip_model.safetensors",

        # LoRA 兜底配置，会优先用 checkpoint config 里的配置
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        lora_on_k=False,

        # hard negative mining
        hard_topk=20,
        extra_topk_for_filter=50,
        mine_image_batch_size=64,
        mine_text_batch_size=256,
        mine_sim_chunk_size=512,

        # hard negative training
        epochs=3,
        train_batch_size=32,
        lr=5e-5,
        weight_decay=1e-4,
        warmup_ratio=0.05,
        logit_scale=100.0,
        hard_loss_weight=0.3,
        grad_clip=1.0,

        # validation
        eval_image_batch_size=64,
        eval_text_batch_size=128,

        # system
        num_workers=4,
        device="cuda" if torch.cuda.is_available() else "cpu",
        amp=True,
        seed=42,
        skip_missing=False,

        # output
        output_dir="./outputs_hard_negative",
        hard_negative_cache="./outputs_hard_negative/hard_negatives_train.pt",
        best_ckpt_name="best_hardneg_lora.pt",
        val_log_name="val_log.csv",
    )

    set_seed(args.seed)

    # 防止误用
    assert args.train_split == "train", "train_split must be 'train'."
    assert args.val_split == "val", "val_split must be 'val'."
    assert "test" not in args.train_split
    assert "test" not in args.val_split

    print("=" * 80)
    print("LoRA + Hard Negative Training")
    print("=" * 80)
    print(f"CSV path: {args.csv_path}")
    print(f"Image dir: {args.img_dir}")
    print(f"Initial LoRA checkpoint: {args.init_lora_ckpt}")
    print(f"Train split: {args.train_split}")
    print(f"Val split: {args.val_split}")
    print("Test split is NOT used in this script.")
    print(f"Device: {args.device}")
    print(f"AMP: {args.amp}")
    print("=" * 80)

    # 加载 best_lora_clip.pt
    model, tokenizer, train_preprocess, val_preprocess, lora_config = load_lora_model_from_checkpoint(args)

    # 构建 train retrieval data，用于挖 hard negative
    train_rows, train_image_paths, train_captions, train_caption_to_image, train_image_to_captions = build_retrieval_data(
        csv_path=args.csv_path,
        img_dir=args.img_dir,
        split=args.train_split,
        skip_missing=args.skip_missing,
    )

    # 只在 train split 上挖 hard negatives
    image_to_hard_captions, caption_to_hard_images = mine_hard_negatives(
        model=model,
        tokenizer=tokenizer,
        preprocess=val_preprocess,
        train_image_paths=train_image_paths,
        train_captions=train_captions,
        train_caption_to_image=train_caption_to_image,
        args=args,
    )

    train_dataset = HardNegativeTrainDataset(
        rows=train_rows,
        captions=train_captions,
        image_to_captions=train_image_to_captions,
        image_to_hard_captions=image_to_hard_captions,
        caption_to_hard_images=caption_to_hard_images,
        transform=train_preprocess,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        drop_last=True,
    )

    # val split 只用于选 best checkpoint
    _, val_image_paths, val_captions, val_caption_to_image, _ = build_retrieval_data(
        csv_path=args.csv_path,
        img_dir=args.img_dir,
        split=args.val_split,
        skip_missing=args.skip_missing,
    )

    optimizer = torch.optim.AdamW(
        get_trainable_params(model),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(args.warmup_ratio * total_steps)

    scheduler = build_warmup_cosine_scheduler(
        optimizer=optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
    )

    scaler = torch.cuda.amp.GradScaler(
        enabled=args.amp and args.device.startswith("cuda")
    )

    best_score = -1.0
    log_rows = []

    best_ckpt_path = os.path.join(args.output_dir, args.best_ckpt_name)
    val_log_path = os.path.join(args.output_dir, args.val_log_name)

    print("\nStart hard negative training...")
    print(f"Total train steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps}")
    print(f"Hard loss weight: {args.hard_loss_weight}")
    print("=" * 80)

    for epoch in range(1, args.epochs + 1):
        avg_loss, avg_clip_loss, avg_hn_loss = train_one_epoch(
            model=model,
            tokenizer=tokenizer,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            args=args,
        )

        print(f"\nEpoch {epoch} finished.")
        print(f"Train loss: {avg_loss:.4f}")
        print(f"CLIP loss:  {avg_clip_loss:.4f}")
        print(f"HN loss:    {avg_hn_loss:.4f}")

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
            title=f"Validation Results - HardNeg Epoch {epoch}"
        )

        row = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "clip_loss": avg_clip_loss,
            "hard_negative_loss": avg_hn_loss,
            "val_mean_recall": val_score,
            **val_results,
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
                lora_config=lora_config,
                save_path=best_ckpt_path,
            )

            print(f"New best val mean recall: {best_score:.2f}%")
        else:
            print(f"No improvement. Best val mean recall: {best_score:.2f}%")

    print("\n" + "=" * 80)
    print("Hard negative training finished.")
    print(f"Best val mean recall: {best_score:.2f}%")
    print(f"Best checkpoint: {best_ckpt_path}")
    print("Reminder: test split was NOT used. Evaluate test after selecting best checkpoint.")
    print("=" * 80)


if __name__ == "__main__":
    main()


# In[ ]:




