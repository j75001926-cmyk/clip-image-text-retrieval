#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import ast
import json
import math
import argparse
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
# Args
# =========================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["adapter", "lora"],
        help="Evaluate adapter checkpoint or lora checkpoint."
    )

    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to best checkpoint."
    )

    parser.add_argument(
        "--csv_path",
        type=str,
        default="/root/autodl-tmp/flickr_annotations_30k.csv"
    )

    parser.add_argument(
        "--img_dir",
        type=str,
        default="/root/autodl-tmp/flickr30k-images"
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["val", "test"]
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="ViT-B-32"
    )

    parser.add_argument(
        "--pretrained",
        type=str,
        default="/root/autodl-tmp/model/open_clip_model.safetensors"
    )

    parser.add_argument("--eval_image_batch_size", type=int, default=64)
    parser.add_argument("--eval_text_batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu"
    )

    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--skip_missing", action="store_true")

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs_test_eval"
    )

    return parser.parse_args()


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


def build_eval_data(
    csv_path: str,
    img_dir: str,
    split: str,
    skip_missing: bool = False
) -> Tuple[List[str], List[str], torch.LongTensor, List[str]]:
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

    print("Sanity check:")
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
# Adapter Model
# =========================

class ResidualAdapter(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,
        bottleneck_dim: int = 128,
        residual_scale: float = 0.2
    ):
        super().__init__()

        self.residual_scale = residual_scale

        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, embed_dim),
        )

        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return x + self.residual_scale * self.net(x)


class FinetuneCLIP(nn.Module):
    def __init__(
        self,
        clip_model,
        embed_dim: int = 512,
        bottleneck_dim: int = 128,
        residual_scale: float = 0.2,
    ):
        super().__init__()

        self.clip = clip_model

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


# =========================
# LoRA Model
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


# =========================
# Load Models
# =========================

def load_adapter_model(args):
    checkpoint = torch.load(args.ckpt_path, map_location="cpu")
    config = checkpoint.get("config", {})

    model_name = config.get("model_name", args.model_name)
    pretrained = config.get("pretrained", args.pretrained)

    embed_dim = config.get("embed_dim", 512)
    bottleneck_dim = config.get("bottleneck_dim", 128)
    residual_scale = config.get("residual_scale", 0.2)

    print("=" * 80)
    print("Loading Adapter Checkpoint")
    print("=" * 80)
    print(f"Checkpoint: {args.ckpt_path}")
    print(f"Model name: {model_name}")
    print(f"Pretrained: {pretrained}")
    print(f"embed_dim: {embed_dim}")
    print(f"bottleneck_dim: {bottleneck_dim}")
    print(f"residual_scale: {residual_scale}")
    print("=" * 80)

    clip_model, _, eval_preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        device=args.device
    )

    model = FinetuneCLIP(
        clip_model=clip_model,
        embed_dim=embed_dim,
        bottleneck_dim=bottleneck_dim,
        residual_scale=residual_scale,
    )

    model.image_adapter.load_state_dict(checkpoint["image_adapter"])
    model.text_adapter.load_state_dict(checkpoint["text_adapter"])

    model = model.to(args.device)
    model.eval()

    tokenizer = open_clip.get_tokenizer(model_name)

    return model, tokenizer, eval_preprocess, model_name, pretrained


def load_lora_model(args):
    checkpoint = torch.load(args.ckpt_path, map_location="cpu")
    config = checkpoint.get("config", {})

    model_name = config.get("model_name", args.model_name)
    pretrained = config.get("pretrained", args.pretrained)

    lora_r = config.get("lora_r", 8)
    lora_alpha = config.get("lora_alpha", 16)
    lora_dropout = config.get("lora_dropout", 0.05)
    lora_on_k = config.get("lora_on_k", False)

    print("=" * 80)
    print("Loading LoRA Checkpoint")
    print("=" * 80)
    print(f"Checkpoint: {args.ckpt_path}")
    print(f"Model name: {model_name}")
    print(f"Pretrained: {pretrained}")
    print(f"LoRA r: {lora_r}")
    print(f"LoRA alpha: {lora_alpha}")
    print(f"LoRA dropout: {lora_dropout}")
    print(f"LoRA on k: {lora_on_k}")
    print("=" * 80)

    model, _, eval_preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        device=args.device
    )

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

    model = model.to(args.device)

    lora_state_dict = checkpoint["lora_state_dict"]
    incompatible = model.load_state_dict(lora_state_dict, strict=False)

    print(f"Injected LoRA into {num_replaced} MultiheadAttention modules.")
    print(f"Loaded LoRA tensors: {len(lora_state_dict)}")
    if len(incompatible.unexpected_keys) > 0:
        print("[Warning] Unexpected keys:")
        print(incompatible.unexpected_keys[:20])

    model.eval()

    tokenizer = open_clip.get_tokenizer(model_name)

    return model, tokenizer, eval_preprocess, model_name, pretrained


# =========================
# Feature Extraction
# =========================

@torch.no_grad()
def encode_eval_images(
    model,
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
        pin_memory=device.startswith("cuda"),
    )

    features_list = []

    model.eval()

    for images in tqdm(loader, desc="Encoding test images"):
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
def encode_eval_texts(
    model,
    tokenizer,
    captions: List[str],
    batch_size: int,
    device: str,
    amp: bool,
) -> torch.Tensor:
    features_list = []

    model.eval()

    for start in tqdm(range(0, len(captions), batch_size), desc="Encoding test texts"):
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
# Recall Evaluation
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
    chunk_size: int = 512,
) -> Dict[str, float]:
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

    print("\nFeature shapes:")
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


def print_results(results: Dict[str, float], title: str):
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


def save_results(
    results: Dict[str, float],
    args,
    model_name: str,
    pretrained: str,
):
    os.makedirs(args.output_dir, exist_ok=True)

    ckpt_name = Path(args.ckpt_path).stem

    row = {
        "method": args.mode,
        "split": args.split,
        "ckpt_path": args.ckpt_path,
        "ckpt_name": ckpt_name,
        "model_name": model_name,
        "pretrained": pretrained,
        **results,
        "mean_recall": mean_recall(results),
    }

    df = pd.DataFrame([row])

    save_path = os.path.join(
        args.output_dir,
        f"test_{args.mode}_{ckpt_name}.csv"
    )

    df.to_csv(save_path, index=False)
    print(f"\nSaved test results to: {save_path}")


# =========================
# Main
# =========================

def main():
    args = parse_args()

    assert args.split == "test", (
        "For final result table, you should use --split test. "
        "Use val only for debugging."
    )

    print("=" * 80)
    print("Evaluate Best Checkpoint on Test Split")
    print("=" * 80)
    print(f"Mode: {args.mode}")
    print(f"Checkpoint: {args.ckpt_path}")
    print(f"CSV path: {args.csv_path}")
    print(f"Image dir: {args.img_dir}")
    print(f"Split: {args.split}")
    print(f"Device: {args.device}")
    print(f"AMP: {args.amp}")
    print("=" * 80)

    image_paths, captions, caption_to_image, _ = build_eval_data(
        csv_path=args.csv_path,
        img_dir=args.img_dir,
        split=args.split,
        skip_missing=args.skip_missing,
    )

    if args.mode == "adapter":
        model, tokenizer, preprocess, model_name, pretrained = load_adapter_model(args)
    elif args.mode == "lora":
        model, tokenizer, preprocess, model_name, pretrained = load_lora_model(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    results = evaluate_retrieval(
        model=model,
        tokenizer=tokenizer,
        preprocess=preprocess,
        image_paths=image_paths,
        captions=captions,
        caption_to_image=caption_to_image,
        args=args,
    )

    title = f"Test Results - {args.mode}"
    print_results(results, title=title)

    save_results(
        results=results,
        args=args,
        model_name=model_name,
        pretrained=pretrained,
    )


if __name__ == "__main__":
    main()


# In[ ]:




