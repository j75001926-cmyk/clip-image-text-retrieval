#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import ast
import json
import math
import types
from pathlib import Path
from typing import List, Dict

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


def build_rows(csv_path: str, img_dir: str, split: str, skip_missing: bool = False):
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
            raise FileNotFoundError(f"Image not found: {img_path}")

        captions = parse_caption_list(row["raw"])

        rows.append({
            "filename": filename,
            "image_path": str(img_path),
            "captions": captions,
        })

    return rows


def build_eval_data(csv_path: str, img_dir: str, split: str, skip_missing: bool = False):
    rows = build_rows(
        csv_path=csv_path,
        img_dir=img_dir,
        split=split,
        skip_missing=skip_missing,
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
    print("Eval Data")
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
# LoRA modules
# 这部分保持和你 eval_test_checkpoint.py 成功跑通的版本一致
# =========================

class LoRAProjection(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.05,
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
                self.base_attn.out_proj.bias,
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
                ),
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


def load_lora_model(args):
    checkpoint = torch.load(args.ckpt_path, map_location="cpu")
    config = checkpoint.get("config", {})

    model_name = config.get("model_name", args.model_name)
    pretrained = config.get("pretrained", args.pretrained)

    lora_r = config.get("lora_r", args.lora_r)
    lora_alpha = config.get("lora_alpha", args.lora_alpha)
    lora_dropout = config.get("lora_dropout", args.lora_dropout)
    lora_on_k = config.get("lora_on_k", args.lora_on_k)

    print("=" * 80)
    print("Loading LoRA Model")
    print("=" * 80)
    print(f"Checkpoint: {args.ckpt_path}")
    print(f"Model name: {model_name}")
    print(f"Pretrained: {pretrained}")
    print(f"LoRA r: {lora_r}")
    print(f"LoRA alpha: {lora_alpha}")
    print(f"LoRA dropout: {lora_dropout}")
    print(f"LoRA on k: {lora_on_k}")
    print("=" * 80)

    model, _, preprocess = open_clip.create_model_and_transforms(
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
        raise RuntimeError("No nn.MultiheadAttention modules found.")

    model = model.to(args.device)

    lora_state_dict = checkpoint["lora_state_dict"]
    model_state = model.state_dict()

    expected_lora_keys = [k for k in model_state.keys() if "lora_" in k]
    missing_lora_keys = [k for k in expected_lora_keys if k not in lora_state_dict]
    unexpected_lora_keys = [k for k in lora_state_dict.keys() if k not in model_state]

    if len(missing_lora_keys) > 0:
        raise RuntimeError(
            f"Missing LoRA keys when loading checkpoint. First keys: {missing_lora_keys[:10]}"
        )

    if len(unexpected_lora_keys) > 0:
        raise RuntimeError(
            f"Unexpected LoRA keys in checkpoint. First keys: {unexpected_lora_keys[:10]}"
        )

    model.load_state_dict(lora_state_dict, strict=False)

    print(f"Injected LoRA into {num_replaced} attention modules.")
    print(f"Loaded LoRA tensors: {len(lora_state_dict)}")
    print("=" * 80)

    model.eval()

    return model, tokenizer, preprocess, model_name, pretrained


# =========================
# Hook-based token extraction
# =========================

@torch.no_grad()
def encode_image_global_and_patches(model, images):
    """
    global feature 用 model.encode_image。
    patch token 用 hook 从 visual.transformer 捕获。
    """
    model.eval()

    cache = {}

    def hook_fn(module, inputs, output):
        if isinstance(output, tuple):
            output = output[0]
        cache["visual_tokens"] = output.detach()

    handle = model.visual.transformer.register_forward_hook(hook_fn)

    global_features = model.encode_image(images)

    handle.remove()

    if "visual_tokens" not in cache:
        raise RuntimeError("Failed to capture visual transformer tokens.")

    tokens = cache["visual_tokens"]

    # OpenCLIP ViT 通常是 [L, B, C]
    if tokens.shape[0] == images.shape[0]:
        # [B, L, C]
        tokens = tokens
    elif tokens.shape[1] == images.shape[0]:
        # [L, B, C] -> [B, L, C]
        tokens = tokens.permute(1, 0, 2)
    else:
        raise RuntimeError(
            f"Unexpected visual token shape: {tokens.shape}, batch size={images.shape[0]}"
        )

    patch_tokens = tokens[:, 1:, :]

    if hasattr(model.visual, "ln_post") and model.visual.ln_post is not None:
        patch_tokens = model.visual.ln_post(patch_tokens)

    if hasattr(model.visual, "proj") and model.visual.proj is not None:
        patch_tokens = patch_tokens @ model.visual.proj

    global_features = F.normalize(global_features.float(), dim=-1)
    patch_features = F.normalize(patch_tokens.float(), dim=-1)

    return global_features, patch_features


@torch.no_grad()
def encode_text_global_and_tokens(model, tokens):
    """
    global feature 用 model.encode_text。
    token feature 用 hook 从 ln_final 捕获。
    """
    model.eval()

    cache = {}

    def hook_fn(module, inputs, output):
        if isinstance(output, tuple):
            output = output[0]
        cache["text_tokens"] = output.detach()

    handle = model.ln_final.register_forward_hook(hook_fn)

    global_features = model.encode_text(tokens)

    handle.remove()

    if "text_tokens" not in cache:
        raise RuntimeError("Failed to capture text tokens from ln_final.")

    token_features = cache["text_tokens"]

    # OpenCLIP encode_text 里的 ln_final 输出通常是 [B, L, C]
    if token_features.shape[0] == tokens.shape[0]:
        token_features = token_features
    elif token_features.shape[1] == tokens.shape[0]:
        token_features = token_features.permute(1, 0, 2)
    else:
        raise RuntimeError(
            f"Unexpected text token shape: {token_features.shape}, batch size={tokens.shape[0]}"
        )

    if hasattr(model, "text_projection") and model.text_projection is not None:
        token_features = token_features @ model.text_projection

    # padding token 为 0；第 0 位一般是 SOS；EOT 也不参与局部匹配
    token_mask = tokens != 0
    token_mask[:, 0] = False

    eot_indices = tokens.argmax(dim=-1)
    batch_indices = torch.arange(tokens.shape[0], device=tokens.device)
    token_mask[batch_indices, eot_indices] = False

    global_features = F.normalize(global_features.float(), dim=-1)
    token_features = F.normalize(token_features.float(), dim=-1)

    return global_features, token_features, token_mask


# =========================
# Feature extraction
# =========================

@torch.no_grad()
def extract_image_features_and_patches(model, preprocess, image_paths: List[str], args):
    dataset = ImageOnlyDataset(image_paths, preprocess)

    loader = DataLoader(
        dataset,
        batch_size=args.image_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )

    all_global = []
    all_patches = []

    model.eval()

    for images in tqdm(loader, desc="Extracting image global + patch tokens"):
        images = images.to(args.device, non_blocking=True)

        if args.amp and args.device.startswith("cuda"):
            with torch.cuda.amp.autocast():
                global_feats, patch_feats = encode_image_global_and_patches(model, images)
        else:
            global_feats, patch_feats = encode_image_global_and_patches(model, images)

        all_global.append(global_feats.cpu())
        all_patches.append(patch_feats.cpu())

    image_global = torch.cat(all_global, dim=0)
    image_patches = torch.cat(all_patches, dim=0)

    return image_global, image_patches


@torch.no_grad()
def extract_text_features_and_tokens(model, tokenizer, captions: List[str], args):
    all_global = []
    all_tokens = []
    all_masks = []

    model.eval()

    for start in tqdm(
        range(0, len(captions), args.text_batch_size),
        desc="Extracting text global + token features",
    ):
        batch_caps = captions[start:start + args.text_batch_size]
        tokens = tokenizer(batch_caps).to(args.device)

        if args.amp and args.device.startswith("cuda"):
            with torch.cuda.amp.autocast():
                global_feats, token_feats, token_mask = encode_text_global_and_tokens(model, tokens)
        else:
            global_feats, token_feats, token_mask = encode_text_global_and_tokens(model, tokens)

        all_global.append(global_feats.cpu())
        all_tokens.append(token_feats.cpu())
        all_masks.append(token_mask.cpu())

    text_global = torch.cat(all_global, dim=0)
    text_tokens = torch.cat(all_tokens, dim=0)
    text_masks = torch.cat(all_masks, dim=0)

    return text_global, text_tokens, text_masks


# =========================
# Global evaluation
# =========================

@torch.no_grad()
def evaluate_global_only(
    image_global: torch.Tensor,
    text_global: torch.Tensor,
    caption_to_image: torch.LongTensor,
    ks=(1, 5, 10),
):
    sim_t2i = text_global @ image_global.T
    sim_i2t = image_global @ text_global.T

    num_images = image_global.size(0)
    max_k = max(ks)

    results = {}

    topk_images = sim_t2i.topk(k=max_k, dim=1).indices
    gt_images = caption_to_image.unsqueeze(1)

    for k in ks:
        results[f"global_t2i_R@{k}"] = (
            topk_images[:, :k].eq(gt_images).any(dim=1).float().mean().item() * 100.0
        )

    topk_caps = sim_i2t.topk(k=max_k, dim=1).indices
    topk_cap_images = caption_to_image[topk_caps]
    gt_image_ids = torch.arange(num_images).unsqueeze(1)

    for k in ks:
        results[f"global_i2t_R@{k}"] = (
            topk_cap_images[:, :k].eq(gt_image_ids).any(dim=1).float().mean().item() * 100.0
        )

    return results, sim_t2i, sim_i2t


def global_mean_recall(global_results: Dict[str, float]) -> float:
    keys = [
        "global_t2i_R@1",
        "global_t2i_R@5",
        "global_t2i_R@10",
        "global_i2t_R@1",
        "global_i2t_R@5",
        "global_i2t_R@10",
    ]
    return sum(global_results[k] for k in keys) / len(keys)


def assert_global_sanity(global_results: Dict[str, float], min_mean_recall: float = 70.0):
    mr = global_mean_recall(global_results)

    print("\n" + "=" * 80)
    print("Global-only sanity check")
    print("=" * 80)
    print(f"global_t2i_R@1 :  {global_results['global_t2i_R@1']:.2f}")
    print(f"global_t2i_R@5 :  {global_results['global_t2i_R@5']:.2f}")
    print(f"global_t2i_R@10:  {global_results['global_t2i_R@10']:.2f}")
    print(f"global_i2t_R@1 :  {global_results['global_i2t_R@1']:.2f}")
    print(f"global_i2t_R@5 :  {global_results['global_i2t_R@5']:.2f}")
    print(f"global_i2t_R@10:  {global_results['global_i2t_R@10']:.2f}")
    print(f"global_mean_recall: {mr:.2f}")
    print("=" * 80)

    if mr < min_mean_recall:
        raise RuntimeError(
            f"Global-only Recall is abnormal: mean_recall={mr:.2f}. "
            f"Do not trust rerank results. Check LoRA loading and feature extraction."
        )


# =========================
# Local score
# =========================

def local_score_text_to_images(
    text_token_features: torch.Tensor,
    image_patch_features: torch.Tensor,
):
    """
    text_token_features:  [L, D]
    image_patch_features: [K, P, D]
    return: [K]
    """
    if text_token_features.size(0) == 0:
        return torch.zeros(image_patch_features.size(0), device=image_patch_features.device)

    sim = torch.einsum("ld,kpd->klp", text_token_features, image_patch_features)
    token_best = sim.max(dim=-1).values
    local_scores = token_best.mean(dim=-1)

    return local_scores


def local_score_image_to_texts(
    image_patch_features: torch.Tensor,
    text_token_features: torch.Tensor,
    text_token_masks: torch.Tensor,
):
    """
    image_patch_features: [P, D]
    text_token_features:  [K, L, D]
    text_token_masks:     [K, L]
    return: [K]
    """
    sim = torch.einsum("pd,kld->kpl", image_patch_features, text_token_features)
    token_best = sim.max(dim=1).values

    masks = text_token_masks.float()
    token_best = token_best * masks

    denom = masks.sum(dim=1).clamp(min=1.0)
    local_scores = token_best.sum(dim=1) / denom

    return local_scores


# =========================
# Rerank evaluation
# =========================

@torch.no_grad()
def evaluate_t2i_rerank(
    sim_t2i: torch.Tensor,
    image_patches: torch.Tensor,
    text_tokens: torch.Tensor,
    text_masks: torch.Tensor,
    caption_to_image: torch.LongTensor,
    alpha: float,
    rerank_topk: int,
    ks=(1, 5, 10),
    device: str = "cuda",
):
    num_captions = sim_t2i.size(0)
    max_k = max(ks)

    correct = {k: 0 for k in ks}

    for cap_idx in tqdm(range(num_captions), desc=f"T2I rerank alpha={alpha}"):
        global_scores = sim_t2i[cap_idx]
        top_scores, top_image_indices = global_scores.topk(k=rerank_topk)

        if alpha == 0:
            final_scores = top_scores
        else:
            valid_mask = text_masks[cap_idx]
            query_tokens = text_tokens[cap_idx][valid_mask].to(device)
            candidate_patches = image_patches[top_image_indices].to(device)

            local_scores = local_score_text_to_images(
                text_token_features=query_tokens,
                image_patch_features=candidate_patches,
            ).cpu()

            final_scores = top_scores + alpha * local_scores

        reranked_order = final_scores.argsort(descending=True)
        reranked_images = top_image_indices[reranked_order][:max_k]

        gt_img = int(caption_to_image[cap_idx].item())

        for k in ks:
            if (reranked_images[:k] == gt_img).any().item():
                correct[k] += 1

    results = {}
    for k in ks:
        results[f"t2i_R@{k}"] = correct[k] / num_captions * 100.0

    return results


@torch.no_grad()
def evaluate_i2t_rerank(
    sim_i2t: torch.Tensor,
    image_patches: torch.Tensor,
    text_tokens: torch.Tensor,
    text_masks: torch.Tensor,
    caption_to_image: torch.LongTensor,
    alpha: float,
    rerank_topk: int,
    ks=(1, 5, 10),
    device: str = "cuda",
):
    num_images = sim_i2t.size(0)
    max_k = max(ks)

    correct = {k: 0 for k in ks}

    for image_idx in tqdm(range(num_images), desc=f"I2T rerank alpha={alpha}"):
        global_scores = sim_i2t[image_idx]
        top_scores, top_caption_indices = global_scores.topk(k=rerank_topk)

        if alpha == 0:
            final_scores = top_scores
        else:
            query_patches = image_patches[image_idx].to(device)
            candidate_tokens = text_tokens[top_caption_indices].to(device)
            candidate_masks = text_masks[top_caption_indices].to(device)

            local_scores = local_score_image_to_texts(
                image_patch_features=query_patches,
                text_token_features=candidate_tokens,
                text_token_masks=candidate_masks,
            ).cpu()

            final_scores = top_scores + alpha * local_scores

        reranked_order = final_scores.argsort(descending=True)
        reranked_captions = top_caption_indices[reranked_order][:max_k]

        matched_image_indices = caption_to_image[reranked_captions]

        for k in ks:
            if (matched_image_indices[:k] == image_idx).any().item():
                correct[k] += 1

    results = {}
    for k in ks:
        results[f"i2t_R@{k}"] = correct[k] / num_images * 100.0

    return results


def mean_recall(results: Dict[str, float]) -> float:
    keys = [
        "t2i_R@1",
        "t2i_R@5",
        "t2i_R@10",
        "i2t_R@1",
        "i2t_R@5",
        "i2t_R@10",
    ]
    return sum(results[k] for k in keys) / len(keys)


def print_result_table(rows: List[Dict]):
    print("\n" + "=" * 100)
    print("Patch-Token Rerank Results")
    print("=" * 100)

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    print("=" * 100)


def save_results(rows: List[Dict], args):
    os.makedirs(args.output_dir, exist_ok=True)

    ckpt_name = Path(args.ckpt_path).stem
    save_path = os.path.join(
        args.output_dir,
        f"rerank_{args.split}_{ckpt_name}.csv",
    )

    pd.DataFrame(rows).to_csv(save_path, index=False)
    print(f"\nSaved rerank results to: {save_path}")


# =========================
# Main
# =========================

def main():
    args = types.SimpleNamespace(
        # data
        csv_path="/root/autodl-tmp/flickr_annotations_30k.csv",
        img_dir="/root/autodl-tmp/flickr30k-images",

        # 先用 val 选 alpha；选完后再改成 test，只跑选好的 alpha
        split="val",

        # 用当前最好的 checkpoint
        ckpt_path="./outputs_hard_negative/best_hardneg_lora.pt",

        # fallback config，优先用 checkpoint 里的 config
        model_name="ViT-B-32",
        pretrained="/root/autodl-tmp/model/open_clip_model.safetensors",

        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        lora_on_k=False,

        # rerank config
        rerank_topk=100,
        alpha_list=[0.0, 0.005, 0.01, 0.02, 0.03],

        # batch
        image_batch_size=64,
        text_batch_size=128,

        # system
        num_workers=4,
        device="cuda" if torch.cuda.is_available() else "cpu",
        amp=True,
        skip_missing=False,

        # output
        output_dir="./outputs_rerank",
    )

    assert args.split in ["val", "test"]
    assert args.rerank_topk >= 10

    print("=" * 80)
    print("Patch-Token Rerank Fixed")
    print("=" * 80)
    print(f"Split: {args.split}")
    print(f"Checkpoint: {args.ckpt_path}")
    print(f"Rerank top-k: {args.rerank_topk}")
    print(f"Alpha list: {args.alpha_list}")
    print(f"Device: {args.device}")
    print(f"AMP: {args.amp}")
    print("=" * 80)

    image_paths, captions, caption_to_image, _ = build_eval_data(
        csv_path=args.csv_path,
        img_dir=args.img_dir,
        split=args.split,
        skip_missing=args.skip_missing,
    )

    model, tokenizer, preprocess, model_name, pretrained = load_lora_model(args)

    print("\nExtracting image global features and patch tokens...")
    image_global, image_patches = extract_image_features_and_patches(
        model=model,
        preprocess=preprocess,
        image_paths=image_paths,
        args=args,
    )

    print("\nExtracting text global features and token features...")
    text_global, text_tokens, text_masks = extract_text_features_and_tokens(
        model=model,
        tokenizer=tokenizer,
        captions=captions,
        args=args,
    )

    print("\nFeature shapes:")
    print(f"image_global: {tuple(image_global.shape)}")
    print(f"image_patches: {tuple(image_patches.shape)}")
    print(f"text_global:  {tuple(text_global.shape)}")
    print(f"text_tokens:  {tuple(text_tokens.shape)}")
    print(f"text_masks:   {tuple(text_masks.shape)}")
    print(f"caption_to_image: {tuple(caption_to_image.shape)}")

    print("\nEvaluating global-only retrieval...")
    global_results, sim_t2i, sim_i2t = evaluate_global_only(
        image_global=image_global,
        text_global=text_global,
        caption_to_image=caption_to_image,
        ks=(1, 5, 10),
    )

    assert_global_sanity(global_results, min_mean_recall=70.0)

    rows = []

    for alpha in args.alpha_list:
        t2i_results = evaluate_t2i_rerank(
            sim_t2i=sim_t2i,
            image_patches=image_patches,
            text_tokens=text_tokens,
            text_masks=text_masks,
            caption_to_image=caption_to_image,
            alpha=alpha,
            rerank_topk=args.rerank_topk,
            ks=(1, 5, 10),
            device=args.device,
        )

        i2t_results = evaluate_i2t_rerank(
            sim_i2t=sim_i2t,
            image_patches=image_patches,
            text_tokens=text_tokens,
            text_masks=text_masks,
            caption_to_image=caption_to_image,
            alpha=alpha,
            rerank_topk=args.rerank_topk,
            ks=(1, 5, 10),
            device=args.device,
        )

        result = {
            "split": args.split,
            "alpha": alpha,
            "rerank_topk": args.rerank_topk,
            **t2i_results,
            **i2t_results,
        }

        result["mean_recall"] = mean_recall(result)
        rows.append(result)

        print("\n" + "=" * 80)
        print(f"Rerank Result | split={args.split} | alpha={alpha} | topk={args.rerank_topk}")
        print("=" * 80)
        print(f"T2I R@1 :  {result['t2i_R@1']:.2f}")
        print(f"T2I R@5 :  {result['t2i_R@5']:.2f}")
        print(f"T2I R@10:  {result['t2i_R@10']:.2f}")
        print(f"I2T R@1 :  {result['i2t_R@1']:.2f}")
        print(f"I2T R@5 :  {result['i2t_R@5']:.2f}")
        print(f"I2T R@10:  {result['i2t_R@10']:.2f}")
        print(f"Mean Recall: {result['mean_recall']:.2f}")

    print_result_table(rows)
    save_results(rows, args)

    best_row = max(rows, key=lambda x: x["mean_recall"])
    print("\nBest alpha on this split:")
    print(best_row)


if __name__ == "__main__":
    main()


# In[ ]:




