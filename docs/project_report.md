# Project Report: CLIP Image-Text Retrieval with LoRA Fine-Tuning and Hard Negative Mining

## 1. Background

Image-text retrieval is a core task in multimodal learning. Given a text query, the system retrieves the most relevant images. Conversely, given an image, the system retrieves matching captions.

CLIP-style models use a dual-encoder architecture, where images and texts are encoded separately into a shared embedding space. This design is efficient for large-scale retrieval because image embeddings can be precomputed and indexed.

However, global CLIP embeddings may be insufficient for fine-grained matching. For example, descriptions involving clothing color, accessories, number of objects, or subtle actions may require more discriminative image-text alignment.

This project focuses on building a practical image-text retrieval system based on OpenCLIP and improving retrieval performance through parameter-efficient fine-tuning and hard negative training.

## 2. Problem Definition

The retrieval task is evaluated on Flickr30k. Each image is associated with five captions.

Two retrieval directions are considered:

1. Text-to-Image Retrieval: given a caption, retrieve the corresponding image.
2. Image-to-Text Retrieval: given an image, retrieve one of its corresponding captions.

Evaluation metrics:

* Recall@1
* Recall@5
* Recall@10
* Mean Recall

For Text-to-Image retrieval, a query is correct if the ground-truth image appears in the top-K retrieved images.

For Image-to-Text retrieval, a query is correct if any one of the five ground-truth captions appears in the top-K retrieved captions.

## 3. Dataset

Dataset: Flickr30k

Data split:

| Split | Number of Images | Number of Captions |
| ----- | ---------------: | -----------------: |
| Train |           29,000 |            145,000 |
| Val   |            1,014 |              5,070 |
| Test  |            1,000 |              5,000 |

The dataset annotation file contains the following fields:

```text
raw, sentids, split, filename, img_id
```

The `raw` field stores five captions for each image.

## 4. Baseline Method

The baseline uses OpenCLIP ViT-B/32 with pretrained weights.

Feature extraction:

```text
image_features = CLIP_image_encoder(image)
text_features  = CLIP_text_encoder(text)
```

All features are L2-normalized. Similarity is computed by inner product:

```text
similarity = text_features @ image_features.T
```

Because both embeddings are normalized, inner product is equivalent to cosine similarity.

Baseline test result:

| Method        | T2I R@1 | T2I R@5 | T2I R@10 | I2T R@1 | I2T R@5 | I2T R@10 | Mean Recall |
| ------------- | ------: | ------: | -------: | ------: | ------: | -------: | ----------: |
| CLIP Baseline |   66.68 |   88.38 |    93.12 |   84.40 |   96.20 |    98.30 |       87.85 |

The baseline already performs strongly, showing that pretrained CLIP has good zero-shot retrieval ability.

## 5. Adapter Fine-Tuning

The first fine-tuning strategy freezes the CLIP backbone and adds lightweight residual adapters after the image and text embeddings.

The residual adapter is initialized close to identity:

```text
output = x + scale × Adapter(x)
```

This avoids destroying the pretrained CLIP embedding space at the beginning of training.

Test result:

| Method              | Mean Recall |
| ------------------- | ----------: |
| CLIP Baseline       |       87.85 |
| Adapter Fine-tuning |       89.46 |

Adapter fine-tuning improves retrieval performance, but the gain is smaller than LoRA.

## 6. LoRA Fine-Tuning

LoRA is inserted into the attention layers of the image and text transformers. The original CLIP parameters are frozen, and only LoRA parameters are trained.

LoRA modifies attention projections as:

```text
W'x = Wx + BAx
```

where `A` and `B` are low-rank trainable matrices.

Training objective:

```text
L = (CE(image_to_text_logits, labels) + CE(text_to_image_logits, labels)) / 2
```

This is the standard CLIP-style bidirectional contrastive loss.

Test result:

| Method           | T2I R@1 | T2I R@5 | T2I R@10 | I2T R@1 | I2T R@5 | I2T R@10 | Mean Recall |
| ---------------- | ------: | ------: | -------: | ------: | ------: | -------: | ----------: |
| LoRA Fine-tuning |   73.52 |   92.88 |    95.92 |   89.70 |   97.20 |    98.80 |       91.34 |

Compared with the baseline, LoRA improves Mean Recall by 3.49 points and T2I R@1 by 6.84 points.

## 7. Hard Negative Mining

Random in-batch negatives are often too easy. For example, a dog image and a cooking caption are easy to distinguish and provide limited training signal.

Hard negatives are semantically similar but incorrect pairs, such as:

```text
positive: a black dog running on grass
negative: a white dog jumping on grass
```

Mining process:

1. Load the best LoRA checkpoint.
2. Encode all training images and captions.
3. For each image, find high-similarity captions that do not belong to that image.
4. For each caption, find high-similarity images that are not the ground-truth image.
5. Use these samples for hard negative training.

Training loss:

```text
loss = CLIP contrastive loss + λ × hard_negative_loss
```

Hard negative test result:

| Method               | T2I R@1 | T2I R@5 | T2I R@10 | I2T R@1 | I2T R@5 | I2T R@10 | Mean Recall |
| -------------------- | ------: | ------: | -------: | ------: | ------: | -------: | ----------: |
| LoRA Fine-tuning     |   73.52 |   92.88 |    95.92 |   89.70 |   97.20 |    98.80 |       91.34 |
| LoRA + Hard Negative |   74.12 |   92.64 |    95.76 |   89.30 |   97.70 |    98.90 |       91.40 |

Hard negative training gives a small but positive improvement in Mean Recall. The largest gain appears in Text-to-Image R@1 and Image-to-Text R@5/R@10.

The improvement is not large, likely because the LoRA model is already strong and Flickr30k is relatively small.

## 8. Patch-Token Rerank Experiment

A patch-token rerank method was explored to improve fine-grained alignment.

The idea:

```text
global retrieval → Top-K candidates
→ compute local token-patch similarity
→ final_score = global_score + α × local_score
```

Local score calculation:

1. Extract image patch tokens from the ViT image encoder.
2. Extract text token features from the text encoder.
3. For each text token, find the most similar image patch.
4. Average token-level max similarities.

Validation result:

| Alpha | T2I R@1 | T2I R@5 | T2I R@10 | I2T R@1 | I2T R@5 | I2T R@10 | Mean Recall |
| ----: | ------: | ------: | -------: | ------: | ------: | -------: | ----------: |
| 0.000 |   74.48 |   92.33 |    95.56 |   88.07 |   97.34 |    98.62 |       91.07 |
| 0.005 |   74.46 |   92.31 |    95.52 |   88.07 |   97.34 |    98.62 |       91.05 |
| 0.010 |   74.42 |   92.29 |    95.52 |   88.07 |   97.34 |    98.62 |       91.04 |
| 0.020 |   74.40 |   92.25 |    95.52 |   88.07 |   97.34 |    98.62 |       91.03 |
| 0.030 |   74.40 |   92.27 |    95.54 |   87.97 |   97.34 |    98.62 |       91.02 |

The best validation result occurs at α = 0. This means the local patch-token score does not improve the ranking under the current design.

Possible reasons:

1. Function words such as “a”, “the”, “in”, and “with” add noise.
2. CLIP patch tokens are not explicitly supervised for phrase-region alignment.
3. The global CLIP score is already strong.
4. Local score and global score may have different numerical distributions.
5. Simple max-pooling over patch-token similarity may overemphasize spurious local matches.

Conclusion: naive patch-token rerank is kept as a negative ablation result, not as the final method.

## 9. FAISS and Gradio Demo

To make the system usable as a retrieval demo, image embeddings are precomputed and indexed with FAISS.

Offline stage:

```text
image collection → CLIP image embeddings → FAISS IndexFlatIP
```

Online stage:

```text
text query → CLIP text embedding → FAISS search → Top-K images
```

Since embeddings are L2-normalized, inner product search is equivalent to cosine similarity.

A Gradio interface is built for text-to-image retrieval.

Example:

```text
Query: a man wearing an orange hat and glasses
Top-1: 1007129816.jpg
Score: 0.3446
```

This is a successful case because the image has the caption:

```text
A man wears an orange hat and glasses.
```

Demo screenshot:

```text
assets/demo_gradio_orange_hat.png
```

## 10. Final Results and Analysis

Final test results:

| Method               | T2I R@1 | T2I R@5 | T2I R@10 | I2T R@1 | I2T R@5 | I2T R@10 | Mean Recall |
| -------------------- | ------: | ------: | -------: | ------: | ------: | -------: | ----------: |
| CLIP Baseline        |   66.68 |   88.38 |    93.12 |   84.40 |   96.20 |    98.30 |       87.85 |
| Adapter Fine-tuning  |   69.76 |   90.42 |    94.76 |   86.80 |   96.40 |    98.60 |       89.46 |
| LoRA Fine-tuning     |   73.52 |   92.88 |    95.92 |   89.70 |   97.20 |    98.80 |       91.34 |
| LoRA + Hard Negative |   74.12 |   92.64 |    95.76 |   89.30 |   97.70 |    98.90 |       91.40 |

Main observations:

1. LoRA fine-tuning gives the largest performance gain.
2. Hard negative training further improves Mean Recall slightly.
3. Text-to-Image R@1 improves significantly from 66.68 to 74.12.
4. Patch-token rerank does not improve performance under the current naive design.
5. FAISS enables efficient text-to-image retrieval for the demo.

## 11. Limitations

Current limitations:

1. Only Flickr30k is evaluated.
2. Hard negative improvement is small.
3. Patch-token rerank is not robust enough.
4. The demo currently uses a limited image index.
5. Model checkpoints are not included in the repository due to file size constraints.

## 12. Future Work

Possible future improvements:

1. Evaluate on MSCOCO or a larger dataset.
2. Add noun phrase extraction for local rerank.
3. Use token weighting or IDF weighting.
4. Train a lightweight cross-attention reranker.
5. Add qualitative analysis for more failure cases.
6. Use FAISS IVF or HNSW index for larger-scale retrieval.
7. Deploy the demo with a persistent web service.

## 13. Summary

This project builds a complete image-text retrieval pipeline based on OpenCLIP.

The final system includes:

* CLIP zero-shot baseline
* Adapter fine-tuning
* LoRA fine-tuning
* Hard negative training
* Patch-token rerank ablation
* FAISS indexing
* Gradio demo

The final method improves Mean Recall from 87.85 to 91.40 on Flickr30k test set, and improves Text-to-Image R@1 from 66.68 to 74.12.

The project demonstrates practical understanding of CLIP, contrastive learning, parameter-efficient fine-tuning, hard negative mining, retrieval evaluation, and vector search deployment.

