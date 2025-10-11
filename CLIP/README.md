# From-Scratch CLIP Implementation in PyTorch 🈲 <-> 🌉

This repository contains a from-scratch implementation of **OpenAI's CLIP (Contrastive Language-Image Pre-training) model**. The goal of this project is to build a first-principles understanding of how multimodal models learn a shared embedding space from raw image-text pairs.

## Core Concepts

CLIP is a self-supervised model that learns visual representations from natural language. It uses a contrastive learning objective to align the vector representations of images and their corresponding text captions in a shared multimodal embedding space.

The training process involves:
1.  Encoding batches of (image, text) pairs using an **Image Encoder** (ViT) and a **Text Encoder** (Transformer).
2.  Projecting the outputs into a shared embedding space.
3.  Calculating a cosine similarity matrix between all image and text embeddings in the batch.
4.  Using a **symmetric cross-entropy loss** to maximize the similarity for the `N` correct pairs (the diagonal) while minimizing it for the `N²-N` incorrect pairs (the off-diagonal).

## Architecture Details

This implementation uses two main components, built from scratch:

#### Text Encoder: BERT-style Transformer
-   **Layers:** 12
-   **Attention Heads:** 8
-   **Embedding Dimensions:** 512
-   **Total Parameters:** ~63M

#### Image Encoder: Vision Transformer (ViT-Base/16)
-   **Layers:** 12
-   **Image Size:** 224x224
-   **Patch Size:** 16x16
-   **Sequence Length (Number of Patches):** (224/16)² = 196
-   **Embedding Dimensions:** 768
-   **Total Parameters:** ~86M

## Training & Experimental Setup

Due to the immense compute required to train CLIP on its original 400M+ dataset, this project serves as an experimental replication on a smaller scale.

-   **Dataset:** Flickr30k (~31,000 images, 5 captions per image).
-   **Regularization:** To improve generalization on this smaller dataset, a random caption was selected for each image at every training step.
-   **Training:** The model was trained for 55 epochs in a single 12-hour session on a Kaggle GPU kernel.

## Results & Analysis

The learning curve below shows the training and validation loss over the 55-epoch run.

![CLIP Learning Curve](./assets/clip_learning_curve.jpg)

**Analysis:**
-   **Success:** The steady downward trend of the training loss confirms that the from-scratch implementation is correct and the model is successfully learning.
-   **Insight:** The validation loss, while also trending downwards, is noisy and "spiky." This is a classic, expected signature of training a very large model on a small dataset. It's a successful experimental replication of the core challenge described in the CLIP paper—demonstrating the model's data-hungry nature.

## Acknowledgment

- This project is based on the original paper: [Learning Transferable Visual Models From Natural Language Supervision by Alec Radford, et al.](https://arxiv.org/abs/2103.00020)

- This project utilises the Flicker30k Dataset released on: [Kaggle](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset)
