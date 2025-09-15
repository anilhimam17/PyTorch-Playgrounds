# Vision Transformer (ViT-Base/16) from Scratch

This repository contains a PyTorch implementation of the Vision Transformer (ViT-Base/16) model built from first principles. It includes modular components for the patch embedding, multi-head self-attention, feed-forward networks, and the overall Transformer Encoder architecture. The project also provides a robust data pipeline and training loop to facilitate experimentation and understanding of ViT's core mechanisms. The project aims to provide a controlled environment for comparing the training behavior, performance, and overfitting characteristics of the model in contrast to the Resnets.

## ✨ Features

*   **Custom ViT Implementation:**
    *   **Patch Embedding:** Implemented using a single `Conv2d` layer for efficient patch extraction and linear projection.
    *   **Multi-Head Self-Attention (MHSA):** Detailed implementation of Q, K, V projections, scaled dot-product attention, and multi-head concatenation.
    *   **Feed-Forward Network (FFN):** Position-wise FFN with GELU activation.
    *   **Vision Encoder Block:** Follows the standard Transformer Encoder structure with Layer Normalization and residual connections.
    *   **Learnable CLS Token & Positional Embeddings:** Correct handling of sequence tokenization and positional information.
*   **Modular Design:** Code is organized into logical components (`data.py`, `optimizer.py`, `vision_transformer.py`, `vision_encoder.py`, `multi_head_attention.py`, `feed_forward.py`) for clarity and extensibility.
*   **Robust Data Pipeline:** Handles image preprocessing (resizing, cropping, normalization) and provides efficient data loading via `DataLoader`.
*   **Flexible Training Loop:** Features AdamW optimizer, `ReduceLROnPlateau` learning rate scheduler, early stopping, and model checkpointing.
*   **Distributed Training Ready:** Configured for `torch.nn.DataParallel` for multi-GPU training.
*   **Performance Visualization:** Generates learning curves to monitor training and validation loss.