# ResNet

This repository contains a comprehensive PyTorch implementation of various ResNet architectures (ResNet-34, ResNet-50) trained from scratch on a subset of the ImageNet dataset the ImageNet - 100. The project aims to provide a controlled environment for comparing the training behavior, performance, and overfitting characteristics of these models in contrast to the Vision Transformer.


## 🚀 Overview
The project provides a modular and extensible framework for deep learning image classification. It focuses on:
- Implementing Vision Transformers (ViT-Base/16) from first principles.
- Implementing ResNet-34 (using Basic Blocks) and ResNet-50 (using Bottleneck Blocks).
- Rigorous data preprocessing and splitting strategies (including stratified sampling).
- Standardized training loops with learning rate scheduling and early stopping.
- Conducting a comparative study on a reduced ImageNet dataset (ImageNet-100 or ImageNet-15).


## ✨ Features
- Modular Architecture: Separated components for data handling, model definitions, and training logic (⁠src directory).
- Custom ResNet Implementations: ResNet-34 (Basic Block) and ResNet-50 (Bottleneck Block) with correct stride and skip connections.
- Robust Data Pipeline: Image normalization, resizing, and stratified train/validation splitting.
- Flexible Training: Configurable learning rates, optimizers (AdamW), and learning rate schedulers (⁠ReduceLROnPlateau).
- Distributed Training Support: Configured for ⁠torch.nn.DataParallel on multi-GPU setups (e.g., Kaggle T4 GPUs).
- Performance Monitoring: Generates learning curves to visualize training progress and detect overfitting.