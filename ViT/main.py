from pathlib import Path
import matplotlib.pyplot as plt
import torch

from src.data import DataHandler
from src.vision_transformer import VisionTransformer
from src.optimizer import TrainingLoop


# Accelerator Device
DEVICE = torch.accelerator.current_accelerator()

# Dataset Path
dataset_path = "../reduced_imagenet"

# Learning Curve Plot Path
ASSET_PATH = Path("./assets")


# The main function for the Vision Transformer
def main():
    # Initialising the DataHandler
    data_handle = DataHandler(root_dir=dataset_path)

    # Loading the Dataset
    loaded_train_set = data_handle.load_set("train")
    loaded_valid_set = data_handle.load_set("valid")

    # Preparing the sets
    train_set = data_handle.prepare_dataset(loaded_train_set)
    valid_set = data_handle.prepare_dataset(loaded_valid_set)

    # Initializing the Vision Transformer
    first_vit = VisionTransformer(n_classes=len(data_handle.class_names)).to(device=DEVICE)
    print("Model Architecture")
    print(first_vit, end="\n\n")

    num_params = sum(layer.numel() for layer in first_vit.parameters() if layer.requires_grad)
    print(f"Total no of learnable params: {num_params}")

    # Loading the Training Loop Handler
    optimizer = TrainingLoop(learning_rate=1e-4, model=first_vit)

    # Training and Validating the model
    train_losses, valid_losses = optimizer.train_model(5, train_set=train_set, valid_set=valid_set)

    # Plotting the losses
    plt.figure(figsize=(10, 8))
    plt.title("Learning Curve")
    plt.plot(range(1, len(train_losses) + 1), train_losses, c="b", ls="-.", label="Train Loss")
    plt.plot(range(1, len(valid_losses) + 1), valid_losses, c="g", ls="-*", label="Valid Loss")
    plt.legend(loc="upper right")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    # Storing the Learning Curve
    if not ASSET_PATH.exists():
        ASSET_PATH.mkdir()
    plt.savefig(ASSET_PATH / "learning_curve.png")

    # Rendering the plot
    plt.show()


# ==== Driver Code ====
if __name__ == "__main__":
    main()