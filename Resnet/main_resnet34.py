import torch

from pathlib import Path
import matplotlib.pyplot as plt

from src.data import DataHandler
from src.resnet34 import Resnet34
from src.optimizer import TrainingLoop


# Learning Curve Asset Path
ASSET_PATH = Path("./assets")

# Accelerator Device
DEVICE = torch.accelerator.current_accelerator()


def main():
    """The main function for the Vision Transformer."""
    print("Hello, Resnet-34")

    data_handle = DataHandler()

    # Loading the sets from disk
    train_set = data_handle.load_set("train")
    valid_set = data_handle.load_set("valid")

    # Preparing the sets
    train_prep = data_handle.prepare_dataset(train_set)
    valid_prep = data_handle.prepare_dataset(valid_set)

    # Loading the Model
    first_resnet_34 = Resnet34().to(device=DEVICE)
    
    # Loading the Training Loop Handler
    optimizer = TrainingLoop(learning_rate=1e-3, model=first_resnet_34)
    
    # Training the model
    train_losses, valid_losses = optimizer.train_model(30, train_prep, valid_prep)

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