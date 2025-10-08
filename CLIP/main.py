import torch
import os

import matplotlib.pyplot as plt
from pathlib import Path

from CLIP.src.data import DataHandler
from CLIP.src.clip import CLIP
from CLIP.src.optimizer import TrainingLoop


# Disable Tokenizer Parallelism to avoid deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ==== Paths ====
ASSETS = Path("./assets")

# ==== Global Variable for Program Configuration ====
BATCH_SIZE = 64
EPOCHS = 2
LEARNING_RATE = 1e-4
DEVICE = torch.accelerator.current_accelerator()


# The Main Function
def main() -> None:
    print("Hello, CLIP")
    
    # Loading the Data Handler
    data_handle = DataHandler()

    # Viewing the sizes of the indices for Training and Validation Splits
    print("Len of Train Indices: ", data_handle.train_idx.shape)
    print("Len of Valid Indices: ", data_handle.valid_idx.shape)

    loaded_train, loaded_valid = data_handle.construct_dataset(batch_size=BATCH_SIZE)
    print("Len of Train Dataset: ", len(data_handle.train_dataset))
    print("Len of Valid Dataset: ", len(data_handle.valid_dataset))

    print("No of Batches in Train Loaded: ", len(loaded_train), " -> ", len(loaded_train) * BATCH_SIZE)
    print("No of Batches in Valid Loaded: ", len(loaded_valid), " -> ", len(loaded_valid) * BATCH_SIZE)

    # Initialising the CLIP model
    first_clip = CLIP().to(DEVICE)

    # Initialsing the Training Loop
    optimizer = TrainingLoop(learning_rate=LEARNING_RATE, model=first_clip)

    # Training the Model
    train_losses, valid_losses = optimizer.train_model(
        epochs=EPOCHS, train_set=loaded_train, valid_set=loaded_valid
    )

    # Plotting the Chart
    ASSETS.mkdir(exist_ok=True)

    plt.title("CLIP Learning Curve")
    plt.plot(train_losses, c="g", ls="-", label="Training Loss")
    plt.plot(valid_losses, c="b", ls="--", label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Losses")
    plt.legend(loc="upper right")
    plt.grid()
    plt.savefig(ASSETS / "clip_learning_curve.jpg")
    plt.show()


# Driver code
if __name__ == "__main__":

    # The Main Function
    main()
