from peft import LoraConfig
import torch

import os
from pathlib import Path
import matplotlib.pyplot as plt

from CLIP.src.clip_lora import PreTrainedCLIP
from CLIP.src.data import DataHandler
from CLIP.src.optimizer import TrainingLoop


# Disable Tokenizer Parallelism to avoid deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ==== Paths ====
ASSETS = Path("./assets")

# ==== Global Variable for Program Configuration ====
BATCH_SIZE = 64
EPOCHS = 2
LEARNING_RATE = 1e-4
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# ==== Global Variables ====
LORA_CONFIG = LoraConfig(
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"]
)

# The Main Function
def main():
    print("Hello, LoRA CLIP")

    # Loading the Data Handler
    data_handle = DataHandler()

    # Viewing the sizes of the indices for Training and Validation Splits
    print("Len of Train Indices: ", data_handle.train_idx.shape)
    print("Len of Valid Indices: ", data_handle.valid_idx.shape)

    loaded_train, loaded_valid = data_handle.construct_dataset(batch_size=BATCH_SIZE, auto_transforms=True)
    print("Len of Train Dataset: ", len(data_handle.train_dataset))
    print("Len of Valid Dataset: ", len(data_handle.valid_dataset))

    print("No of Batches in Train Loaded: ", len(loaded_train), " -> ", len(loaded_train) * BATCH_SIZE)
    print("No of Batches in Valid Loaded: ", len(loaded_valid), " -> ", len(loaded_valid) * BATCH_SIZE)

    # Initialising the pretrained and adapted model
    peft_clip = PreTrainedCLIP(lora_config=LORA_CONFIG, device=DEVICE)

    # Verifying the Architecture
    print("Traininable Parameters for the Custom Model:")
    peft_clip.peft_model.print_trainable_parameters()

    # Initialising the Training Loop
    optimizer = TrainingLoop(learning_rate=LEARNING_RATE, model=peft_clip)

    # Training the Model
    train_losses, valid_losses = optimizer.train_model(
        epochs=EPOCHS, train_set=loaded_train, valid_set=loaded_valid
    )

    # Plotting the Chart
    ASSETS.mkdir(exist_ok=True)

    plt.title("LoRA-PEFT CLIP Learning Curve")
    plt.plot(train_losses, c="g", ls="-", label="Training Loss")
    plt.plot(valid_losses, c="b", ls="--", label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Losses")
    plt.legend(loc="upper right")
    plt.grid()
    plt.savefig(ASSETS / "peft_clip_learning_curve.jpg")
    plt.show()


if __name__ == "__main__":
    main()