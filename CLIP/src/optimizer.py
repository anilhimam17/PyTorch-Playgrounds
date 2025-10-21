from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import torch

import time
from pathlib import Path

from CLIP.src.clip_lora import PreTrainedCLIP

# ==== Model Checkpoint Path ====
CHECKPOINT_PATH = "./models"

# ==== Local Accelerator Device ====
DEVICE = "mps"


class TrainingLoop:
    """This class handles the training loop for the models."""
    def __init__(
            self, 
            learning_rate: float,
            model: torch.nn.Module | PreTrainedCLIP,
        ):
        
        # Loading the Model Instance for the Training Loop
        self.model = model

        # Optimizer
        self.optim = torch.optim.AdamW(
            params=self.model.parameters(),
            lr=learning_rate
        )

        # Learning Rate Scheduler
        self.lr_schedule = ReduceLROnPlateau(
            optimizer=self.optim, mode="min", patience=3, min_lr=1e-6
        )

        # Creating the Checkpoint Storage Directory
        self.model_dir = Path(CHECKPOINT_PATH)
        if not self.model_dir.exists():
            self.model_dir.mkdir()
    
    def train_model(
            self, 
            epochs: int,
            train_set: DataLoader, 
            valid_set: DataLoader
        ) -> tuple[list[float], list[float]]:
        """Trains the model for the given number of epochs."""

        # Cache Losses
        train_losses = []
        valid_losses = []

        # Mean Training Variables
        mean_loss_train = 0
        mean_loss_valid = 0

        # Training Checkpoint
        best_valid_loss = torch.inf
        patience = 5
        patience_counter = 0

        # Loss Function
        loss_fn = torch.nn.CrossEntropyLoss()

        # Training Loop
        print("The training process has started")
        for i in range(epochs):

            # Average Epoch Time tracking
            start = time.time()

            # ==== Training Step ====

            # Completing a single epoch
            for batch in train_set:

                # Moving the batches to GPU
                image_batch = batch["image"]
                text_batch = batch["caption"]
            
                # Training Step
                logits = self.model(image_batch, text_batch)

                # Constrastive Learning Labels: Indices of the respective encoder encodings
                labels = torch.arange(logits.shape[0], device=DEVICE)

                # Loss Calculation
                image_to_text_loss = loss_fn(logits, labels)
                text_to_image_loss = loss_fn(logits.T, labels)
                train_loss = (image_to_text_loss + text_to_image_loss) / 2.0
                mean_loss_train += train_loss.item()

                # Backpropagation
                self.optim.zero_grad()
                train_loss.backward()
                self.optim.step()

            # Completion of Epoch
            end = time.time()

            # ==== Validation Step ====

            # Turning on the Eval mode on the model for the BN-Layers
            self.model.eval()
            with torch.no_grad():
                for batch in valid_set:

                    # Moving the batches to GPU
                    image_batch = batch["image"]
                    text_batch = batch["caption"]

                    # Validation Calculation
                    logits = self.model(image_batch, text_batch)

                    # Constrative Learning Labels
                    labels = torch.arange(logits.shape[0], device=DEVICE)

                    # Loss Calculation
                    image_to_text_loss = loss_fn(logits, labels)
                    text_to_image_loss = loss_fn(logits.T, labels)
                    valid_loss = (image_to_text_loss + text_to_image_loss) / 2.0
                    mean_loss_valid += valid_loss.item()
            
            # Switching the model back to training mode
            self.model.train()

            # ==== End of Epoch Metrics & Model Checkpointing ====
            mean_loss_train /= len(train_set)
            mean_loss_valid /= len(valid_set)
            time_epoch = end - start

            # Update the LR-Schedule
            self.lr_schedule.step(mean_loss_valid)

            # Updating the Caches
            train_losses.append(mean_loss_train)
            valid_losses.append(mean_loss_valid)

            print(f"Epoch {i + 1}: Train Loss -> {mean_loss_train} | Valid Loss -> {mean_loss_valid} | Time Epoch -> {time_epoch}")

            # Updating the best validation loss so far
            if mean_loss_valid < best_valid_loss:
                best_valid_loss = mean_loss_valid

                # Saving the Model by weights
                torch.save(obj=self.model.state_dict(), f=self.model_dir / "clip.pth")
                print("New best model was saved")
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"No improvement: {patience_counter} / {patience}")
                if patience_counter >= patience:
                    print("Early Stopping")
                    break

            # ==== Reset the Training Loop Metrics ====
            mean_loss_train, mean_loss_valid = 0, 0
        
        return train_losses, valid_losses
