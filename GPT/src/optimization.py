import torch

from src.data_preprocess import DataPreprocessor
from src.my_gpt import MyGPT


class OptimizationLoop:
    """Class implements the train-valid and test loops."""
    def __init__(self, preprocessor: DataPreprocessor, model: MyGPT, learning_rate: float) -> None:
        self.preprocessor = preprocessor
        self.model = model
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

    def train(
            self, epochs: int, train_set: torch.Tensor, valid_set: torch.Tensor,
            batch_size: int, block_size: int) -> None:
        """Implements the PyTorch Training Loop for the model."""

        mean_train_loss = 0
        mean_valid_loss = 0
        
        for i in range(epochs):
            sample_train_X, sample_train_y = self.preprocessor.get_batch(train_set, batch_size, block_size)
            sample_valid_X, sample_valid_y = self.preprocessor.get_batch(valid_set, batch_size, block_size)

            # Training Step
            loss_train, _ = self.model(sample_train_X, sample_train_y)
            self.optimizer.zero_grad()
            loss_train.backward()
            self.optimizer.step()

            # Validation Step
            with torch.no_grad():
                loss_valid, _ = self.model(sample_valid_X, sample_valid_y)

            mean_train_loss += loss_train
            mean_valid_loss += loss_valid
            if i % 100 == 0:
                print(f"Loss at {i}th Epoch -> Train Set {mean_train_loss / 100} | Valid Set {mean_valid_loss / 100}")
                mean_train_loss, mean_valid_loss = 0, 0


