import torch

from src.my_gpt import MyGPT


class DataPreprocessor:
    """Class handles all of the data pre-processing necessary for the dataset."""

    def __init__(self, data_string: str, device: str = "") -> None:
        self.data_string = data_string
        self.device = device if device else None

    def create_data_tensor(self, model: MyGPT) -> None:
        """Create the data tensor on accelerator device."""
        
        token_list = model.encode(self.data_string)
        self.data_tensor = torch.tensor(
            data=token_list,
            dtype=torch.long
        )
        self.tensor_size = self.data_tensor.size()[0]
    
    def train_valid_test(self, valid_percentage: float = 0.1, test_percentage: float = 0.05) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Peforms the split on the data tensor."""

        size_valid = int(self.tensor_size * valid_percentage)
        size_test = int(self.tensor_size * test_percentage)
        size_train = self.tensor_size - (size_valid + size_test)

        train_set = self.data_tensor[:size_train].clone().to(self.device)
        valid_set = self.data_tensor[size_train : size_train + size_valid].clone().to(self.device)
        test_set = self.data_tensor[size_train + size_valid :].clone().to(self.device)

        return train_set, valid_set, test_set
    
    def get_batch(self, set_: torch.Tensor, batch_size: int, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Batches the data_tensors and provides the pointer to the result."""

        # Creating random batches
        ix = torch.randint(0, set_.size()[0] - block_size, (batch_size,))
        X = torch.stack([set_[i : i + block_size] for i in ix])
        y = torch.stack([set_[i + 1 : i + 1 + block_size] for i in ix])

        return X, y

