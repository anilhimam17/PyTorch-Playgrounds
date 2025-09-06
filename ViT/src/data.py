from torchvision.datasets import DatasetFolder, ImageFolder
from torch.utils.data import DataLoader


class DataHandler:
    """This class is responsible for loading the datasets."""
    def __init__(self, root_dir: str = "./reduced_imagenet") -> None:
        self.root_dir = root_dir

    def load_set(self, set_name: str) -> DatasetFolder:
        """Loads the set by the specified set_name."""

        if set_name == "train":
            dataset = ImageFolder(root=(self.root_dir+"/train"))
        elif set_name == "valid":
            dataset = ImageFolder(root=(self.root_dir+"/valid"))
        else:
            raise UnboundLocalError("Invalid set name provided.")

        return dataset
    
    def prepare_dataset(self, dataset: DatasetFolder, batch_size: int=64, shuffle: bool=True) -> DataLoader:
        """Returns the prepared and loaded dataset."""

        return DataLoader(
            dataset=dataset, 
            batch_size=batch_size,
            shuffle=shuffle
        )
