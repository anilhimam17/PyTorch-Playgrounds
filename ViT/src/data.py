from torchvision.datasets import DatasetFolder, ImageFolder
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import torch

import matplotlib.pyplot as plt


class DataHandler:
    """This class is responsible for loading the datasets."""
    def __init__(self, root_dir: str = "./reduced_imagenet") -> None:
        self.root_dir = root_dir
        self.norm_means = [0.485, 0.456, 0.406]
        self.norm_stds = [0.229, 0.224, 0.225]

        self.apply_transforms = v2.Compose([
            v2.Resize(size=256, interpolation=v2.InterpolationMode.BICUBIC),  # Maintaining the Aspect Ratio
            v2.CenterCrop(size=(224, 224)),  # Crop the Image to the Subject
            v2.ToImage(),  # Converts PIL Image to Tensor
            v2.ToDtype(torch.float32, scale=True),  # Converting the Dtype for Normalisation
            # v2.Normalize(mean=self.norm_means, std=self.norm_stds),  # Applies Normalisation
        ])

    def load_set(self, set_name: str) -> DatasetFolder:
        """Loads the set by the specified set_name."""

        if set_name == "train":
            dataset = ImageFolder(root=(self.root_dir+"/train"), transform=self.apply_transforms)
        elif set_name == "valid":
            dataset = ImageFolder(root=(self.root_dir+"/valid"), transform=self.apply_transforms)
        else:
            raise UnboundLocalError("Invalid set name provided.")
        
        self.class_names = dataset.classes
        return dataset
    
    def prepare_dataset(self, dataset: DatasetFolder, batch_size: int=64, shuffle: bool=True) -> DataLoader:
        """Returns the prepared and loaded dataset."""

        return DataLoader(
            dataset=dataset, 
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=4
        )
    
    def view_images(self, loaded_set: DataLoader) -> None:
        """Helper function just to view the images."""
        images, targets = next(iter(loaded_set))

        plt.figure(figsize=(12, 10))
        for i in range(20):
            img = images[i].squeeze()
            label = targets[i]

            plt.subplot(5, 4, i + 1)
            plt.title(f"{self.class_names[label]}", fontdict={"size": 7})
            plt.imshow(img.permute(1, 2, 0))
            plt.axis("off")
        
        plt.tight_layout()
        plt.show()