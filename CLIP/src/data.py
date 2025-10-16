import kagglehub
from pathlib import Path
import pandas as pd

import torch
from torchvision.transforms import v2

from src.dataset import Flickr30Dataset


# ==== Global Variables ====
# Kagglehub Dataset Handle
DATASET_HANDLE = "hsankesara/flickr-image-dataset"

# Local Dataset Path
DATASET_PATH = Path("/Users/narukami/.cache/kagglehub/datasets/hsankesara/flickr-image-dataset/versions/1/flickr30k_images")


class DataHandler:
    """This class is responsible for loading and handling the data for CLIP."""

    def __init__(self, kaggle_handle: str = DATASET_HANDLE, dataset_path: Path = DATASET_PATH, valid_ratio: float = 0.1) -> None:
        
        # Paths to the Dataset Files
        self.local_path = dataset_path
        self.caption_path = self.local_path / "results.csv"
        
        # Downloading the Dataset if not Exists
        if not self.local_path.exists():
            print("Dataset doesn't exist, Downloading from Kagglehub.")
            self.local_path = Path(kagglehub.dataset_download(kaggle_handle))
        else:
            print("Dataset already exists, Skipping Download.")

        # Checking for the Captions if not Exists
        if not self.caption_path.exists():
            print("Missing Captions CSV, Please Download it from Kaggle.")
        else:
            print("Captions exist, proceeding to dataset construction.")

        # Transforms Based on Mode
        self.train_transforms = v2.Compose([
            v2.Resize(size=256, interpolation=v2.InterpolationMode.BICUBIC),
            v2.RandomResizedCrop(size=(224, 224), antialias=True),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        ])

        self.test_transforms = v2.Compose([
            v2.Resize(size=256, interpolation=v2.InterpolationMode.BICUBIC),
            v2.CenterCrop(size=(224, 224)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        ])
        
        # Creating the split for the dataset
        self.full_captions_df = pd.read_csv(self.caption_path, sep="|", encoding="utf-8")
        n_unique_images = self.full_captions_df["image_name"].nunique()
        
        # Calculating the sizes of the splits
        valid_len = int(n_unique_images * valid_ratio)

        # Retrieving the Random Indices for the splits
        full_shuffled_indices = torch.randperm(n=n_unique_images)
        self.train_idx = full_shuffled_indices[:n_unique_images - valid_len]
        self.valid_idx = full_shuffled_indices[n_unique_images - valid_len:]

    def construct_dataset(self, batch_size: int = 64, auto_transforms: bool = True) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """Constructs and loads the preprocessed, transformed and batched dataset."""

        # Constructing the Splits
        if auto_transforms:
            self.train_dataset = Flickr30Dataset(
                root_dir=self.local_path, df=self.full_captions_df, transforms=self.train_transforms, idx=self.train_idx
            )
            self.valid_dataset = Flickr30Dataset(
                root_dir=self.local_path, df=self.full_captions_df, transforms=self.test_transforms, idx=self.valid_idx
            )
        else:
            self.train_dataset = Flickr30Dataset(
                root_dir=self.local_path, df=self.full_captions_df, idx=self.train_idx, transforms=None
            )
            self.valid_dataset = Flickr30Dataset(
                root_dir=self.local_path, df=self.full_captions_df, idx=self.valid_idx, transforms=None
            )
        
        # Loading an Iterable for the Splits
        loaded_train_dataset = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True
        )
        loaded_valid_dataset = torch.utils.data.DataLoader(
            self.valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True
        )

        return loaded_train_dataset, loaded_valid_dataset
