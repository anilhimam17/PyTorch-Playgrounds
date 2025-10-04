import kagglehub
from pathlib import Path
import matplotlib.pyplot as plt

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

    def __init__(self, kaggle_handle: str = DATASET_HANDLE, dataset_path: Path = DATASET_PATH, batch_size: int = 64) -> None:
        
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

        # Transforms
        transforms = v2.Compose([
            v2.Resize(size=256, interpolation=v2.InterpolationMode.BICUBIC),
            v2.RandomResizedCrop(size=(224, 224), antialias=True),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            # v2.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        ])

        # Constructing the Dataset
        self.dataset = Flickr30Dataset(root_dir=self.local_path, csv_path=self.caption_path, transforms=transforms)
        self.dataset = torch.utils.data.DataLoader(
            self.dataset, batch_size=64, shuffle=True, num_workers=0
        )
   