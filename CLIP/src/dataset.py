import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torchvision.transforms import v2

from pathlib import Path


class Flickr30Dataset(torch.utils.data.Dataset):
    """This class is responsible for constructing the custom dataset for Flickr."""

    def __init__(self, root_dir: Path, df: pd.DataFrame, transforms: v2.Compose | None, idx: torch.Tensor) -> None:
        
        # Loading the Properties of the Super Class
        super().__init__()

        # Local path for all the images
        self.local_path = root_dir / "flickr30k_images"
        
        # Accessing the Full Captions DataFrame and finding the Unique Images
        full_captions_df = df
        unique_image_names = full_captions_df["image_name"].unique()

        # Creating the Split for the Set based on Unique Image Names with all 5 captions
        self.split_based_image_names = [unique_image_names[id] for id in idx]
        self.captions_df = full_captions_df[full_captions_df["image_name"].isin(self.split_based_image_names)]

        # Custom Transforms to operate on the Dataset
        self.transforms = transforms

    def __len__(self) -> int:
        """Provides the length of the dataset."""

        return self.captions_df["image_name"].nunique()
    
    def __getitem__(self, index: int) -> dict:
        """Retrieves the ith example from the dataset."""

        # Querying the Dataframe for all the samples of Image
        image_name = self.split_based_image_names[index]
        image_name_subset = self.captions_df[self.captions_df["image_name"] == image_name]

        # Loading the Image
        image_path = self.local_path / str(image_name)
        image_data = Image.open(image_path)

        # Loading the list of Captions
        image_captions = image_name_subset.iloc[:, -1].astype(str).tolist()
        caption_idx = torch.randint(0, len(image_captions), size=(1,))
        image_caption = image_captions[caption_idx[0]]

        # Creating the Dataset Entry
        sample = {
            "image": image_data,
            "caption": image_caption
        }
        
        # If transforms are available
        if self.transforms:
            sample["image"] = self.transforms(sample["image"])

        return sample

    def show_sample_image_text_pairs(self) -> None:
        """Loads a single random sample and displays it."""

        # Creating the Canvas for Images to be Displayed
        plt.figure(figsize=(12, 7))

        # Random sampling 6 images.
        for i in range(6):

            # Randomly Sampling an Example
            random_idx = int(torch.randint(low=0, high=len(self), size=(1,)))
            sample = self[random_idx]
            sample["image"] = sample["image"].permute(dims=[1, 2, 0])

            # Displaying the Image
            ax = plt.subplot(2, 3, i + 1)
            ax.set_title(f"{sample["caption"]}: {sample['image'].shape}", fontdict={"fontsize": 5})
            ax.imshow(sample["image"])
            ax.set_axis_off()
        
        plt.tight_layout()
        plt.show()