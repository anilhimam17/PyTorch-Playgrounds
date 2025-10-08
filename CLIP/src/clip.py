import torch

from CLIP.src.clip_text_encoder import TextEncoder
from CLIP.src.clip_vision_encoder import VisionEncoder


# Accelerator Param
DEVICE = torch.accelerator.current_accelerator()


class CLIP(torch.nn.Module):
    """This class implements the combined CLIP model for constrastive learning."""

    def __init__(self, embed_dims: int = 512) -> None:

        # Inheriting all the super class properties
        super().__init__()

        # Initialising the Encoders
        self.image_encoder = VisionEncoder()
        self.text_encoder = TextEncoder()

        # Tunable Temperature Parameter for Creativity
        self.temperature = torch.nn.Parameter(torch.tensor(0.07, device=DEVICE))

    def forward(self, image_batch: torch.Tensor, text_batch: list[str]) -> torch.Tensor:
        """Implements the combined forward propagation of the text and image encoders for CLIP.
        
        args:
        - X -> Is a batch of aligned samples from the dataset with Image, Text pairs.
        
        returns:
        - logits -> Logits from the similarity calculation."""

        # Retrieving the Self-Supervised Projections from each encoder
        image_features: torch.Tensor = self.image_encoder(image_batch)      # B, n_patch, 512
        text_features: torch.Tensor = self.text_encoder(text_batch)         # B, T, 512

        # Normalising the Projections
        norm_image = image_features / image_features.norm(dim=-1, keepdim=True)
        norm_text = text_features / text_features.norm(dim=-1, keepdim=True)

        # Clamping the Temperature for Stability
        max_temp = torch.log(torch.tensor(100.0, device=DEVICE))
        self.temperature.data = torch.clamp(self.temperature.data, max=max_temp)

        # Calculating the Similarity Pattern
        similarity_scores = norm_image @ norm_text.transpose(-2, -1) * torch.exp(self.temperature)

        return similarity_scores










