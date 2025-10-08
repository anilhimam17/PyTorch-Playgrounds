import torch

from ViT.src.vision_transformer import VisionTransformer


class VisionEncoder(VisionTransformer):
    """This class inherits from the original VisionTransformer and adapts it for CLIP.
    It overrides the final layer to produce an embedding instead of classification logits."""

    def __init__(self, embed_dims: int = 512):

        # Inheriting the properties of the Super Class
        super().__init__(n_classes=1000)    # Dummy No of Classes which won't be used for the MLP Head

        # Re-purposing the MLP head for Feature Projection into the Uniform Multimodal Embedding Space
        self.mlp_head = torch.nn.Linear(in_features=768, out_features=embed_dims)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Implements the forward propagation of the ViT for Feature rich Embedding generation."""

        return super().forward(X)
