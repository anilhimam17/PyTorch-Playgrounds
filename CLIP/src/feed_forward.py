import torch


class FeedForward(torch.nn.Module):
    """This class implements a simple feedfoward module that connect the attended tokens."""

    def __init__(self, embed_dims: int = 512) -> None:

        # Inheriting all the properties from the super class
        super().__init__()

        # Embedding Dimensions
        self.embed_dims = embed_dims

        # Sequential Block of the Feed Forward Layers
        self.sequential_block = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.embed_dims, out_features=self.embed_dims*4),   # Up Projection: 512 -> 2048
            torch.nn.GELU(),    # Activation
            torch.nn.Linear(in_features=self.embed_dims*4, out_features=self.embed_dims)    # Down Projection: 2048 -> 512
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward propagation of the Input Across the Feed Forward Layer."""

        return self.sequential_block(X)
    