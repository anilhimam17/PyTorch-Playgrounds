import torch


class FeedForward(torch.nn.Module):
    """This class implements a position-wise feed-forward network applied to each token in the sequence."""

    def __init__(self, embed_dims: int = 768) -> None:

        # Inheriting all the properties of the Super Class
        super().__init__()

        # Sequential Feedfoward Block
        self.feed_forward_block = torch.nn.Sequential(
            torch.nn.Linear(in_features=embed_dims, out_features=embed_dims * 4, bias=True),  # Up-Projection from Attention Embeddings
            torch.nn.GELU(),
            torch.nn.Linear(in_features=embed_dims * 4, out_features=embed_dims, bias=True),  # Down-Projection for Residual Connection
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Implements the forward propagation of the simple feedfoward block.
        
        args:
        - X: torch.Tensor -> Batch_Size, N_Patch, Embedding Dims
        
        returns:
        - torch.Tensor -> Batch_Size, N_Patch, Embedding Dims"""

        return self.feed_forward_block(X)
