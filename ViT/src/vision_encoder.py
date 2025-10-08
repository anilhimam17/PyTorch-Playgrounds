import torch

from ViT.src.feed_forward import FeedForward
from ViT.src.multi_head_attention import MultiHeadSelfAttention


class VisionEncoder(torch.nn.Module):
    """This class implements the complete Vision Encoder block for the Vision Transformer."""

    def __init__(self, embed_dims: int = 768, n_heads: int = 12, in_features: int = 768, dropout_rate: float = 0.2):

        # Loading all the properties from the Super Class
        super().__init__()

        # MultiHead Self-Attention Block
        self.mhsa_block = MultiHeadSelfAttention(
            embed_dims=embed_dims,
            n_heads=n_heads,
            in_features=in_features,
            dropout_rate=dropout_rate
        )

        # Feedfoward Block
        self.ff_block = FeedForward(embed_dims=embed_dims)

        # Normalization Layers
        self.ln1 = torch.nn.LayerNorm(embed_dims)
        self.ln2 = torch.nn.LayerNorm(embed_dims)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Implements the complete forward propagation of a single Vision Encoder block."""

        # MultiHead Self-Attention Embeddings Calculation
        attention_out = self.mhsa_block(self.ln1(X))
        residual_attention_scores = attention_out + X

        # Position-Wise Feedfoward Calculation
        ff_logits = self.ff_block(self.ln2(residual_attention_scores))
        final_residual_scores = ff_logits + residual_attention_scores

        return final_residual_scores
