import torch

from src.mhsa import MultiHeadSelfAttention
from src.feed_forward import FeedForward


class TextEncoderBlock(torch.nn.Module):
    """This class implements a complete Text Encoder block for the Text Input."""

    def __init__(self, embed_dims: int = 512, n_heads: int = 8, in_features: int = 512, dropout_rate: float = 0.1) -> None:

        # Inheriting all the super class properties
        super().__init__()

        # Composing the MultiHead Self Attention Block
        self.mhsa = MultiHeadSelfAttention(
            embed_dims=embed_dims, n_heads=n_heads, in_features=in_features, dropout_rate=dropout_rate
        )

        # Composing the Feed Forward Block
        self.ffwd = FeedForward(embed_dims=embed_dims)

        # Normalisation Layers
        self.ln1 = torch.nn.LayerNorm(normalized_shape=embed_dims)
        self.ln2 = torch.nn.LayerNorm(normalized_shape=embed_dims)

    def forward(self, X: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Implements the forward propagation of the full text encoder block."""

        # Step - 1
        attention_out = self.mhsa(self.ln1(X), attention_mask=attention_mask)
        residual_attention_scores = attention_out + X

        # Step - 2
        ff_logits = self.ffwd(self.ln2(residual_attention_scores))
        final_residual_scores = ff_logits + residual_attention_scores

        return final_residual_scores

