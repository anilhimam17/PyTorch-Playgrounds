import torch

from src.feed_forward import FeedForward
from src.multihead_attention import MultiHeadAttention


class GPTDecoderBlock(torch.nn.Module):
    """Implements a single GPT decoder block comprising the Attention Mechanism and the FeedForward Network."""

    def __init__(self, num_heads: int, n_embd: int, block_size: int) -> None:
        super().__init__()

        # Multihead Attention Block
        self.multihead_attention = MultiHeadAttention(
            num_heads=num_heads, 
            head_size=n_embd//num_heads, 
            block_size=block_size,
            n_embd=n_embd
        )
        # Feedforward Network Block
        self.ffwd = FeedForward(n_embd=n_embd)

        # Layer Normalisation Blocks
        self.ln1 = torch.nn.LayerNorm(n_embd)
        self.ln2 = torch.nn.LayerNorm(n_embd)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Implements the forward propagation of a single GPT decoder block."""
        
        attention_out = self.multihead_attention(self.ln1(X))
        x = attention_out + X
        ffwd_out = self.ffwd(self.ln2(x))
        residual_scores = ffwd_out + x
        return residual_scores