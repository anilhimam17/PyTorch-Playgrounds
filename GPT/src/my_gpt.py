import torch

from src.multihead_attention import MultiHeadAttention
from src.feed_forward import FeedForward


class MyGPT(torch.nn.Module):
    """Class implments the Generatively Pretrained Transformer from scratch using PyTorch."""
    def __init__(self, vocab: list[str] = [], n_embd: int = 64, block_size: int = 128, attention_heads: int = 4) -> None:
        super().__init__()

        # Hyperparameters
        self.vocab_size = len(vocab)
        self.block_size = block_size

        # Tokenization Maps
        self.encode_map = {ch: i for i, ch in enumerate(vocab)}
        self.decode_map = {i: ch for i, ch in enumerate(vocab)}

        # Embedding Layers
        self.token_embed = torch.nn.Embedding(self.vocab_size, n_embd)
        self.position_embed = torch.nn.Embedding(self.block_size, n_embd)

        # Self Attention Head Blocks
        self.first_multiple_attention_heads = MultiHeadAttention(
            num_heads=attention_heads, head_size=n_embd//attention_heads, block_size=block_size, n_embd=n_embd 
        )

        # First FeedForward Block
        self.first_ffwd = FeedForward(n_embd=n_embd)

        # Linear Layers
        self.last_linear = torch.nn.Linear(128, self.vocab_size)

    def encode(self, input_string: str = "") -> list[int]:
        """Encode operation for the simple character level tokenizer."""
        return [self.encode_map[ch] for ch in input_string]

    def decode(self, input_seq: list[int] = []) -> str:
        """Invert operation for the simple character level tokenizer."""
        return "".join([self.decode_map[token] for token in input_seq])
    
    def forward(self, X: torch.Tensor, y: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Implements the forward propagation of the model."""

        B, T = X.shape

        # Initial Scores
        embed_score = self.token_embed(X)
        pos_score = self.position_embed(torch.arange(T, device=torch.accelerator.current_accelerator()))
        x = embed_score + pos_score

        # Attention Layers
        attention_embed = self.first_multiple_attention_heads(x)

        # Feedforward Layers
        ffwd_scores = self.first_ffwd(attention_embed + x)

        # Deeper Layers
        logits = self.last_linear(ffwd_scores)

        loss = torch.tensor([])
        if y is not None:
            B, T, C = logits.shape
            logits = logits.view((B * T), C)
            y = y.view(B * T)
            loss = torch.nn.functional.cross_entropy(input=logits, target=y)

        return loss, logits
    
    def generate(self, previous_tokens: torch.Tensor, max_tokens: int) -> torch.Tensor:
        """Generates novel tokens based on the previous context."""

        for i in range(max_tokens):
            
            # Clipping the generations to the last block size tokens
            last_block_size = previous_tokens[:, -self.block_size:]

            # Generating the next token
            _, logits = self(last_block_size)

            # Taking the consideration of only the last token
            logits = logits[:, -1, :]

            # Getting the probabilities of the words
            probs = torch.nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, 1)

            previous_tokens = torch.cat((previous_tokens, idx_next), dim=1)

        return previous_tokens
