import torch


class MyGPT(torch.nn.Module):
    """Class implments the Generatively Pretrained Transformer from scratch using PyTorch."""
    def __init__(self, vocab: list[str] = [], n_embd: int = 32) -> None:
        super().__init__()

        # Hyperparameters
        self.vocab_size = len(vocab)

        # Tokenization Maps
        self.encode_map = {ch: i for i, ch in enumerate(vocab)}
        self.decode_map = {i: ch for i, ch in enumerate(vocab)}

        # Embedding Layers
        self.initial_embed = torch.nn.Embedding(self.vocab_size, n_embd)

        # Linear Layers
        self.first_linear = torch.nn.Linear(n_embd, self.vocab_size)

    def encode(self, input_string: str = "") -> list[int]:
        """Encode operation for the simple character level tokenizer."""
        return [self.encode_map[ch] for ch in input_string]

    def decode(self, input_seq: list[int] = []) -> str:
        """Invert operation for the simple character level tokenizer."""
        return "".join([self.decode_map[token] for token in input_seq])
    
    def forward(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Implements the forward propagation of the model."""
        
        embed_score = self.initial_embed(X)
        logits = self.first_linear(embed_score)
        B, T, C = logits.shape
        logits = logits.view((B * T), C)
        y = y.view(B * T)
        loss = torch.nn.functional.cross_entropy(input=logits, target=y)

        return loss
