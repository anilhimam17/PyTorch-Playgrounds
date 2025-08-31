import torch


class FeedForward(torch.nn.Module):
    """Implements a simple sequential feedforward 
    network to decide the next token based on attention."""

    def __init__(self, n_embd: int = 64, dropout_rate: float = 0.2) -> None:
        super().__init__()

        self.sequential = torch.nn.Sequential(
            torch.nn.Linear(in_features=n_embd, out_features=4 * n_embd),
            torch.nn.ReLU(),
            torch.nn.Linear(4 * n_embd, n_embd),
            torch.nn.Dropout(p=dropout_rate)
        )

    def forward(self, X) -> torch.Tensor:
        """Implements the forward propagation of the feedforward model."""
        return self.sequential(X)

