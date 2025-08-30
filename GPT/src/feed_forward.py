import torch


class FeedForward(torch.nn.Module):
    """Implements a simple sequential feedforward 
    network to decide the next token based on attention."""

    def __init__(self, layer_units: int = 128, n_embd: int = 64) -> None:
        super().__init__()

        self.sequential = torch.nn.Sequential(
            torch.nn.Linear(in_features=n_embd, out_features=layer_units),
            torch.nn.ReLU()
        )

    def forward(self, X) -> torch.Tensor:
        """Implements the forward propagation of the feedforward model."""
        return self.sequential(X)

