import torch


class SelfAttentionHead(torch.nn.Module):
    """Class implements a single self-attention head."""
    def __init__(self, head_size: int, input_features: int, block_size: int, dropout_rate: float = 0.2) -> None:
        super().__init__()

        # Attention Matrices
        self.queries = torch.nn.Linear(input_features, head_size, bias=False)
        self.keys = torch.nn.Linear(input_features, head_size, bias=False)
        self.values = torch.nn.Linear(input_features, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        # Dropout
        self.dropout = torch.nn.Dropout(p=dropout_rate)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Implements the forward propagation of the Self Attention Head."""
        B, T, C = X.shape

        # Calculating the Queries, Keys and Values as Linear Projections
        queries: torch.Tensor = self.queries(X)
        keys: torch.Tensor = self.keys(X)
        values: torch.Tensor = self.values(X)

        # Calculating the Dot Product of the Queries and Keys for the Attention Pattern
        attention_pattern: torch.Tensor = queries @ keys.transpose(-2, -1) * C ** -0.5
        attention_pattern = attention_pattern.masked_fill(self.tril[:T, :T] == 0, -torch.inf)  # type: ignore
        attention_pattern = torch.nn.functional.softmax(attention_pattern, dim=-1)

        # Regularization of the Attention Patterns
        reg_attention_pattern = self.dropout(attention_pattern)

        # Weighted Sum
        output_attended_embeddings = reg_attention_pattern @ values
        return output_attended_embeddings

