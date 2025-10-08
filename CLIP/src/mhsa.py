import torch


class MultiHeadSelfAttention(torch.nn.Module):
    """This class implements the MultiHead Self-Attention Mechanism for the Encoder only Transformer.
    
    args:
    - embed_dims -> The Embedding Dimensions of the Text Encoder itself, in accordance to the paper set to 512.
    - n_heads -> The no of Individual Self Attention Heads working in parallel, in accordance to the paper set to 8.
    - in_features -> The input dimensions from the Text Tokenizer for the Token Sequences, in accordance to the paper set to 512: (vocab_size, 512).

    return:
    - torch.Tensor -> Rich embedding of the Input Text for Contrastive Learning.
    """

    def __init__(self, embed_dims: int = 512, n_heads: int = 8, in_features: int = 512, dropout_rate: float = 0.1) -> None:

        # Loading all the properties from the Super Class
        super().__init__()

        # Instance Variables for MHSA module
        self.embed_dims = embed_dims
        self.n_heads = n_heads
        self.head_size = self.embed_dims // self.n_heads

        # Attention Matrices
        self.full_keys = torch.nn.Linear(in_features=in_features, out_features=self.embed_dims, bias=False)
        self.full_queries = torch.nn.Linear(in_features=in_features, out_features=self.embed_dims, bias=False)
        self.full_values = torch.nn.Linear(in_features=in_features, out_features=self.embed_dims, bias=False)

        # Context Sharing Layer
        self.context_share = torch.nn.Linear(in_features=self.embed_dims, out_features=self.embed_dims)

        # Dropout Layer
        self.dropout_layer = torch.nn.Dropout(p=dropout_rate)

    def forward(self, X: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Carries out the forward propagation for the Text Encoder."""

        # Input Dimensions: B -> Batch Size, T -> Input Sequence Length (76), C -> Tokenised Embedding Dims (512)
        B, T, C = X.shape

        # Scaling Constant for Self Attention
        scaling_const = self.head_size ** -0.5

        # Calculating the Linear Projections against the Entire Layers
        keys: torch.Tensor = self.full_keys(X)          # B, T, 512 @ 512, 512 => B, T, 512
        queries: torch.Tensor = self.full_queries(X)    # B, T, 512 @ 512, 512 => B, T, 512
        values: torch.Tensor = self.full_values(X)      # B, T, 512 @ 512, 512 => B, T, 512

        # Reshaping for MHSA calculation
        keys = keys.reshape((B, T, self.n_heads, self.head_size))           # B, T, 8, 64
        queries = queries.reshape((B, T, self.n_heads, self.head_size))     # B, T, 8, 64
        values = values.reshape((B, T, self.n_heads, self.head_size))       # B, T, 8, 64

        # Perumuting the Tensor Dimensions
        keys_mhsa = keys.permute(dims=[0, 2, 1, 3])             # B, 8, T, 64
        queries_mhsa = queries.permute(dims=[0, 2, 1, 3])       # B, 8, T, 64
        values_mhsa = values.permute(dims=[0, 2, 1, 3])         # B, 8, T, 64

        # Multihead Attention Calculation
        attention_pattern_mhsa: torch.Tensor = queries_mhsa @ keys_mhsa.transpose(-2, -1) * scaling_const     # B, 8, T, 64 @ B, 8, 64, T => B, 8, T, T

        # Applying Attention Padding
        if attention_mask is not None:

            # Finding all the values that are equal to 0 [padding tokens] and setting them to a very small value to be 0ed during softmax
            attention_pattern_mhsa = attention_pattern_mhsa.masked_fill(attention_mask.unsqueeze(1).unsqueeze(2) == 0, -1e-9)

        # Taking Softmax
        attention_pattern = torch.nn.functional.softmax(attention_pattern_mhsa, dim=-1)

        # Weighted Score
        attention_scores = attention_pattern @ values_mhsa     # B, 8, T, T @ B, 8, T, 64 => B, 8, T, 64

        # Reshaping back to the original dims
        attention_scores = attention_scores.permute(dims=[0, 2, 1, 3])          # B, T, 8, 64
        concatenated_heads = attention_scores.flatten(start_dim=-2, end_dim=-1)   # B, T, 512

        # Global Context Sharing
        rich_embeddings = self.context_share(concatenated_heads)

        # Dropout Reg
        reg_embeddings = self.dropout_layer(rich_embeddings)

        return reg_embeddings


