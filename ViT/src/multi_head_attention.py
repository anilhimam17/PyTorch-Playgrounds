import torch


class MultiHeadSelfAttention(torch.nn.Module):
    """This class implements the MultiHead Self-Attention Mechanism that is central to Transformers."""

    def __init__(self, embed_dims: int = 768, n_heads: int = 12, in_features: int = 768, dropout_rate: float = 0.2) -> None:
        
        # Loading all the properties from the Super Class
        super().__init__()

        # Instance Variables of the MHSA
        self.embed_dims = embed_dims
        self.n_heads = n_heads

        # Head Size
        self.head_size = self.embed_dims // self.n_heads

        # Attention Matrices
        self.queries_full = torch.nn.Linear(in_features=in_features, out_features=self.embed_dims, bias=False)
        self.keys_full = torch.nn.Linear(in_features=in_features, out_features=self.embed_dims, bias=False)
        self.values_full = torch.nn.Linear(in_features=in_features, out_features=self.embed_dims, bias=False)

        # Context Sharing Layer
        self.context_share = torch.nn.Linear(in_features=in_features, out_features=embed_dims, bias=True)

        # Dropout Layer
        self.dropout = torch.nn.Dropout(p=dropout_rate)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Applies the forward propagation for the entire MultiHead Self-Attention Block.
        
        For implementation efficiency it calculates the Linear Transformations for the Q, K, V on the full input.
        It then permutes the tensors into individual heads to calculate the attention scores and provide the output.
        
        args:
        - X: torch.Tensor -> Expected Batch_Size, N_Patch + 1 [CLS Token], Embedding Dim.
        
        returns:
        - torch.Tensor -> Batch_Size, N_Patch + 1 [CLS Token], Embedding Dim."""

        # Input Dimensions
        B, T, C = X.shape

        # Scaling Constant for attention calculation
        scaling_const = self.head_size ** -0.5

        # Calculating all the Linear Projections
        queries: torch.Tensor = self.queries_full(X)  # 197, 768 @ 768, 768 => (Batch_Size, 197, 768)
        keys: torch.Tensor = self.keys_full(X)        # 197, 768 @ 768, 768 => (Batch_Size, 197, 768)
        values: torch.Tensor = self.values_full(X)    # 197, 768 @ 768, 768 => (Batch_Size, 197, 768)

        # Reshaping the tensor to Self Attention Head Sizes for Calculation
        queries = queries.reshape((B, T, self.n_heads, self.head_size))  # Batch_Size, 197, 12, 64
        keys = keys.reshape((B, T, self.n_heads, self.head_size))        # Batch_Size, 197, 12, 64
        values = values.reshape((B, T, self.n_heads, self.head_size))    # Batch_Size, 197, 12, 64

        # Permuting the Tensors for MultiHead Attention Calculation
        queries_mhsa = torch.permute(input=queries, dims=[0, 2, 1, 3])   # Batch_Size, 12, 197, 64
        keys_mhsa = torch.permute(input=keys, dims=[0, 2, 1, 3])         # Batch_Size, 12, 197, 64
        values_mhsa = torch.permute(input=values, dims=[0, 2, 1, 3])     # Batch_Size, 12, 197, 64

        # MultiHead Attention Pattern Calculation for each Attention Head
        attention_pattern_mhsa = queries_mhsa @ keys_mhsa.transpose(-2, -1) * scaling_const  # 197, 64 @ 64, 197 => Batch_Size, 12, 197, 197
        attention_pattern_mhsa = torch.nn.functional.softmax(attention_pattern_mhsa, dim=-1)

        # Weighted Score
        attended_embeddings_mhsa = attention_pattern_mhsa @ values_mhsa  # 197, 197 @ 197, 64 => Batch_Size, 12, 197, 64

        # Resized Attension Scores
        attended_embeddings = attended_embeddings_mhsa.permute(dims=[0, 2, 1, 3])   # Batch_Size, 197, 12, 64
        attended_embeddings = attended_embeddings.flatten(start_dim=2, end_dim=-1)  # Batch_Size, 197, 768

        # Context Sharing
        rich_embeddings = self.context_share(attended_embeddings)  # 197, 768 @ 768, 768 => 197, 786

        # Dropout Reg for better Generalization of the Attention Scores
        regularized_rich_embeddings = self.dropout(rich_embeddings)

        return regularized_rich_embeddings
