import torch

from ViT.src.vision_encoder import VisionEncoder


class VisionTransformer(torch.nn.Module):
    """This class implements the complete Vision Transformer from scratch."""

    def __init__(
            self, n_classes: int, n_layers: int = 12, image_size: int = 224, 
            patch_size: int = 16, embed_dims: int = 768, n_heads: int = 12, 
            in_features: int = 768, dropout_rate: float = 0.2
        ) -> None:

        # Loading all the properties from the Super Class
        super().__init__()

        # Vision Transformer Properties
        self.n_patches = image_size // patch_size

        # Initial Convolution Layer
        self.patch_conv = torch.nn.Conv2d(
            in_channels=3,
            out_channels=embed_dims,
            kernel_size=patch_size,
            stride=patch_size
        )

        # Positional Embeddings
        self.positional_embeddings = torch.nn.Embedding(
            num_embeddings=(self.n_patches ** 2) + 1,
            embedding_dim=embed_dims
        )

        # Vision Transformer Encoder Blocks
        self.deep_encoder_blocks = torch.nn.Sequential(
            *[
                VisionEncoder(
                    embed_dims=embed_dims,
                    n_heads=n_heads,
                    in_features=in_features,
                    dropout_rate=dropout_rate
                )
                for _ in range(n_layers)
            ]
        )

        # Final Layer Norm before MLP Head
        self.final_ln = torch.nn.LayerNorm(embed_dims)

        # MLP Head
        self.mlp_head = torch.nn.Linear(
            in_features=embed_dims,
            out_features=n_classes,
            bias=True
        )

        # Learnable Class Token a small random value
        self.cls_token = torch.nn.Parameter(
            data=torch.randn(1, 1, embed_dims) * 0.02
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Implements the forward propagation of the complete Vision Transformer."""

        # Initial Conv
        image_conv: torch.Tensor = self.patch_conv(X)  # Batch_Size, 768, 14, 14

        # Patch Embeddings
        image_tensors: torch.Tensor = image_conv.flatten(start_dim=-2, end_dim=-1)  # Batch_Size, 768, 196
        image_patches: torch.Tensor = image_tensors.permute(dims=[0, 2, 1])         # Batch_Size, 196, 768

        # Prepending the Learnable CLS Token
        image_patches = torch.cat(
            [
                self.cls_token.expand(X.shape[0], -1, -1),
                image_patches
            ],
            dim=1
        )

        # Position Embedding
        pos_scores = self.positional_embeddings(
            torch.arange(
                start=0, 
                end=(self.n_patches ** 2) + 1,
                device=torch.accelerator.current_accelerator()
            )
        )

        # Image Patch Embeddings
        image_patch_embeddings = image_patches + pos_scores

        # Deep Vision Encoder blocks
        deep_logits = self.deep_encoder_blocks(image_patch_embeddings)

        # MLP Head
        final_logits = self.mlp_head(self.final_ln(deep_logits[:, 0]))

        return final_logits