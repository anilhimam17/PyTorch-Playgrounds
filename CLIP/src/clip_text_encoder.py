import torch
from transformers import CLIPTokenizerFast

from CLIP.src.text_encoder_block import TextEncoderBlock


# Accelerator Device
DEVICE = torch.accelerator.current_accelerator()


class TextEncoder(torch.nn.Module):
    """This class implements the complete Text Encoder for generating rich embeddings for contrastive learning."""

    def __init__(self, n_layers: int = 12, embed_dims: int = 512, n_heads: int = 8, in_features: int = 512, dropout_rate: float = 0.1) -> None:

        # Inheriting all the properties from the Super Class
        super().__init__()

        # Initialsing the Tokenizer
        self.tokenizer: CLIPTokenizerFast = CLIPTokenizerFast.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

        # Initial Embeddings
        self.token_embeddings = torch.nn.Embedding(
            num_embeddings=self.tokenizer.vocab_size,
            embedding_dim=embed_dims
        )

        # Positional Embeddings
        self.positional_embeddings = torch.nn.Embedding(
            num_embeddings=self.tokenizer.model_max_length,
            embedding_dim=embed_dims
        )

        # Composing the Text Encoder
        self.deep_encoder_blocks = torch.nn.ModuleList(
            [
                TextEncoderBlock(
                    embed_dims=embed_dims, 
                    n_heads=n_heads, 
                    in_features=in_features, 
                    dropout_rate=dropout_rate
                )
                for _ in range(n_layers)
            ]
        )

        # Final Layer Norm
        self.final_ln = torch.nn.LayerNorm(embed_dims)

        # Final Projection Layer
        self.projection_layer = torch.nn.Linear(in_features=embed_dims, out_features=embed_dims)

    def forward(self, text_input: list[str]) -> None:
        """Implements the forward propagation for the entire text encoder."""

        # Tokenizing the Text Inputs
        inputs = self.tokenizer(
            text=text_input,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt"
        )

        # Accessing the Token Sequences and Attention Masks
        input_ids = inputs["input_ids"].to(DEVICE)  # type: ignore
        attention_mask = inputs["attention_mask"].to(DEVICE)  # type: ignore

        # Creating the Position Ids
        position_ids = torch.arange(
            start=0,
            end=self.tokenizer.model_max_length,
            device=DEVICE
        )

        # Generating the Initial Token and Positional Embeddings
        token_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.positional_embeddings(position_ids)

        # Combined Input Embeddings
        input_embeddings = token_embeddings + position_embeddings

        # Forward Propagation through the Layers
        x = input_embeddings
        for block in self.deep_encoder_blocks:
            x = block(x, attention_mask=attention_mask)

        # Final Layer Normalisation
        final_hidden_state = self.final_ln(x)

        # Taking the Embeddings from the EOS token
        eos_token_position = input_ids.argmax(dim=-1)
        eos_features = final_hidden_state[torch.arange(final_hidden_state.shape[0]), eos_token_position]

        # Projection of the EOS features to the Uniform Multimodal Embedding space
        text_projection = self.projection_layer(eos_features)

        return text_projection
