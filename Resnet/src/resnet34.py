import torch

from src.residual_block34 import ResidualBlock34


class Resnet34(torch.nn.Module):
    """This class implements the Resnet-34 model."""

    initial_fmaps = 64
    initial_kernel_size = 7
    initial_stride = 2
    initial_pool_size = 3

    def __init__(self, in_channels: int = 3, fc_size: int = 100, dropout_rate: float = 0.2):

        # Loading all the properties and parameters from the super class
        super().__init__()

        # Initial Layers of the Model conv1
        self.model_modules = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=self.initial_fmaps,
                kernel_size=self.initial_kernel_size,
                stride=self.initial_stride,
                padding=3
            ),
            torch.nn.GroupNorm(
                num_groups=1,
                num_channels=self.initial_fmaps
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(
                kernel_size=self.initial_pool_size,
                stride=2,
                padding=1
            )
        )
        
        # Model Architecture Configurations
        model_config = [(64, 3), (128, 4), (256, 6), (512, 3)]
        prev_num_channels = model_config[0][0]

        # Constructing the rest of the layers
        for layer_idx, (num_channels, num_blocks) in enumerate(model_config):
            
            # Iterating over each of the block sequences
            for block in range(num_blocks):

                # Default stride
                stride = 1

                # If the first block of a sequence check for downsampling
                if block == 0:

                    # Avoiding downsampling for the first layer
                    stride = 2 if layer_idx != 0 else 1

                    # Appending the new block
                    self.model_modules.append(
                        ResidualBlock34(
                            in_channels=prev_num_channels,
                            out_channels=num_channels,
                            stride=stride
                        )
                    )
                # If block in the middle of the sequence default stride
                else:
                    self.model_modules.append(
                        ResidualBlock34(
                            in_channels=prev_num_channels,
                            out_channels=num_channels,
                            stride=stride
                        )
                    )

                # Updating the no of previous channels
                prev_num_channels = num_channels
            
            # Dropout Regularization after each block
            self.model_modules.append(
                torch.nn.Dropout(p=dropout_rate)
            )
        
        # Global Average Pooling
        self.model_modules.append(
            torch.nn.AdaptiveAvgPool2d(output_size=1)
        )

        # Final Downstream Layers
        self.sequential_fc_layers = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Dropout(p=dropout_rate),
            torch.nn.Linear(
                in_features=prev_num_channels,
                out_features=fc_size
            )
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Implements the foward propagation for the entire Resnet-34 model."""

        conv_layer_scores = self.model_modules(X)
        logits = self.sequential_fc_layers(conv_layer_scores)

        return logits
