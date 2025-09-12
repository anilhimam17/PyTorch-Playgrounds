import torch

from src.residual_block import ResidualLearningBlock


class Resnet50(torch.nn.Module):
    """This class implements the complete Resnet-50 model architecture using ResidualLearningBlocks."""

    initial_fmaps = 64
    initial_kernel_size = 7
    initial_stride = 2
    initial_pool_size = 3
    
    def __init__(self, input_channels: int = 3, fc_size: int = 100) -> None:

        # Loading all the properties from the super class
        super().__init__()

        # Intial Convolution Block
        self.model_conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=input_channels,
                out_channels=self.initial_fmaps,
                kernel_size=self.initial_kernel_size,
                stride=self.initial_stride,
                padding=3   
            ),
            torch.nn.BatchNorm2d(num_features=self.initial_fmaps),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(
                kernel_size=self.initial_pool_size,
                stride=2,
                padding=1
            )
        )

        # No of channel for the deep convolutional layers
        self.no_channels = [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3

        # Constructing the remaining model architecture
        prev_channels = self.no_channels[0]
        prev_out_channels = self.no_channels[0]
        for no_channels in self.no_channels:

            # Appending the Residual Block Sequences into the main Resnet Module
            self.model_conv_layers.append(
                ResidualLearningBlock(
                    in_channels=prev_out_channels,
                    stride=1 if prev_channels == no_channels else 2,
                    no_of_filters=no_channels
                )
            )

            # Updating the no of previous channels
            prev_channels = no_channels
            prev_out_channels = no_channels * 4
        
        # Adding the Average Pool Layer on successful execution
        else:
            self.model_conv_layers.add_module(
                module=torch.nn.AvgPool2d(kernel_size=self.initial_kernel_size),
                name="Global Average Pool"
            )

        # Downstream Layers
        self.fc = torch.nn.Linear(in_features=2048 * 1, out_features=fc_size)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Implements the forward propagation of the complete Resnet Model."""

        avg_pool_scores = self.model_conv_layers(X)
        avg_pool_flatten = torch.nn.Flatten()(avg_pool_scores)
        logits = self.fc(avg_pool_flatten)

        return logits
