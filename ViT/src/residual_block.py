import torch


class ResidualLearningBlock(torch.nn.Module):
    """Class Implements a Residual Learning Block to build the ResNet."""

    def __init__(self, in_channels: int, stride: int, no_of_filters: int) -> None:

        # Loading all the properties from the super class
        super().__init__()
        
        # Defining the layers for each block
        self.sequential_block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=no_of_filters, kernel_size=1),  # 1x1 Convolution, Stride: 1
            torch.nn.BatchNorm2d(num_features=no_of_filters),
            torch.nn.ReLU(),

            torch.nn.Conv2d(in_channels=no_of_filters, out_channels=no_of_filters, padding=1, kernel_size=3),  # 3x3 Convolution, Stride: 1
            torch.nn.BatchNorm2d(num_features=no_of_filters),
            torch.nn.ReLU(),

            torch.nn.Conv2d(in_channels=no_of_filters, out_channels=no_of_filters*4, kernel_size=1, stride=stride),  # 1x1 Convolution, Stride: Stride
            torch.nn.BatchNorm2d(num_features=no_of_filters*4)
        )

        # Skip Connection Layer
        if stride > 1 or in_channels != no_of_filters * 4:
            self.skip_block = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=in_channels, out_channels=no_of_filters*4, kernel_size=1, stride=stride),
                torch.nn.BatchNorm2d(num_features=no_of_filters*4)
            )
        else:
            self.skip_block = torch.nn.Identity()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Implements the forward propagation of the residual network along with skip connection."""

        # Copying the Inputs for the Block
        inputs = X

        # Block Bottleneck Propagation
        x = self.sequential_block(X)

        # Skip Connection Propagation
        inputs = self.skip_block(inputs)

        # Concatenation of the Skip Connection
        concat = x + inputs

        # Activating the parameters
        concat = torch.nn.ReLU()(concat)

        return concat
