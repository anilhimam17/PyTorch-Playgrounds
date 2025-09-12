import torch


class ResidualBlock34(torch.nn.Module):
    """This class implements the Residual Learning block for Resnet34."""

    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:

        # Loading all the parameters and properties from the parent class.
        super().__init__()

        self.sequential_conv_block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=stride),
            torch.nn.GroupNorm(num_groups=1, num_channels=out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            torch.nn.GroupNorm(num_groups=1, num_channels=out_channels),
        )

        if stride > 1 or in_channels != out_channels:
            self.skip_layer = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride),
                torch.nn.GroupNorm(num_groups=1, num_channels=out_channels)
            )
        else:
            self.skip_layer = torch.nn.Identity()
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Implementation of the forward propagation of the residual learning block."""

        # Conv Layer Prop
        conv_logits = self.sequential_conv_block(X)

        # Skip Layer Prop
        skip_logits = self.skip_layer(X)

        # Concatenating the logits
        logits = conv_logits + skip_logits

        # Activating the logits
        activated_logits = torch.nn.ReLU()(logits)

        return activated_logits
