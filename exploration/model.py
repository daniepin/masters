import torch
import torch.nn as nn


def block(
    in_channels, out_channels, padding=0, kernel_size=3, maxpool=True, mp_stride=1
):  # 2
    if maxpool:
        return nn.Sequential(
            nn.Conv3d(
                in_channels, out_channels, padding=padding, kernel_size=kernel_size
            ),
            nn.BatchNorm3d(out_channels),
            nn.MaxPool3d(2, stride=mp_stride),
            nn.ReLU(),
        )

    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, padding=padding, kernel_size=kernel_size),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(),
    )


class SFCN(nn.Module):
    # Simply Fully Convolutional Neural Network
    # https://pubmed.ncbi.nlm.nih.gov/33197716/

    def __init__(self, input_dim, channels, output_dim, mode="hardbinary") -> None:
        super().__init__()
        self.channels = output_dim
        self.blocks = nn.Sequential()

        self.blocks.add_module(
            "Conv_1", block(in_channels=input_dim, out_channels=channels[0])
        )

        for i in range(0, len(channels) - 2):
            self.blocks.add_module(
                f"Conv_{i + 2}",
                block(in_channels=channels[i], out_channels=channels[i + 1]),
            )

        self.blocks.add_module(
            f"Conv_{i + 3}",
            block(
                in_channels=channels[i + 1], out_channels=channels[-1], maxpool=False
            ),
        )

        if mode == "hardbinary":
            self.fc = nn.Sequential(
                # nn.AvgPool3d(3),  # [5, 6, 5]
                # nn.Dropout3d(0.5),
                nn.Conv3d(channels[-1], output_dim, padding=0, kernel_size=1),
                # nn.LogSoftmax(dim=1),
                nn.AdaptiveAvgPool3d(1),
                # nn.Flatten(),
                # nn.Linear(output_dim, output_dim),
                # nn.Sigmoid(),
            )
        elif mode == "softbinary":
            self.fc = nn.Sequential(
                # nn.AvgPool3d(3),  # [5, 6, 5]
                # nn.Dropout3d(0.5),
                nn.Conv3d(channels[-1], output_dim, padding=0, kernel_size=1),
                # nn.LogSoftmax(dim=1),
                nn.AdaptiveAvgPool3d(1),
                nn.Flatten(),
                # nn.Sigmoid(),
            )

        self.last = nn.Linear(output_dim, output_dim)

    def forward_virtual(self, x):
        x = self.blocks(x)
        x = self.fc(x)
        x = x.view(-1, self.channels)
        return self.last(x), x

    def forward(self, x):
        x = self.blocks(x)
        x = self.fc(x)
        x = self.last(x)
        return x
