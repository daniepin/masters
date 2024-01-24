import torch
import torch.nn as nn


def block(
    in_channels, out_channels, padding=0, kernel_size=3, maxpool=True, mp_stride=1
):  # 2
    if maxpool:
        return nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                padding=padding,
                kernel_size=kernel_size,
                bias=False,
            ),
            #nn.BatchNorm3d(out_channels),
            nn.InstanceNorm3d(out_channels),
            nn.MaxPool3d(2, stride=mp_stride),
            nn.ReLU(),
        )

    return nn.Sequential(
        nn.Conv3d(
            in_channels,
            out_channels,
            padding=padding,
            kernel_size=kernel_size,
            bias=False,
        ),
        nn.InstanceNorm3d(out_channels),
        #nn.BatchNorm3d(out_channels),
        nn.ReLU(),
    )


class SFCN(nn.Module):
    # Simply Fully Convolutional Neural Network
    # https://pubmed.ncbi.nlm.nih.gov/33197716/

    def __init__(self, input_dim, channels, output_dim) -> None:
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

        self.adaptive = nn.AdaptiveAvgPool3d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(channels[-1], output_dim)

    def forward_virtual(self, x):
        x = self.blocks(x)
        x = self.adaptive(x)
        # x = x.view(-1, self.channels)
        x = self.flatten(x)
        return self.fc(x), x

    def forward(self, x):
        x = self.blocks(x)
        x = self.adaptive(x)
        # x = x.view(-1, self.channels)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class VOS_SFCN(SFCN):
    def __init__(self, input_dim, channels, output_dim, num_classes, samples, beta, device) -> None:
        super().__init__(input_dim, channels, output_dim)
        self.data_tensor = torch.zeros(num_classes, samples, channels[-1], device=device)

        self.classes_dict = {}
        for i in range(num_classes):
            self.classes_dict[i] = 0

        self.weight_energy = torch.nn.Linear(num_classes, 1, device=device)
        torch.nn.init.uniform_(self.weight_energy.weight)

        self.log_reg_criterion = torch.nn.Sequential(
            torch.nn.Linear(1, 12), torch.nn.ReLU(), torch.nn.Linear(12, 2)
        ).to(device)

        self.I = torch.eye(channels[-1], device=device)

        self.beta = beta
