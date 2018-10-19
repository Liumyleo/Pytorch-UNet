import torch
import torch.nn as nn


class down(nn.Module):
    def __init__(self, in_ch, out_ch, kernal_size, padding):
        super().__init__()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=(1, kernal_size), stride=(1, 2), dilation=1, padding=(0, padding)),
            # nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.down(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, kernal_size, padding):
        super().__init__()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=(1, kernal_size), stride=(1, 2), padding=(0, padding),
                               output_padding=(0, 1)),
            # nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True))

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return x


class out_conv(nn.Module):
    def __init__(self, in_ch, out_ch, kernal_size, padding):
        super().__init__()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.out_conv = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=(1, kernal_size), stride=(1, 2), padding=(0, padding),
                               output_padding=(0, 1)),
            # nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True))

    def forward(self, x1, x2):
        x1 = self.out_conv(x1)
        x = torch.add(x1, x2)
        return x


class UNet(nn.Module):
    def __init__(self, n_filters):
        super().__init__()
        self.down1 = down(1, n_filters[0], kernal_size=7, padding=3)
        self.down2 = down(n_filters[0], n_filters[1], kernal_size=7, padding=3)
        self.down3 = down(n_filters[1], n_filters[2], kernal_size=7, padding=3)
        self.up1 = up(n_filters[2], n_filters[3], kernal_size=7, padding=3)
        self.up2 = up(n_filters[1]+n_filters[3], n_filters[4], kernal_size=7, padding=3)
        self.out_conv = out_conv(n_filters[0]+n_filters[4], 1, kernal_size=7, padding=3)

    def forward(self, x):

        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)

        x4 = self.up1(x3, x2)
        x5 = self.up2(x4, x1)
        x6 = self.out_conv(x5, x)

        return x6