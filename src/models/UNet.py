import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class DoubleConv(nn.Module):
    """(conv => BN => ReLU) * 2"""
    def __init__(self, in_ch, out_ch, mid_ch=None):
        super().__init__()
        if not mid_ch:
            mid_ch = out_ch
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.pool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv. Uses bilinear upsampling by default.

    NOTE: fixed channel arithmetic: after upsampling x1 (in_ch channels) we concatenate
    the encoder feature x2 (out_ch channels) -> conv input channels = in_ch + out_ch
    """
    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()
        self.bilinear = bilinear
        if bilinear:
            # simple upsample; channel count of x1 remains in_ch
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            conv_in = in_ch + out_ch
            self.conv = DoubleConv(conv_in, out_ch, mid_ch=conv_in // 2)
        else:
            # ConvTranspose2d that halves channels of x1 to match expected concat shape
            # Here we make x1 -> in_ch//2 so that after concat with x2 (out_ch) we get in_ch//2 + out_ch
            # but to keep it simple set transpose to produce out_ch channels then conv gets out_ch + out_ch
            self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
            conv_in = out_ch + out_ch
            self.conv = DoubleConv(conv_in, out_ch)

    def forward(self, x1, x2):
        # x1: decoder feature (to be upsampled), x2: encoder skip connection
        x1 = self.up(x1)
        # pad x1 to the size of x2 (in case of odd sizes)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, features: List[int] = None, bilinear=True):
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512]
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, features[0])
        self.down1 = Down(features[0], features[1])
        self.down2 = Down(features[1], features[2])
        self.down3 = Down(features[2], features[3])

        factor = 2 if bilinear else 1
        self.down4 = Down(features[3], features[3] * factor)

        self.up1 = Up(features[3] * factor, features[3], bilinear)
        self.up2 = Up(features[3], features[2], bilinear)
        self.up3 = Up(features[2], features[1], bilinear)
        self.up4 = Up(features[1], features[0], bilinear)
        self.outc = OutConv(features[0], n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def logits_to_mask(logits, threshold=0.5):
        """
        returns integer mask:
        - for single-channel output -> binary mask (B,H,W) with 0/1 using threshold
        - for multi-channel (C>1) -> class ids (B,H,W) via argmax (0=background,1=foreground)
        """
        '''if logits.shape[1] == 1:
            probs = torch.sigmoid(logits)
            return (probs > threshold).long().squeeze(1)'''
        probs = torch.softmax(logits, dim=1)
        return torch.argmax(probs, dim=1)

    def predict_mask(self, x, threshold=0.5):
        logits = self.forward(x)
        return UNet.logits_to_mask(logits, threshold=threshold)


# Dice Loss for binary segmentation
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        # preds: logits -> apply sigmoid
        preds = torch.sigmoid(preds)
        preds = preds.contiguous().view(preds.shape[0], -1)
        targets = targets.contiguous().view(targets.shape[0], -1).float()

        intersection = (preds * targets).sum(dim=1)
        union = preds.sum(dim=1) + targets.sum(dim=1)
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        loss = 1 - dice
        return loss.mean()


# Utility: initialize weights
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight)
            if getattr(m, 'bias', None) is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
