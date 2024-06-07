"""
@Description: Modules used by the ReResUNet++ model
@Author: Ken Zh0ng
@date: 2024-06-05
"""

import torch.nn as nn
import torch


class Residual_Conv_Block(nn.Module):
    """
    Ref: https://github.com/rishikksh20/ResUnet
    While the Encoder and Decoder's conv blocks are diff in paper,
    same in repo implementation.
    """
    def __init__(self, input_channel, output_channel, stride, padding=1) -> None:
        super(Residual_Conv_Block, self).__init__()
        
        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_channel),
            nn.ReLU(),
            nn.Conv2d(input_channel, output_channel, 3, stride, padding),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(),
            nn.Conv2d(output_channel, output_channel, 3, 1, 1), # keep size
        )
        
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, 1, stride, 0), # resize channel
            nn.BatchNorm2d(output_channel),
        )
        
    def forward(self, x):
        return self.conv_block(x) + self.conv_skip(x)


class Squeeze_Excitation_Block(nn.Module):
    def __init__(self, channel, reduction=16) -> None:
        super(Squeeze_Excitation_Block, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # Global feature squeeze
        self.fc = nn.Sequential(                # Excitation: learn 'attention' on important features
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c) # squeeze
        y = self.fc(y).view(b, c, 1, 1) # excitation
        return x * y.expand_as(x)       # scale, Apply the channel attention weight to all pixels in each channel
    

class ASPP_Block_(nn.Module):
    """
    Atrous Spatial Pyramidal Pooling
    Rrf: https://github.com/rishikksh20/ResUnet
    with non-linear activation
    use concatenation instead of summation
    3 rates: 6, 12, 18
    """
    def __init__(self, input_channel, output_channel, rate=[6, 12, 18]) -> None:
        super(ASPP_Block_, self).__init__()

        assert len(rate) == 3, "Rate must be a list of 3 integers"
        
        self.aspp_block1 = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, 3, 1, padding=rate[0], dilation=rate[0]),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(output_channel),
        )
        
        self.aspp_block2 = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, 3, 1, padding=rate[1], dilation=rate[1]),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(output_channel),
        )
        
        self.aspp_block3 = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, 3, 1, padding=rate[2], dilation=rate[2]),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(output_channel),
        )
        
        self.output_layer = nn.Conv2d(3 * output_channel, output_channel, 1, 1, 0)
        self._init_weight()
    
    def forward(self, x):
        x1 = self.aspp_block1(x)
        x2 = self.aspp_block2(x)
        x3 = self.aspp_block3(x)
        x = torch.cat([x1, x2, x3], dim=1)
        return self.output_layer(x)
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)           


class ASPP_Block(nn.Module):
    """
    Atrous Spatial Pyramidal Pooling
    Rrf: https://github.com/DebeshJha/ResUNetPlusPlus/tree/master
    without non-linear activation
    use summation to fuse
    4 rates: 1, 6, 12, 18
    """
    def __init__(self, input_channel, output_channel, rates=[1, 6, 12, 18]) -> None:
        super(ASPP_Block, self).__init__()     

        self.aspp_block1 = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, 3, 1, padding=rates[0], dilation=rates[0]),
            nn.BatchNorm2d(output_channel),
        )
        self.aspp_block2 = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, 3, 1, padding=rates[1], dilation=rates[1]),
            nn.BatchNorm2d(output_channel),
        )
        self.aspp_block3 = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, 3, 1, padding=rates[2], dilation=rates[2]),
            nn.BatchNorm2d(output_channel),
        )
        self.aspp_block4 = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, 3, 1, padding=rates[3], dilation=rates[3]),
            nn.BatchNorm2d(output_channel),
        )
        
        self.output_layer = nn.Conv2d(output_channel, output_channel, 1, 1, 0)
        
    def forward(self, x):
        x1 = self.aspp_block1(x)
        x2 = self.aspp_block2(x)
        x3 = self.aspp_block3(x)
        x4 = self.aspp_block4(x)
        x = x1 + x2 + x3 + x4
        return self.output_layer(x)
        

class Upsample(nn.Module):
    """
    Upsample using ConvTranspose2d
    Ref: https://github.com/rishikksh20/ResUnet
    while in original paper, they use nearest interpolation,
    Ref github repo uses bilinear interpolation instead.
    """
    def __init__(self, scale=2, mode="nearest") -> None:
        super(Upsample, self).__init__()
        if mode == "nearest":
            self.upsample = nn.Upsample(scale_factor=scale, mode="nearest", align_corners=True)
        elif mode == "bilinear":
            self.upsample = nn.Upsample(scale_factor=scale, mode="bilinear", align_corners=True)
        else:
            raise ValueError("Invalid mode")
        
    def forward(self, x):
        return self.upsample(x)


class Upsample_(nn.Module):
    """
    Upsample using ConvTranspose2d
    Ref: https://github.com/rishikksh20/ResUnet
    """
    def __init__(self, input_channel, output_channel, kernel, stride) -> None:
        super(Upsample_, self).__init__()
        
        self.Upsample = nn.ConvTranspose2d(
            input_channel, output_channel, kernel_size=kernel, stride=stride
        )
    
    def forward(self, x):
        return self.Upsample(x)


class Attention_Block(nn.Module):
    """
    Attention block
    Ref: https://github.com/rishikksh20/ResUnet
    Ref: https://github.com/DebeshJha/ResUNetPlusPlus/tree/master
    """
    def __init__(self, in_encoder, in_decoder, out_dim) -> None: # out_dim = in_decoder
        super(Attention_Block, self).__init__()
        
        self.conv_encoder = nn.Sequential(
            nn.BatchNorm2d(in_encoder),
            nn.ReLU(),
            nn.Conv2d(in_encoder, out_dim, 3, 1, padding=1),
            nn.MaxPool2d(2, 2), # w, h -> w/2, h/2
        )
        
        self.conv_decoder = nn.Sequential(
            nn.BatchNorm2d(in_decoder),
            nn.ReLU(),
            nn.Conv2d(in_decoder, out_dim, 3, 1, padding=1),
        )
        
        self.conv_attn = nn.Sequential(
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
            nn.Conv2d(out_dim, 1, 1), # c -> 1
        )
        
    def forward(self, x1, x2):
        """
        x1: encoder output, with size (N, C, H, W)
        x2: decoder output, with seze (N, C, H/2, W/2)
        out: attention applied decoder output
        """
        out = self.conv_encoder(x1) + self.conv_decoder(x2)
        out = self.conv_attn(out)
        return out * x2



        