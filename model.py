import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import (
    Residual_Conv_Block,
    Squeeze_Excitation_Block,
    ASPP_Block, # original
    ASPP_Block_, # modified, without first conv layer, cat[x1, x2, x3] for output
    Upsample, # bilinear
    Upsample_, # ConvTranspose2d
    Attention_Block,
)


class ResUnetPP(nn.Module):
    """
    ResUnet++ model
    Ref: https://github.com/rishikksh20/ResUnet
    """
    def __init__(self, input_channel, filters=[32, 64, 128, 256, 512], Original=True) -> None:
        super(ResUnetPP, self).__init__()
        
        # Input Layer
        self.input_layer = nn.Sequential(
            nn.Conv2d(input_channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(), # res connection
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        
        self.input_residual = nn.Sequential(
            nn.Conv2d(input_channel, filters[0], kernel_size=1),
        )
        
        # Encoder
        self.squeeze_excite1 = Squeeze_Excitation_Block(filters[0])
        self.residual_conv_block1 = Residual_Conv_Block(filters[0], filters[1], 2, 1)
        
        self.squeeze_excite2 = Squeeze_Excitation_Block(filters[1])
        self.residual_conv_block2 = Residual_Conv_Block(filters[1], filters[2], 2, 1)
        
        self.squeeze_excite3 = Squeeze_Excitation_Block(filters[2])
        self.residual_conv_block3 = Residual_Conv_Block(filters[2], filters[3], 2, 1)
        
        # Bridge
        if Original:
            self.aspp = ASPP_Block(filters[3], filters[4])
        else:
            self.aspp = ASPP_Block_(filters[3], filters[4])
        
        # Decoder           
        self.attn1 = Attention_Block(filters[2], filters[4], filters[4])
        if Original:
            self.upsample1 = Upsample(2)
        else:
            self.upsample1 = Upsample(2, mode="bilinear")
        self.up_residual_conv_block1 = Residual_Conv_Block(filters[2] + filters[4], filters[3], 1, 1)
        
        self.attn2 = Attention_Block(filters[1], filters[3], filters[3])
        if Original:
            self.upsample2 = Upsample(2)
        else:
            self.upsample2 = Upsample(2, mode="bilinear")
        self.up_residual_conv_block2 = Residual_Conv_Block(filters[1] + filters[3], filters[2], 1, 1)
        
        self.attn3 = Attention_Block(filters[0], filters[2], filters[2])
        if Original:
            self.upsample3 = Upsample(2)
        else:
            self.upsample3 = Upsample(2, mode="bilinear")
        self.up_residual_conv_block3 = Residual_Conv_Block(filters[0] + filters[2], filters[1], 1, 1)
        
        # Output Layer
        self.output_layer = nn.Sequential(
            ASPP_Block(filters[1], filters[0]),
            nn.Conv2d(filters[0], 1, kernel_size=1),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        # Input layer
        x1 = self.input_layer(x)
        x1_res = self.input_residual(x)
        x1 = x1 + x1_res
        
        # Encoder
        x2 = self.squeeze_excite1(x1)
        x2 = self.residual_conv_block1(x2)
        
        x3 = self.squeeze_excite2(x2)
        x3 = self.residual_conv_block2(x3)
        
        x4 = self.squeeze_excite3(x3)
        x4 = self.residual_conv_block3(x4)
        
        # Bridge
        x5 = self.aspp(x4)
        
        # Decoder
        x6 = self.attn1(x3, x5)
        x6 = self.upsample1(x6)
        x6 = torch.cat([x6, x3], dim=1)
        x6 = self.up_residual_conv_block1(x6)
        
        x7 = self.attn2(x2, x6)
        x7 = self.upsample2(x7)
        x7 = torch.cat([x7, x2], dim=1)
        x7 = self.up_residual_conv_block2(x7)
        
        x8 = self.attn3(x1, x7)
        x8 = self.upsample3(x8)
        x8 = torch.cat([x8, x1], dim=1)
        x8 = self.up_residual_conv_block3(x8)
        
        out = self.output_layer(x8)
        
        return out
