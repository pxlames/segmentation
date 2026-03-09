import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet_parts import *

class UNet_SECOND(nn.Module):
    def __init__(self, n_channels, n_classes, start_filters, bilinear=True):
        super(UNet_SECOND, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Encoder
        self.inc = DoubleConv(n_channels, start_filters)
        self.down1 = Down(start_filters, start_filters*2)
        self.down2 = Down(start_filters*2, start_filters*4)
        self.down3 = Down(start_filters*4, start_filters*8)
        factor = 2 if bilinear else 1
        self.down4 = Down(start_filters*8, start_filters*16 // factor)
        
        # Attention modules
        self.attention4 = DualMaskAttention(start_filters*16 // factor)
        self.attention3 = DualMaskAttention(start_filters*8)
        self.attention2 = DualMaskAttention(start_filters*4)
        self.attention1 = DualMaskAttention(start_filters*2)
        
        # Decoder
        self.up1 = Up(start_filters*16, start_filters*8 // factor, bilinear)
        self.up2 = Up(start_filters*8, start_filters*4 // factor, bilinear)
        self.up3 = Up(start_filters*4, start_filters*2 // factor, bilinear)
        self.up4 = Up(start_filters*2, start_filters, bilinear)
        self.outc = OutConv(start_filters, n_classes)

    def forward(self, x, threshold_mask, cp_mask):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # 使用实际特征图的尺寸来计算掩码尺寸
        t_mask4 = F.interpolate(threshold_mask, size=x5.shape[2:], mode='bilinear', align_corners=True)
        cp_mask4 = F.interpolate(cp_mask, size=x5.shape[2:], mode='bilinear', align_corners=True)
        
        # Decoder with attention
        x5 = self.attention4(x5, t_mask4, cp_mask4)
        x = self.up1(x5, x4)
        
        # 使用当前特征图的尺寸来计算掩码尺寸
        t_mask3 = F.interpolate(threshold_mask, size=x.shape[2:], mode='bilinear', align_corners=True)
        cp_mask3 = F.interpolate(cp_mask, size=x.shape[2:], mode='bilinear', align_corners=True)
        x = self.attention3(x, t_mask3, cp_mask3)
        x = self.up2(x, x3)
        
        # 使用当前特征图的尺寸
        t_mask2 = F.interpolate(threshold_mask, size=x.shape[2:], mode='bilinear', align_corners=True)
        cp_mask2 = F.interpolate(cp_mask, size=x.shape[2:], mode='bilinear', align_corners=True)
        x = self.attention2(x, t_mask2, cp_mask2)
        x = self.up3(x, x2)
        
        # 使用当前特征图的尺寸
        t_mask1 = F.interpolate(threshold_mask, size=x.shape[2:], mode='bilinear', align_corners=True)
        cp_mask1 = F.interpolate(cp_mask, size=x.shape[2:], mode='bilinear', align_corners=True)
        x = self.attention1(x, t_mask1, cp_mask1)
        x = self.up4(x, x1)
        
        logits = self.outc(x)
        return logits

class DualMaskAttention(nn.Module):
    def __init__(self, in_channels):
        super(DualMaskAttention, self).__init__()
        
        # 掩码嵌入网络
        self.mask_embedding = nn.Sequential(
            nn.Conv2d(2, in_channels, kernel_size=1),  # 使用1x1卷积避免尺寸变化
            nn.ReLU(inplace=True)
        )
        
        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels, 1),
            nn.Sigmoid()
        )
        
        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x, threshold_mask, cp_mask):
        # 掩码嵌入
        masks = torch.cat([threshold_mask, cp_mask], dim=1)
        mask_features = self.mask_embedding(masks)
        
        # 通道注意力
        channel_input = torch.cat([x, mask_features], dim=1)
        channel_weight = self.channel_attention(channel_input)
        x = x * channel_weight
        
        # 空间注意力
        diff_mask = (cp_mask - threshold_mask).clamp(0, 1)
        spatial_weight = self.spatial_attention(diff_mask)
        x = x * spatial_weight
        
        return x