""" Full assembly of the parts to form the complete network """

from .unet_parts import *

class UNet_SECOND(nn.Module):
    def __init__(self, n_channels, n_classes, start_filters, bilinear=True):
        super(UNet_SECOND, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, start_filters)
        self.down1 = Down(start_filters, start_filters*2)
        self.down2 = Down(start_filters*2, start_filters*4)
        self.down3 = Down(start_filters*4, start_filters*8)
        factor = 2 if bilinear else 1
        self.down4 = Down(start_filters*8, start_filters*16 // factor)
        self.up1 = Up(start_filters*16, start_filters*8 // factor, bilinear)
        self.up2 = Up(start_filters*8, start_filters*4 // factor, bilinear)
        self.up3 = Up(start_filters*4, start_filters*2 // factor, bilinear)
        self.up4 = Up(start_filters*2, start_filters, bilinear)
        self.outc = OutConv(start_filters, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # alpha = 0.5  # 可调整的权重系数
        # x = self.up1(x5, x4*alpha)
        # x = self.up2(x, x3*alpha)
        # x = self.up3(x, x2*alpha)
        # x = self.up4(x, x1*alpha)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

def test_unet():
    img = torch.ones(1, 3, 224, 224)
    unet = UNet_SECOND(n_channels=3, n_classes=1, start_filters=64)
    
    # 计算模型参数总量
    total_params = sum(p.numel() for p in unet.parameters())
    print(f'模型总参数量: {total_params:,}')
    
    # 计算模型大小(MB)
    model_size = total_params * 4 / (1024 * 1024) # 假设每个参数占4字节
    print(f'模型大小: {model_size:.2f} MB')
    
    assert unet(img).shape == torch.Size([1, 1, 224, 224])
    

# test_unet()