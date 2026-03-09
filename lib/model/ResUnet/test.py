import torch
from res_unet import ResUnet, ResidualConv, Upsample

def test_resunet():
    img = torch.ones(1, 1, 224, 224)
    resunet = ResUnet(1)
    
    # 计算模型参数总量
    total_params = sum(p.numel() for p in resunet.parameters())
    print(f'模型总参数量: {total_params:,}')
    
    # 计算模型大小(MB)
    model_size = total_params * 4 / (1024 * 1024) # 假设每个参数占4字节
    print(f'模型大小: {model_size:.2f} MB')
    
    
    resunet(img)
    
    
def test_residual_conv():
    x = torch.ones(1, 64, 224, 224)
    res_conv = ResidualConv(64, 128, 2, 1) 
    assert res_conv(x).shape == torch.Size([1, 128, 112, 112]) 
    

def test_upsample():
    x = torch.ones(1, 512, 28, 28)
    upsample = Upsample(512, 512, 2, 2)
    assert upsample(x).shape == torch.Size([1, 512, 56, 56])
    
test_resunet()