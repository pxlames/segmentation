from skimage.morphology import skeletonize, binary_dilation, binary_erosion
import numpy as np
import torch
import torch.nn as nn
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
class SkeletonLoss(nn.Module):
    def __init__(self, filter_p = 0.5, loss = 'BCELoss', reduction = 'mean', normalization = 'Sigmoid', dilation_iter = 0, patch = False, **kwargs):
        super(SkeletonLoss, self).__init__()
        self.filter_p = filter_p
        self.patch = patch
        assert loss in ['MSELoss', 'BCELoss', 'DiceLoss'], "Only MSELoss and BCELoss are supported."
        if loss == 'MSELoss':
            self.loss = nn.MSELoss(reduction = reduction)
        elif loss == 'BCELoss':
            self.loss = nn.BCELoss(reduction = reduction)
        if normalization == 'Sigmoid':
            self.normalization = nn.Sigmoid()
        elif normalization == 'Softmax':
            self.normalization = nn.Softmax(dim=1)
        else:
            print('No normalization layer')
        self.reduction = reduction
        self.dilation_iter = dilation_iter
    def forward(self, pred, target, weight=None):
        if weight is not None and weight.shape == pred.shape:
            pred = pred * weight
            target = target * weight
        target = target.type(pred.dtype)
        
        batch_size = pred.shape[0]
        total_loss = torch.Tensor([0.0]).type(pred.dtype).to(pred.device)
        
        # 循环处理每个批次
        for i in range(batch_size):
            curr_pred = pred[i].squeeze(0)    # 移除通道维度
            curr_target = target[i].squeeze(0) # 移除通道维度
            
            # 转换为二值图像
            gt_binary = curr_target.detach().cpu().numpy() > 0.5
            pred_np = curr_pred.detach().cpu().numpy()
            pred_binary = pred_np > self.filter_p
            
            # 计算骨架
            gt_sk = skeletonize(gt_binary)
            pred_sk = skeletonize(pred_binary)
            
            # 计算选择器
            positive_selector = np.logical_xor(np.logical_and(gt_sk, pred_binary), gt_sk)
            negative_selector = np.logical_xor(np.logical_and(pred_sk, gt_binary), pred_sk)
            final_selector = np.logical_or(positive_selector, negative_selector)
            
            # 计算损失
            c_pred = curr_pred[final_selector]
            c_target = curr_target[final_selector]
            
            if len(c_pred) > 0:
                    total_loss += self.loss(c_pred, c_target)
            # 创建新数组并在final_selector位置设置为255
            visualization = np.zeros_like(pred_binary, dtype=np.uint8)
            visualization[final_selector] = 255
            
            '''
            # 保存可视化结果
            plt.figure()
            plt.imshow(visualization, cmap='gray')
            plt.axis('off')
            plt.savefig(f'final_selector_{i}.png')
            # 保存预测结果
            plt.figure()
            plt.imshow(pred_binary, cmap='gray')
            plt.axis('off')
            plt.savefig(f'pred_binary_{i}.png')            # 保存预测结果
            plt.figure()
            plt.imshow(gt_binary, cmap='gray')
            plt.axis('off')
            plt.savefig(f'gt_binar_{i}.png')
            '''
            
        # 计算平均损失
        return total_loss / batch_size