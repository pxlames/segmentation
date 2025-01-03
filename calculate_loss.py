import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.skeletonloss import SkeletonLoss
from lib.metric_utilities import torch_dice_fn_bce,torch_betti_error_loss
from lib.metrics.CLDiceLoss import soft_dice_cldice
from lib.metrics.topoloss_pd import TopoLossMSE2D
import matplotlib.pyplot as plt
import os

class LossCalculator:
    """损失函数计算器类
    
    根据配置文件中指定的损失函数名称计算相应的损失
    支持的损失函数包括:
    - CrossEntropyLoss
    - DiceLoss 
    - FocalLoss
    - BCELoss
    - SkeletonLoss
    
    使用示例:
    ```python
    # 在训练循环中:
    for epoch in range(num_epochs):
        for batch in dataloader:
            # 前向传播
            pred = model(inputs)
            
            # 计算损失
            loss_dict = loss_calculator.compute_loss(pred, target, epoch)
            
            # 反向传播
            loss_dict['total'].backward()
            optimizer.step()
            
        # 每个epoch结束后绘制损失曲线
        if (epoch + 1) % plot_interval == 0:
            loss_calculator.plot_loss_history(save_dir='./loss_plots')
    ```
    """
    
    def __init__(self, cfg):
        """初始化
        
        Args:
            cfg: 配置文件,包含loss_types和loss_weights等参数
            
        配置文件示例:
        {
            "common": {
                "loss_config": {
                    "loss_types": ["ce", "dice", "focal", "skeleton"],  # 支持的损失函数类型
                    "loss_weights": [0.4, 0.4, 0.1, 0.1]  # 对应的权重
                }
            }
        }
        """
        self.cfg = cfg
        self.loss_types = cfg.get('loss_types', ['ce'])  # 损失函数名称数组
        self.loss_weights = cfg.get('loss_weights', None) # 损失函数权重数组
        
        # 如果没有指定权重,则默认每个损失权重为1
        if self.loss_weights is None:
            self.loss_weights = [1.0] * len(self.loss_types)
            
        # 检查loss_types和loss_weights长度是否一致
        if len(self.loss_types) != len(self.loss_weights):
            raise ValueError('loss_types和loss_weights的长度必须一致')
        
        # 初始化SkeletonLoss
        self.skeleton_loss = SkeletonLoss()
        self.soft_dice_cldice_loss = soft_dice_cldice(iter_=50, alpha=0.5, smooth=1.)
        self.topo_loss_func = TopoLossMSE2D(0.0001, 73)

        # 初始化损失历史记录
        self.loss_history = {loss_type: [] for loss_type in self.loss_types}
        self.loss_history['total'] = []

    def compute_loss(self, pred, target, epoch=0):
        """计算损失值
        
        Args:
            pred: 模型预测结果
            target: 真实标签
            epoch: 当前训练的epoch数
            
        Returns:
            loss_dict: 包含各个损失值和总损失的字典
        """
        loss_dict = {}
        total_loss = torch.zeros(1, device=pred.device)  # 修改为1维tensor
        
        for loss_type, weight in zip(self.loss_types, self.loss_weights):
            if loss_type == 'ce':
                criterion = nn.CrossEntropyLoss()
                loss = criterion(pred, target)
                
            elif loss_type == 'dice':
                loss = 1 - torch_dice_fn_bce(pred, target)
                
            elif loss_type == 'focal':
                loss = self.focal_loss(pred, target)
                
            elif loss_type == 'bce':
                criterion = nn.BCELoss()
                loss = criterion(pred, target)
                
            elif loss_type == 'skeletonLoss':
                loss = self.skeleton_loss(pred, target)
            
            elif loss_type == 'cldice':
                loss = self.soft_dice_cldice_loss(pred, target)
             
            elif loss_type == 'PD':
                loss = self.topo_loss_func(pred, target)
                
            elif loss_type == 'smooth':
                loss = self.smoothness_loss(pred, target, epoch)
                    
            else:
                raise ValueError(f'不支持的损失函数类型: {loss_type}')
            
            # 确保loss是tensor且维度正确
            if not isinstance(loss, torch.Tensor):
                loss = torch.tensor([loss], device=pred.device)
            elif loss.dim() == 0:  # 如果是标量tensor,转换为1维
                loss = loss.unsqueeze(0)
                
            loss_dict[loss_type] = {'value': loss.item(), 'weight': weight}
            total_loss += weight * loss
            
            # 记录损失历史
            self.loss_history[loss_type].append(loss.item())
            
        loss_dict['total'] = total_loss
        self.loss_history['total'].append(total_loss.item())
        
        return loss_dict
    
    def plot_loss_history(self, save_dir):
        """绘制损失历史曲线并保存
        
        Args:
            save_dir: 保存图像的目录
        """
        # 计算需要的行数和列数
        n_losses = len(self.loss_history)
        n_cols = 3  # 每行3张图
        n_rows = (n_losses + n_cols - 1) // n_cols
        
        # 创建一个大的图形
        fig = plt.figure(figsize=(15, 5*n_rows))
        
        # 为每个损失类型创建一个子图
        for idx, loss_type in enumerate(self.loss_history):
            ax = fig.add_subplot(n_rows, n_cols, idx+1)
            ax.plot(self.loss_history[loss_type], label=loss_type)
            ax.set_xlabel('迭代次数')
            ax.set_ylabel('损失值')
            ax.set_title(f'{loss_type}损失曲线')
            ax.legend()
            ax.grid(True)
        
        # 调整子图之间的间距
        plt.tight_layout()
        
        # 确保保存目录存在
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'loss_history.png')
        plt.savefig(save_path)
        plt.close()
        
    def focal_loss(self, pred, target, alpha=0.25, gamma=2.0):
        """计算Focal Loss
        
        Args:
            pred: 预测结果
            target: 真实标签
            alpha: 平衡正负样本的参数
            gamma: 调节难易样本的参数
            
        Returns:
            loss: Focal损失值
        """
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        p = torch.exp(-ce_loss)
        loss = alpha * (1-p)**gamma * ce_loss
        return loss.mean()
    
    def smoothness_loss_v0(self, pred, lambda_smooth=1.0):
        """
        计算平滑约束项。

        参数:
            y (torch.Tensor): 分割网络的输出，形状为 (batch_size, channels, height, width)。
            lambda_smooth (float): 平滑项的权重系数，默认为 1.0。

        返回:
            smooth_loss (torch.Tensor): 平滑约束项的值。
        """
        # 计算 y 的梯度
        grad_x = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])  # 水平方向梯度
        grad_y = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])  # 垂直方向梯度

        # 计算梯度的 L1 范数（绝对值之和）
        smooth_loss = lambda_smooth * (torch.mean(grad_x) + torch.mean(grad_y))

        return smooth_loss

    def smoothness_loss(self, pred, mask, epoch=0, lambda_smooth=1.0, eps=1e-6, search_distance=5):
        """
        计算血管区域内部的平滑约束项，基于 8 方向统计确定血管方向。

        参数:
            pred (torch.Tensor): 分割网络的输出，形状为 (batch_size, channels, height, width)。
            mask (torch.Tensor): 血管区域的掩码，形状为 (batch_size, channels, height, width)。
                                1 表示血管区域，0 表示背景。
            epoch (int): 当前训练的 epoch 数。
            lambda_smooth (float): 平滑项的权重系数，默认为 1.0。
            eps (float): 数值稳定性的小常数，默认为 1e-6。
            search_distance (int): 方向搜索的距离，默认为 5。

        返回:
            smooth_loss (torch.Tensor): 血管区域内部的平滑约束项的值。
        """
        assert mask.shape == pred.shape, "掩码和预测的形状必须相同"
        
        # 计算当前 epoch 的阈值（线性衰减）
        # initial_threshold = 0.5  # 初始阈值
        # min_threshold = 0.2      # 最小阈值
        # decay_epochs = 1200      # 总 epoch 数
        # decay_rate = epoch / decay_epochs  # 衰减率
        # threshold = initial_threshold - (initial_threshold - min_threshold) * decay_rate  # 线性衰减从 0.5 到 0.2
        
        # 定义 8 个方向的偏移量
        directions = [
            (0, 1),   # 0°
            (1, 1),   # 45°
            (1, 0),   # 90°
            (1, -1),  # 135°
            (0, -1),  # 180°
            (-1, -1), # 225°
            (-1, 0),  # 270°
            (-1, 1)   # 315°
        ]

        # 初始化平滑损失
        smooth_loss = torch.tensor(0.0, device=pred.device)

        # 遍历每个像素点
        for i in range(pred.shape[2]):  # 遍历所有行
            for j in range(pred.shape[3]):  # 遍历所有列
                if mask[0, 0, i, j] == 0:  # 只处理血管区域
                    continue

                # 初始化方向统计
                direction_counts = np.zeros(8, dtype=int)

                # 统计 8 个方向的血管像素个数
                for k, (dx, dy) in enumerate(directions):
                    for d in range(1, search_distance + 1):
                        # 计算当前方向的像素坐标
                        x = i + d * dx
                        y = j + d * dy

                        # 检查是否超出 mask 范围
                        if x < 0 or x >= pred.shape[2] or y < 0 or y >= pred.shape[3]:
                            break  # 超出范围，停止该方向的后续统计

                        # 如果当前像素是 mask 外（值为 0），停止该方向的后续统计
                        if mask[0, 0, x, y] == 0:
                            break

                        # 统计血管像素个数
                        direction_counts[k] += 1

                # 找到血管像素个数最多的方向
                best_direction_index = np.argmax(direction_counts)
                dx, dy = directions[best_direction_index]

                # 方向性梯度计算：只计算该方向上两个像素的差值
                # 检查是否超出 pred 范围
                x = i + dx
                y = j + dy
                if 0 <= x < pred.shape[2] and 0 <= y < pred.shape[3]:
                    grad = torch.abs(pred[0, 0, i, j] - pred[0, 0, x, y])
                else:
                    grad = torch.tensor(0.0, device=pred.device)  # 超出范围时梯度为 0

                # 动态阈值筛选：如果梯度差值大于阈值，则计入平滑损失
                if grad > 0.2:
                    smooth_loss += grad

        # 计算有效像素数量
        num_valid_pixels = torch.sum(mask) + eps

        # 计算平滑约束项
        smooth_loss = lambda_smooth * smooth_loss / num_valid_pixels

        return smooth_loss


# if __name__ == '__main__':
    
#         # 创建损失计算器实例
#     cfg = {
#         'loss_types': ['smooth'],
#         'loss_weights': [1.0]
#     }
#     loss_calculator = LossCalculator(cfg)
    
    
#     # 定义完全不平滑的图像矩阵
#     pred_noisy = torch.tensor([[
#         [0.1, 0.9, 0.2, 0.8, 0.5],
#         [0.7, 0.4, 0.6, 0.5, 0.4],
#         [0.2, 0.8, 0.6, 0.7, 0.4],
#         [0.6, 0.1, 0.1, 0.9, 0.2],
#         [0.6, 0.7, 0.4, 0.6, 0.5]
#     ]]).unsqueeze(0)  # 形状为 (1, 1, 5, 5)

#     mask_noisy = torch.tensor([[
#         [0.0, 0.0, 0.0, 1.0, 1.0],
#         [0.0, 0.0, 1.0, 1.0, 1.0],
#         [0.0, 1.0, 1.0, 1.0, 0.0],
#         [1.0, 1.0, 0.0, 0.0, 0.0],
#         [1.0, 0.0, 0.0, 0.0, 0.0]
#     ]]).unsqueeze(0)  # 形状为 (1, 1, 5, 5)
    
    
#         # 定义完全不平滑的图像矩阵
#     pred_noisy2 = torch.tensor([[
#         [0.1, 0.9, 0.2, 0.8, 0.5],
#         [0.7, 0.4, 0.6, 0.5, 0.4],
#         [0.2, 0.8, 0.6, 0.7, 0.4],
#         [0.6, 0.2, 0.1, 0.9, 0.2],
#         [0.6, 0.1, 0.4, 0.6, 0.5]
#     ]]).unsqueeze(0)  # 形状为 (1, 1, 5, 5)




#     loss_smooth = loss_calculator.smoothness_loss(pred_noisy2, mask_noisy,1200)
#     print(f"完全平滑情况下的平滑损失值: {loss_smooth.item()} (预期: 0.0)")
