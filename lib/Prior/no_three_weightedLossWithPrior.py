import 差分约束 as DifferentialConnection
import matplotlib.pyplot as plt
from scipy.spatial import distance
from skimage.measure import label, regionprops
import numpy as np
from PIL import Image
import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
import random
import sys
import networkx as nx
from scipy.ndimage import distance_transform_edt
from scipy.spatial import distance
from numpy.polynomial.polynomial import Polynomial
from no_one提取图结构 import segment_and_visualize_vessels,crop_image,visualize_segments_andNo,visualize_segment_graph
""" 作为loss加权 """
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

def f(logits):
    """
    从 2 通道的 logits 提取二值分割结果。
    
    参数:
    - logits: torch.Tensor, 形状为 [B, 2, H, W]，表示网络输出的 logits。
    
    返回:
    - binary_image: numpy.array, 二值分割结果 (0 和 255) 的图像。
    """
    # 1. 对 logits 应用 softmax 操作以得到每个通道的概率
    probabilities = F.softmax(logits, dim=1)  # 输出形状仍为 [B, 2, H, W]
    # 2. 获取前景通道 (channel=1) 的概率
    foreground_prob = probabilities[:, 1, :, :]  # 形状为 [B, H, W]
    # 3. 设定一个阈值 (例如 0.5) 来确定前景和背景
    threshold = 0.5
    binary_image = (foreground_prob > threshold).float() * 255  # 将前景设置为 255，背景为 0
    # 4. 转换为 numpy array，并确保类型为 uint8
    binary_image = binary_image.squeeze(0).cpu().numpy().astype(np.uint8)  # 移除 batch 维度
    return binary_image
def generate_line_pixels(point1, point2):
    """
    给定两个端点，生成在这两个端点之间的所有像素坐标。
    
    参数:
    - point1: 第一个端点的坐标 (x1, y1)。
    - point2: 第二个端点的坐标 (x2, y2)。
    
    返回:
    - pixels: 端点之间所有像素的坐标列表 [(x, y), ...]。
    """
    x1, y1 = point1
    x2, y2 = point2
    pixels = []

    # 使用 Bresenham 算法来生成直线上的所有像素
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    while True:
        # 将当前坐标添加到像素列表中
        pixels.append((x1, y1))
        
        # 如果到达终点，则退出
        if (x1, y1) == (x2, y2):
            break
        
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy

    return pixels
def find_closest_segment_endpoints(graph, target_node, max_distance):
    """
    找到与目标节点的两个端点最近的段。
    
    参数:
    - graph: nx.Graph, 当前的图。
    - target_node: int, 当前要查找的目标节点。
    - max_distance: float, 最大允许的空间距离。
    
    返回:
    - closest_nodes: list, 与目标节点的两个端点最近的段的索引列表。
    """
    target_endpoints = graph.nodes[target_node]['endpoints']
    closest_nodes = []

    # 遍历当前段的两个端点
    for endpoint in target_endpoints:
        min_distance = float('inf')
        closest_node = None

        # 遍历所有其他节点，寻找与当前端点最近的段
        for node, features in graph.nodes(data=True):
            if node == target_node:
                continue

            node_endpoints = features['endpoints']
            # 计算与当前端点的最小距离
            for ep in node_endpoints:
                distance = np.sqrt((endpoint[0] - ep[0])**2 + (endpoint[1] - ep[1])**2)
                if distance < min_distance and distance <= max_distance:
                    min_distance = distance
                    closest_node = node

        # 如果找到了符合条件的最近节点，则添加到结果列表
        if closest_node is not None:
            closest_nodes.append((closest_node, min_distance))

    return closest_nodes
# 先验修复，生成featuremap
def apply_static_graph_rules(pred, graph, min_pixel_count=3, direction_threshold=np.pi/8, max_distance=20,debug=False):
    """
    应用静态图规则对图进行优化，包括连通性和孤立团删除。
    
    参数:
    - graph: nx.Graph, 当前的段图结构。
    - unet_feature_size: tuple, UNet 特征图的大小 (H, W)。
    - min_pixel_count: int, 最小像素数量阈值。
    - direction_threshold: float, 方向相似度阈值。
    - max_distance: float, 最大允许的空间距离。
    
    返回:
    - updated_graph: nx.Graph, 经过规则优化后的图。
    - feature_map: np.array, 更新后的 feature_map。
    """
    # 1. 初始化空的 feature_map
    unet_feature_size = (pred.shape[0],pred.shape[1])
    feature_map = np.zeros(unet_feature_size)
    
    # 复制图用于修改
    updated_graph = graph.copy()
    
    # 规则 1：连通规则 - 连接空间上接近且方向相似的段
    for node_i, features_i in updated_graph.nodes(data=True):
        # 2. 顶点获取坐标
        point1, point2 = features_i['endpoints']
        closest_nodes = find_closest_segment_endpoints(updated_graph, node_i, max_distance)
        
        for closest_node, _ in closest_nodes:
            k_i = features_i['direction_k']
            k_closest = updated_graph.nodes[closest_node]['direction_k']
            
            # 2. 顶点获取坐标
            point3, point4 = updated_graph.nodes[closest_node]['endpoints']

            # 检查方向相似性
            if abs(k_i - k_closest) < direction_threshold:
                updated_graph.add_edge(node_i, closest_node, weight=1)
                if(debug):
                    print(f"添加边: ({node_i}, {closest_node}) - 符合连通条件")

                # 3. 拟合直线路径
                pixels1 = generate_line_pixels(point1, point3)
                pixels2 = generate_line_pixels(point2, point3)
                pixels3 = generate_line_pixels(point1, point4)
                pixels4 = generate_line_pixels(point2, point4)
                
                # 遍历所有像素列表，更新 feature_map
                for pixels in [pixels1, pixels2, pixels3, pixels4]:
                    for (x, y) in pixels:
                        if 0 <= x < feature_map.shape[0] and 0 <= y < feature_map.shape[1]:
                            feature_map[x, y] = 1.5
                            if(debug):
                                print(f"设置权重: feature_map[{x}, {y}] = 1.5")  # 打印日志

    # 规则 2：删除孤立团 - 删除不满足条件的孤立段
    for node, features in list(updated_graph.nodes(data=True)):
        if updated_graph.degree(node) == 0:
            if features['pixel_count'] < min_pixel_count:
                updated_graph.remove_node(node)
                if(debug):
                    print(f"删除孤立节点: {node} - 像素数量低于阈值")

                # 2. 顶点获取坐标
                point1, point2 = features['endpoints']
                # 3. 拟合直线路径
                pixels = generate_line_pixels(point1, point2)
                for pixel in pixels:
                    # 4. 设置对应坐标处权重
                    if 0 <= pixel[0] < feature_map.shape[0] and 0 <= pixel[1] < feature_map.shape[1]:
                        feature_map[pixel[0], pixel[1]] = 0.1
                        if(debug):
                            print(f"设置权重: feature_map[{pixel[0]}, {pixel[1]}] = 0.1")  # 打印日志

    return updated_graph, feature_map
# 应用高斯核
def apply_gaussian_to_pixels(feature_map, sigma=3.0):
    """
    对 feature_map 中的非零像素点应用高斯核扩展权重范围。
    
    参数:
    - feature_map: np.array, 输入的特征图。
    - sigma: float, 高斯核的标准差 (默认: 2.0)。
    
    返回:
    - expanded_feature_map: np.array, 经过高斯扩展后的特征图。
    """
    # 创建一个与 feature_map 大小相同的空图像
    expanded_feature_map = np.zeros_like(feature_map) * 3
    
    # 获取所有非零像素的位置
    non_zero_pixels = np.argwhere(feature_map > 0)
    
    # 对每个非零像素点应用高斯核
    for pixel in non_zero_pixels:
        temp_map = np.zeros_like(feature_map)
        temp_map[pixel[0], pixel[1]] = feature_map[pixel[0], pixel[1]]
        # 应用高斯滤波器在这个点上扩散
        expanded_feature_map += gaussian_filter(temp_map, sigma=sigma)
    
    return expanded_feature_map

# 融合loss
class WeightedLossWithPrior(nn.Module):
    def __init__(self, base_loss_fn,gradDebug=False):
        """
        初始化带有先验信息加权的损失函数。

        参数:
        - base_loss_fn: 基础损失函数 (如 nn.CrossEntropyLoss)。
        """
        super(WeightedLossWithPrior, self).__init__()
        self.base_loss_fn = base_loss_fn
        self.gradDebug = gradDebug

    def forward(self, logits, targets):
        """
        计算带有先验加权的损失值。

        参数:
        - logits: torch.Tensor, 模型输出的 logits，形状为 [B, C, H, W]。
        - targets: torch.Tensor, 真实标签，形状为 [B, H, W]。

        返回:
        - loss: 计算得到的加权损失。
        """
        # 转换输入数据为 PyTorch 张量并设置梯度
        logits_t = torch.from_numpy(logits).float().unsqueeze(0).unsqueeze(0).to('cpu').requires_grad_(True)
        targets_t = torch.from_numpy(targets).long().unsqueeze(0).to('cpu')

        # 使用 no_grad 进行不影响梯度的处理
        with torch.no_grad():
            logits_arr = logits_t.detach().cpu().numpy().squeeze(0).squeeze(0)
            if logits_arr.max() <= 1:
                pred = (logits_arr * 255).astype(np.uint8)

            _, binary = cv2.threshold(pred, 127, 255, cv2.THRESH_BINARY)
            segment_graph = segment_and_visualize_vessels(binary.copy(), debug=False)
            updated_graph, feature_map = apply_static_graph_rules(pred, segment_graph, debug=False)
            expanded_gaussian_feature_map = apply_gaussian_to_pixels(feature_map)

        # 转换为 PyTorch 张量
        expanded_gaussian_feature_map_tensor = torch.tensor(expanded_gaussian_feature_map, dtype=torch.float32, device='cpu').unsqueeze(0).unsqueeze(0)

        # 模拟两个通道的 logits (C=2)
        logits_two_channel = torch.cat([(1 - logits_t), logits_t], dim=1)

        # 计算基础的交叉熵损失
        base_loss = self.base_loss_fn(logits_two_channel, targets_t)

        # 直接使用 feature_map 作为权重
        weights = expanded_gaussian_feature_map_tensor + 1

        # 计算加权后的损失并取平均
        weighted_loss = (base_loss * weights).mean()

        if(self.gradDebug):
            # 反向传播
            weighted_loss.backward() 
            print("\nLogits Gradient:")
            print(logits_t.grad)
        
        return base_loss.mean(), weighted_loss