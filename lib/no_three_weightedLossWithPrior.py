import numpy as np
import cv2
import numpy as np
from no_one提取图结构 import segment_and_visualize_vessels
""" 作为loss加权 """
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

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
def connectRepair(pred, endpoint1, endpoint2, node_i, features_i, graph, max_distance, direction_threshold, feature_map, debug=False):
    """
    连接修复函数：在满足距离和方向条件的情况下，连接两个段之间的断点。
    
    参数:
    - endpoint1, endpoint2: tuple, 当前段的两个端点坐标。
    - node_i: int, 当前段的节点索引。
    - features_i: dict, 当前段的特征信息。
    - graph: nx.Graph, 当前的段图结构。
    - max_distance: float, 最大允许的距离。
    - direction_threshold: float, 方向相似度的阈值。
    - feature_map: np.array, 要更新的特征图。
    - debug: bool, 是否启用调试信息。
    
    返回:
    - feature_map: np.array, 更新后的特征图。
    """
    state = False  # 记录是否修补, 用于统计个数
    k_i = features_i['direction_k']  # 当前段的方向

    # 检查是否为“一个点”的情况
    if features_i['length'] == 1:
        # 当前段是一个点，找到最近的其他段的端点
        min_distance = float('inf')
        best_connection = None

        for node_j, features_j in graph.nodes(data=True):
            if node_i == node_j:
                continue

            if 'direction_k' not in features_j:
                continue
            
            k_j = features_j['direction_k']  # 其他段的方向
            
            # 遍历 node_j 的两个端点
            for other_endpoint in features_j['endpoints']:
                distance = np.sqrt((endpoint1[0] - other_endpoint[0])**2 + (endpoint1[1] - other_endpoint[1])**2)
                
                # 检查距离和方向条件
                if distance < min_distance and distance <= max_distance:
                    direction = np.arctan2(other_endpoint[0] - endpoint1[0], other_endpoint[1] - endpoint1[1])
                    if abs(k_i - direction) <= direction_threshold:
                        min_distance = distance
                        best_connection = (endpoint1, other_endpoint, node_i, node_j)

        # 如果找到最佳连接则生成连接线并更新 feature_map
        if best_connection:
            point1, point2, node_i, node_j = best_connection
            line_pixels = generate_line_pixels(point1, point2)

            # 在生成的线中统计前景和背景的比例
            foreground_count = sum(1 for x, y in line_pixels if 0 <= x < pred.shape[0] and 0 <= y < pred.shape[1] and pred[x, y] == 255)
            foreground_ratio = foreground_count / len(line_pixels) if line_pixels else 0

            # 检查前景占比，只有在前景占比 >= 0.5 时才进行修补
            if all(0 <= x < pred.shape[0] and 0 <= y < pred.shape[1] and pred[x, y] == 255 for x, y in line_pixels):
                # print("跳过：完全包含在前景")
                feature_map = feature_map 
            elif foreground_ratio > 0.7:
                # print("跳过：前景占比过低")
                feature_map = feature_map
            else:
                # 对每个像素点进行填充
                for (x, y) in line_pixels:
                    if 0 <= x < feature_map.shape[0] and 0 <= y < feature_map.shape[1]:
                        feature_map[x, y] = 1.5  # 或者使用适当的值进行填充
                # 添加到图中表示已连接
                graph.add_edge(node_i, node_j)
                state = True
        return feature_map, state

    # 如果不是一个点的情况，继续进行普通连接
    # 初始化最小距离和最佳连接信息
    min_distance = float('inf')
    best_connection = None

    # 确定当前段的左右端点
    right_endpoint = endpoint1 if endpoint1[1] > endpoint2[1] else endpoint2
    left_endpoint = endpoint1 if endpoint1[1] <= endpoint2[1] else endpoint2

    # 遍历所有其他段的节点
    for node_j, features_j in graph.nodes(data=True):
        if node_i == node_j:
            continue
        
        ## 度有个范围的再处理.
        # if(graph.degree[node_j] >= 1):
        #     continue

        # 检查是否已经存在连接, 这里目前存在bug, 因为段之间的关系有错误, 有缺少!
        if graph.has_edge(node_i, node_j):
            # if debug:
                # print(f"跳过段 ({node_i}, {node_j}) - 已存在连接")
            continue
        
        if 'direction_k' not in features_j:
            continue
        
        k_j = features_j['direction_k']  # 其他段的方向
        if(abs(k_i-k_j) > direction_threshold):
            continue
        
        # 获取新段的左右端点
        new_right_endpoint = features_j['endpoints'][0] if features_j['endpoints'][0][1] > features_j['endpoints'][1][1] else features_j['endpoints'][1]
        new_left_endpoint = features_j['endpoints'][0] if features_j['endpoints'][0][1] <= features_j['endpoints'][1][1] else features_j['endpoints'][1]
        
        # 检查新段在当前段的左侧还是右侧
        if new_left_endpoint[1] > right_endpoint[1]:  # 新段在当前段右边
            # 尝试连接当前段右端点和新段左端点
            distance = np.sqrt((right_endpoint[0] - new_left_endpoint[0])**2 + (right_endpoint[1] - new_left_endpoint[1])**2)
            if distance > max_distance:
                continue
            
            # 计算缺失的段k和已有的k方向差异
            endpoint_direction = np.arctan2(new_left_endpoint[0] - right_endpoint[0], new_left_endpoint[1] - right_endpoint[1])
            endpoint_direction_diff = abs(k_i - endpoint_direction)
            if endpoint_direction_diff > direction_threshold:
                continue

            # 更新最佳连接
            if distance < min_distance:
                min_distance = distance
                best_connection = (right_endpoint, new_left_endpoint, node_i, node_j)
        
        elif new_right_endpoint[1] < left_endpoint[1]:  # 新段在当前段左边
            # 尝试连接当前段左端点和新段右端点
            distance = np.sqrt((left_endpoint[0] - new_right_endpoint[0])**2 + (left_endpoint[1] - new_right_endpoint[1])**2)
            if distance > max_distance:
                continue
            
            # 计算方向差异
            endpoint_direction = np.arctan2(new_right_endpoint[0] - left_endpoint[0], new_right_endpoint[1] - left_endpoint[1])
            endpoint_direction_diff = abs(k_i - endpoint_direction)
            if endpoint_direction_diff > direction_threshold:
                continue

            # 更新最佳连接
            if distance < min_distance:
                min_distance = distance
                best_connection = (new_right_endpoint, left_endpoint, node_i, node_j)

    # 如果找到合适的连接，则更新 feature_map
    if best_connection:
        point1, point2, node_i, node_j = best_connection
        
        # if debug:
            # print(f"连接段: ({node_i}, {node_j}) 端点: {point1} -> {point2}, 距离: {min_distance}, 方向差: {endpoint_direction_diff}")
        
        # 生成连接线并更新 feature_map
        line_pixels = generate_line_pixels(point1, point2)
        # 统计 line_pixels 中前景和背景的数量
        foreground_count = 0
        total_count = len(line_pixels)
        for x, y in line_pixels:
            # 检查是否在 pred 范围内，并统计前景和背景数量
            if 0 <= x < pred.shape[0] and 0 <= y < pred.shape[1]:
                if pred[x, y] == 255:  # 假设 255 表示前景，0 表示背景
                    foreground_count += 1
        # 计算前景占比
        foreground_ratio = foreground_count / total_count if total_count > 0 else 0

        # 检查前景占比，只有在前景占比 >= 0.5 时才进行修补
        if total_count < 0:
            # print("跳过：线条太短")
            feature_map = feature_map
        elif all(0 <= x < pred.shape[0] and 0 <= y < pred.shape[1] and pred[x, y] == 255 for x, y in line_pixels):
            # print("跳过：完全包含在前景")
            feature_map = feature_map 
        elif foreground_ratio > 0.7:
            # print("跳过：前景占比过低")
            feature_map = feature_map
        else:
            # 对每个像素点进行填充
            for (x, y) in line_pixels:
                if 0 <= x < feature_map.shape[0] and 0 <= y < feature_map.shape[1]:
                    feature_map[x, y] = 1.5  # 或者使用适当的值进行填充
            # 添加到图中表示已连接
            graph.add_edge(node_i, node_j)
            state = True
    return feature_map, state

# 先验修复，生成featuremap
def apply_static_graph_rules(pred, graph, min_pixel_count=3, direction_threshold=np.pi/8, max_distance=15,debug=False):
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
    feature_map = np.zeros_like(pred, dtype=np.float32)
    
    # 复制图用于修改
    updated_graph = graph.copy()
    num = 0
    # 规则 1：连通规则 - 连接空间上接近且方向相似的段
    for node_i, features_i in updated_graph.nodes(data=True):
        # 2. 顶点获取坐标
        if 'endpoints' not in features_i:
            continue
        ## 度有个范围的再处理.
        if(updated_graph.degree[node_i] >= 3):
            # print("度 012处理,其他跳过")
            continue
        point1, point2 = features_i['endpoints']
        
        # 添加一个额外的情况. 如果是一个点的话, 这个点和其他段的最近端点连接,看k是否在direction_threshold中,然后生成直线.
        # 3. 直接修复,在这个函数中同时判断距离和方向, 方向进行优化!
        feature_map, state = connectRepair(pred, point1,point2, node_i, features_i, updated_graph, max_distance, direction_threshold, feature_map, debug=True)
        if state:
            num += 1
            
    # print(f'修复了{num}次')
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

# 调试可视化
def save_feature_map(feature_map, title="Feature Map Visualization"):
    """
    保存 feature_map 到指定路径。

    参数:
    - feature_map: np.array, 需要保存的特征图。
    - save_path: str, 保存特征图的完整文件路径 (如 "path/to/save/feature_map.png")。
    - title: str, 可视化图像的标题 (默认: "Feature Map Visualization")。
    """
    save_path = '/home/pxl/myProject/血管分割/molong-深度插值/molong-utils/featuremap'
    if not save_path.endswith(('.png', '.jpg', '.jpeg')):
        save_path += '.png'
    # 使用 plt.imsave 保存特征图，并应用颜色映射 'viridis' 或其他颜色映射
    plt.imsave(save_path, feature_map, cmap='viridis')
    print(f"Feature map saved to {save_path}")
    
def visualize_heatmap_on_image(feature_map, binary_image, sigma=3.0, alpha=0.6, colormap='jet', title="Heatmap Overlay"):
    """
    生成并显示特征图的热力图效果，应用高斯扩展并叠加在原始二值图像上。
    
    参数:
    - feature_map: np.array, 输入的特征图。
    - binary_image: np.array, 输入的二值图像，用于显示骨架背景。
    - sigma: float, 高斯核的标准差 (默认: 2.0)。
    - alpha: float, 热力图的透明度 (默认: 0.6)。
    - colormap: str, 使用的颜色映射 (默认: 'jet')。
    - title: str, 可视化图像的标题 (默认: "Heatmap Overlay")。
    """
    # 对 feature_map 应用高斯核扩展
    expanded_feature_map = apply_gaussian_to_pixels(feature_map, sigma=sigma)
    print("Max value:", np.max(expanded_feature_map))
    print("Min value:", np.min(expanded_feature_map))
    # 确保二值图像为单通道灰度图
    if len(binary_image.shape) == 3:
        gray_image = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = binary_image

    # 归一化 expanded_feature_map 到 [0, 255]
    normalized_feature_map = cv2.normalize(expanded_feature_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 使用 OpenCV 生成彩色热力图
    heatmap = cv2.applyColorMap(normalized_feature_map, cv2.COLORMAP_JET)

    # 将二值图像转换为 3 通道
    colored_binary = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

    # 叠加热力图在二值图像上
    overlay = cv2.addWeighted(colored_binary, 1 - alpha, heatmap, alpha, 0)

    # 显示最终结果
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()



class WeightedLossWithPrior(nn.Module):
    def __init__(self, base_loss_fn=nn.BCEWithLogitsLoss(reduction='mean'), device='gpu', gradDebug=False):
        """
        初始化带有先验信息加权的损失函数。

        参数:
        - base_loss_fn: 基础损失函数 (如 nn.BCELoss)。
        """
        super(WeightedLossWithPrior, self).__init__()
        self.base_loss_fn = base_loss_fn
        self.gradDebug = gradDebug
        self.device = device

    def forward(self, logits_t, targets_t):
        B, C, H, W = logits_t.shape  
        
        # 初始化特征图列表
        feature_maps = []
        
        # 提取特征图的逻辑
        with torch.no_grad():
            for i in range(B):
                single_logits = logits_t[i, 0, :, :].detach().cpu().numpy()  
                logits_arr = (single_logits * 255).astype(np.uint8) if single_logits.max() <= 1 else single_logits.astype(np.uint8)

                _, binary = cv2.threshold(logits_arr, 127, 255, cv2.THRESH_BINARY)
                segment_graph, skeleton, segments = segment_and_visualize_vessels(binary.copy(), debug=self.gradDebug)
                updated_graph, feature_map = apply_static_graph_rules(logits_arr, segment_graph, debug=False)
                feature_maps.append(torch.tensor(feature_map, dtype=torch.float32).unsqueeze(0))

        expanded_gaussian_feature_map_tensor = torch.stack(feature_maps, dim=0).to(self.device)
        mask = (expanded_gaussian_feature_map_tensor == 1.5)
        mask = mask.expand_as(logits_t)

        targets_t_squeezed = targets_t.squeeze(1)

        # 对应位置的 logits 和 targets
        masked_logits = logits_t[mask].view(-1)
        masked_targets = targets_t_squeezed[mask[:, 0, :, :]].view(-1).float()

        base_loss = self.base_loss_fn(masked_logits, masked_targets)

        if self.gradDebug:
            base_loss.backward() 
            print("\nLogits Gradient:")
            print(logits_t.grad)
        
        return base_loss
    
class RefineWithPrior():
    def __init__(self, device='cuda', gradDebug=False):
        """
        初始化精修类
        
        参数:
        - device: 运行设备
        - gradDebug: 是否开启调试模式
        """
        self.device = device
        self.gradDebug = gradDebug

    def forward(self, logits_t):
        B, C, H, W = logits_t.shape
        
        # 创建输出张量的副本
        refined_logits = logits_t.clone()
        
        # 提取特征图的逻辑
        with torch.no_grad():
            for i in range(B):
                single_logits = logits_t[i, 0].detach().cpu().numpy()
                logits_arr = (single_logits * 255).astype(np.uint8) if single_logits.max() <= 1 else single_logits.astype(np.uint8)

                _, binary = cv2.threshold(logits_arr, 127, 255, cv2.THRESH_BINARY)
                segment_graph, skeleton, segments = segment_and_visualize_vessels(binary.copy(), debug=self.gradDebug)
                updated_graph, feature_map = apply_static_graph_rules(logits_arr, segment_graph, debug=False)
                
                # 将特征图转换为布尔掩码并应用到输出
                feature_mask = torch.tensor(feature_map == 1.5, dtype=torch.bool).to(self.device)
                refined_logits[i, 0][feature_mask] = 1.0

        return refined_logits