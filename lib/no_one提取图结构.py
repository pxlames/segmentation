###——————————————————————————————
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

def fill_small_holes(image, min_area=30):
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    inverted_binary = cv2.bitwise_not(binary)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(inverted_binary, connectivity=8)
    filled_img = binary.copy()
    for i in range(1, num_labels):  # 跳过背景
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            filled_img[labels == i] = 255
    return np.array(filled_img)
def crop_image(image_path, position, size,isCrop=True):
    # 打开图像
    cropped_image = Image.open(image_path).convert('L') 
    if(isCrop):
      # 计算裁剪区域的边界
      left = position[0]
      upper = position[1]
      right = left + size[0]
      lower = upper + size[1]
      # 裁剪图像
      cropped_image = cropped_image.crop((left, upper, right, lower))
    return cropped_image

# 标识骨架图中的独立连通组件。
def identify_connected_components(skeleton):
    """
    标识骨架图中的独立连通组件。
    
    :param skeleton: 骨架图像
    :return: 各连通组件的坐标列表
    """
    # 使用标签标记连通组件
    labeled_image, num_labels = label(skeleton, connectivity=2, return_num=True)
    # 提取每个连通组件的坐标,这个地方需要再改进.
    components = [region.coords for region in regionprops(labeled_image)]
    return components

# 对[一个]连通组件进行路径段分割。
def segment_paths_in_component(component, max_segment_length=50):
    """
    对于一个连通组件，将其分割为路径段。
    这里假设一个简单的分段方法，实际中需根据应用调整。
    
    :param component: 单个连通组件的坐标
    :param max_segment_length: 每个段的最大长度
    :return: 该组件内的路径段列表
    """
    
    segments = []  # 存储分割后的路径段
    current_segment = [component[0]]  # 初始化当前段点

    # 遍历组件的每个点，根据距离分段
    for i in range(1, len(component)):
        prev_point = component[i - 1]
        current_point = component[i]
        
        # 计算当前段的长度
        segment_length = np.linalg.norm(np.array(current_point) - np.array(current_segment[0]))

        # 如果当前段的长度超过了最大段长，则将当前段存储并重新开始新的段
        if segment_length > max_segment_length:
            segments.append(current_segment)  # 保存当前段
            current_segment = [prev_point, current_point]  # 开始新段
        else:
            current_segment.append(current_point)  # 继续添加到当前段

    # 最后一个段加入
    segments.append(current_segment)
    
    # for i, segment in enumerate(segments):
    #     print(f"Segment {i}:")
    #     for coord in segment:
    #         print(f"  {coord}")
    #     print("-" * 40)  # 分隔线        
    
    return segments  ## Segment 0:[ 0 30]、[ 0 31]、[ 0 32]; Segment 1; Segment 2; ……

###**************************提取图的结构******************************###
# 区域分段+特征点信息提取+段与段邻接关系添加=真正的Graph
def preprocess_image(binary):
        """
        骨架提取，输入二值化图像，返回骨架图像。
        """
        skeleton = skeletonize(binary)
        return skeleton

def detect_junctions(skeleton):
        """
        改进的交点检测，标记交点而不是删除。
        """
        points = np.argwhere(skeleton)
        junctions = set()
        for point in points:
            x, y = point
            neighbors = skeleton[max(0, x-1):x+2, max(0, y-1):y+2]
            if np.sum(neighbors) > 3:  # 交点判定条件，可根据需求调整
                junctions.add((x, y))
        return list(junctions)

def build_graph_and_segment(skeleton, junctions):
    """
    提取段, 构建图结构
    """
    G = nx.Graph() # 每个像素点就是一个图顶点.
    points = np.argwhere(skeleton)
    # 添加所有节点，并标记是否为交点
    for point in points:
        node = tuple(point) # 坐标是一个元组
        is_junction_value = node in junctions
        G.add_node(node, is_junction=is_junction_value)
    # 对点与点之间添加邻居，避免在交点之间直接连通
    for point in points:
        point = tuple(point) # 当前点
        neighbors = [(point[0] + dx, point[1] + dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1] if (dx, dy) != (0, 0)] # 当前点的所有的邻居
        for n in neighbors:
            if G.has_node(n):
                # 交点连通策略：交点只与相邻段连接，不直接连接其他交点
                if not (G.nodes[point]['is_junction'] and G.nodes[n]['is_junction']):
                    G.add_edge(point, n)
    # 改进环处理：检测图中的环，并确保交点正确分段
    # cycles = list(nx.cycle_basis(G))
    # # 确保每个环在交点处分割成独立段
    # for cycle in cycles:
    #     for node in cycle:
    #         print(f"Processing node: {node}")
    #         if node in G.nodes:
    #             if G.nodes[node]['is_junction']:
    #                 G.remove_node(node)
    #             else:
    #                 print(f"Node {node} exists but is not a junction")
    #         else:
    #             print(f"Node {node} does not exist in the graph")
    
    # 段的构建：其实和添加邻居是一样的,也可以在那里完成
    segments = []
    visited = set() # 是否已经添加过段
    segment_index_map = {}  # 保存每个节点所在的段索引
    junction_assigned_map = {}  # 记录每个交点是否已经分配到段
    def extend_segment(start_node):
        """扩展段，包含普通节点和交点相邻节点"""
        segment = []
        stack = [start_node]
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                segment.append(node)
                segment_index_map[node] = len(segments)  # 标记这个节点属于哪个段
                for neighbor in G.neighbors(node): # 所有邻居
                    if not G.nodes[neighbor]['is_junction'] or neighbor == start_node: # 非交点、?
                        stack.append(neighbor)
                    elif G.nodes[neighbor]['is_junction']:
                        # 如果是交点，检查是否已经被分配到段
                        if neighbor not in junction_assigned_map:
                            # 如果还没有分配过，就将它加入到当前段，并标记为已分配
                            segment.append(neighbor)
                            junction_assigned_map[neighbor] = len(segments)
                            segment_index_map[neighbor] = len(segments)
                        # 无论如何都要将该交点标记为访问过，以免重复加入
                        visited.add(neighbor)
        return segment
    
    # 构建所有段: 
    for node in G.nodes:
        if node not in visited and not G.nodes[node]['is_junction']:
            segment = extend_segment(node)
            if segment:
                segments.append(segment)
    return segments, G, segment_index_map
def assign_pixels_to_nearest_segment(segments, original_image):
    """
    根据原始图像像素，归属到最近的分段，返回分段标签矩阵。
    """
    label_map = np.zeros(original_image.shape, dtype=np.int32)
    dist_map = np.full(original_image.shape, np.inf)
    # 遍历每个段，计算段内骨架点到所有像素的距离
    for idx, segment in enumerate(segments):
        mask = np.zeros_like(original_image, dtype=np.uint8)
        for point in segment:
            if 0 <= point[0] < mask.shape[0] and 0 <= point[1] < mask.shape[1]:
                mask[point[0], point[1]] = 1
        
        # 计算每个像素到当前段的距离
        dist_to_segment = distance_transform_edt(1 - mask)
        # 更新距离和段标签
        update_mask = dist_to_segment < dist_map
        dist_map[update_mask] = dist_to_segment[update_mask]
        label_map[update_mask] = idx + 1  # 确保标签从1开始
    
    # 仅在原始图像中有像素的地方返回标签值
    label_map[original_image == 0] = 0
    return label_map 

def calculate_segment_features(segment):
    def find_endpoints(segment):
        """
        找到给定骨架段的两个端点。
        端点定义为具有唯一一个邻居的点。
        """
        segment = np.array(segment)
        segment_set = set(map(tuple, segment))  # 将 segment 转换为集合，便于快速查找
        endpoints = []

        if(len(segment)==1):
            return segment[0], segment[0] 
        
        # 创建一个 8 连通邻接表
        for point in segment:
            # 计算邻居点坐标
            x, y = point
            neighbors = [
                (x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1),
                (x - 1, y - 1), (x - 1, y + 1), (x + 1, y - 1), (x + 1, y + 1)
            ]

            # 统计邻居中有多少个点在 segment 内
            neighbor_count = sum([1 for n in neighbors if n in segment_set])

            # 如果只有一个邻居，则这是一个端点
            if neighbor_count == 1:
                endpoints.append((x, y))
        # 如果找到了多个端点，只取前两个
        if len(endpoints) >= 2:
            endpoint1, endpoint2 = endpoints[:2]
            return endpoint1, endpoint2
        
        # 会有端点小于2个的时候!! 不确定!
        if len(endpoints) == 1:
            endpoint1 = endpoints[0]
            endpoint2 = endpoints[0]
            return endpoint1,endpoint2
        
        if len(endpoints) == 0:
            return segment[0], segment[0] 
        
    """
    计算段的特征信息，包括两个端点及其斜率，骨架长度和总像素数量。
    """
    # 获取两个端点
    endpoints = find_endpoints(segment)
    
    # 检查是否成功获取到端点
    if endpoints is None:
        raise ValueError("No valid endpoints found for the segment.")

    endpoint1, endpoint2 = endpoints
    # 获取两个端点
    # endpoint1 = segment[0]
    # endpoint2 = segment[-1]
    # 计算斜率k
    dx = endpoint2[1] - endpoint1[1]
    dy = endpoint2[0] - endpoint1[0]
    direction_k = np.arctan2(dy, dx)  # 斜率，范围 -π 到 π
    # 计算骨架长度（两端点之间的欧氏距离）
    length = distance.euclidean(endpoint1, endpoint2)
    # 计算总像素数量
    pixel_count = len(segment)
    # 返回特征信息
    return {
        'endpoints': (tuple(endpoint1), tuple(endpoint2)),
        'direction_k': direction_k,
        'length': length,
        'pixel_count': pixel_count
    }

########## 构建段图，over

def build_segment_graph(G, segments, junctions, segment_index_map, max_distance=10, direction_threshold=np.pi / 4, debug=True):
    """
    构建段图结构，包含段与段之间的邻接关系，并将段的特征信息作为顶点属性。
    
    参数:
    - G: 已经构建好的骨架像素点图结构。
    - segments: 由骨架分割而得到的所有段，每段为一个点的列表。
    - junctions: 交点列表，包含所有交点的坐标。
    - segment_index_map: 映射每个节点到其所在的段索引。
    - max_distance: float, 最大允许连接的距离。
    - direction_threshold: float, 方向相似度的阈值。
    
    返回:
    - segment_graph: 代表段与段之间关系的图结构，段作为顶点，边表示邻接关系。
    """
    # 初始化段图
    segment_graph = nx.Graph()
    
    # 在段图中添加段作为节点，并计算段的特征信息
    for i, segment in enumerate(segments):
        features = calculate_segment_features(segment)
        if debug:
            print(f"Segment {i+1}:")
            print(f"  Endpoints: {features['endpoints']}")
            print(f"  Direction (k): {features['direction_k']:.2f} radians")
            print(f"  Length: {features['length']:.2f} pixels")
            print(f"  Pixel Count: {features['pixel_count']}")
            print("-" * 40)
        segment_graph.add_node(i+1, **features)

    # 遍历每个段的端点，并尝试通过邻域连接其他段
    for i, features_i in segment_graph.nodes(data=True):
        endpoint1, endpoint2 = features_i['endpoints']
        direction_k_i = features_i['direction_k']

        for endpoint in [endpoint1, endpoint2]:
            # 找到在骨架图 G 中与当前端点相邻的节点（1步到 max_distance 步以内）
            for neighbor in nx.single_source_shortest_path_length(G, endpoint, cutoff=max_distance):
                # 检查 neighbor 是否在 segment_index_map 中，并且该邻居属于一个不同的段
                if neighbor in segment_index_map:
                    seg_j = segment_index_map[neighbor]
                    if seg_j != i and seg_j in segment_graph.nodes:  # 确保 seg_j 在 segment_graph 中
                        features_j = segment_graph.nodes[seg_j]
                        direction_k_j = features_j['direction_k']
                        
                        # 计算端点距离
                        distance_ij = distance.euclidean(endpoint, neighbor)
                        if distance_ij > max_distance:
                            continue
                        
                        # 检查方向差异
                        direction_diff = abs(direction_k_i - direction_k_j)
                        if direction_diff > direction_threshold:
                            continue
                        
                        # 添加段之间的边
                        segment_graph.add_edge(i, seg_j, weight=1 / distance_ij)
                        
    return segment_graph

###################### 这个是可视化分段
def visualize_segments(segments, image_shape):
        """
        可视化线段。
        """
        canvas = np.zeros(image_shape, dtype=np.uint8)
        colors = [(random.randint(50, 255), random.randint(50, 255), random.randint(50, 255)) for _ in range(len(segments))]
        for idx, segment in enumerate(segments):
            for point in segment:
                canvas[point[0], point[1]] = 255
        colored_image = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
        for idx, segment in enumerate(segments):
            color = colors[idx]
            for point in segment:
                colored_image[point[0], point[1]] = color
        plt.figure(figsize=(10, 10))
        plt.imshow(colored_image)
        plt.axis('off')
        plt.title('fen duan result')
        plt.show()
def visualize_segments_andNo(segments, image_shape):
    """
    可视化线段。
    
    参数:
    - segments: 段的列表，每个段是一个像素坐标的列表。
    - image_shape: 二值图像的形状，用于生成可视化图像的尺寸。
    """
    # 创建一个空的彩色图像，初始化为黑色
    colored_image = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)
    # 为每个段分配一个随机颜色
    colors = [(random.randint(100, 255), random.randint(100, 255), random.randint(100, 255)) for _ in range(len(segments))]
    # 绘制每个段，使用分配的颜色
    for idx, segment in enumerate(segments):
        color = colors[idx]
        for point in segment:
            if 0 <= point[0] < image_shape[0] and 0 <= point[1] < image_shape[1]:
                colored_image[point[0], point[1]] = color
    # 转换为 BGR 格式，用于绘制文本
    bgr_image = cv2.cvtColor(colored_image, cv2.COLOR_RGB2BGR)
    # 在每个段的第一个像素位置标记段编号
    for idx, segment in enumerate(segments):
        if segment:
            start_pixel = segment[0]
            if 0 <= start_pixel[0] < bgr_image.shape[0] and 0 <= start_pixel[1] < bgr_image.shape[1]:
                # 在图像上绘制段编号
                cv2.putText(bgr_image, str(idx), (start_pixel[1], start_pixel[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.23, (0, 255, 0), 1)
    # 显示结果
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('带有段编号的血管分段结果')
    plt.show()

### 废弃了，整合进其他函数了
def extract_vertex_information(segments):
    """
    提取所有段的顶点特征信息。
    """
    segment_features = []
    for segment in segments:
        features = calculate_segment_features(segment)
        segment_features.append(features)
    return segment_features

def visualize_segment_graph(segment_graph):
    """
    改进后的可视化段图函数，显示段与段之间的邻接关系，并在图顶点上添加顶点编号。
    """
    # 使用 spring 布局，并设置参数避免节点重叠
    pos = nx.spring_layout(segment_graph, seed=42, k=0.5, iterations=50)

    plt.figure(figsize=(14, 10))

    # 绘制节点
    nx.draw_networkx_nodes(segment_graph, pos, node_size=500, node_color='skyblue', edgecolors='black')

    # 绘制边，设置边的颜色和粗细
    nx.draw_networkx_edges(segment_graph, pos, edge_color='gray', width=1.5)

    # 设置节点标签为顶点编号
    node_labels = {i: f"{i}" for i in segment_graph.nodes()}
    nx.draw_networkx_labels(segment_graph, pos, labels=node_labels, font_size=8, font_color='black')

    # 显示每个段的详细信息（如斜率、长度等），在节点下方显示
    detailed_labels = {i: f"Len: {d['length']:.1f}, k: {d['direction_k']:.2f}" for i, d in segment_graph.nodes(data=True)}
    for node, (x, y) in pos.items():
        plt.text(x, y - 0.05, detailed_labels[node], fontsize=6, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3'))

    plt.title('改进后的段图及其邻接关系', fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    

# 在二值图像上可视化图信息
def visualize_image_and_graph(binary_image, segment_graph):
    """
    可视化段和图的节点编号。
    每个段显示不同的颜色，同时在每个段的顶点上标记编号。
    
    参数:
    - binary_image: 二值图像，用于显示背景的骨架。
    - segment_graph: 图结构，用于显示节点和边以及段编号。
    """
    # 创建一个与输入图像大小相同的彩色图像
    colored_image = np.zeros((binary_image.shape[0], binary_image.shape[1], 3), dtype=np.uint8)

    # 为每个段分配随机颜色
    colors = [(random.randint(50, 255), random.randint(50, 255), random.randint(50, 255)) for _ in range(len(segment_graph.nodes()))]

    # 绘制段
    for idx, (node, data) in enumerate(segment_graph.nodes(data=True)):
        segment_pixels = data.get('pixels', [])  # 获取段内像素点
        color = colors[idx]  # 为每个段分配颜色
        for pixel in segment_pixels:
            if 0 <= pixel[0] < binary_image.shape[0] and 0 <= pixel[1] < binary_image.shape[1]:
                colored_image[pixel[0], pixel[1]] = color

    # 在段顶点上标记编号
    for node, (x, y) in nx.spring_layout(segment_graph, seed=42).items():
        # 在图像上标记节点编号
        plt.text(x, y, str(node), fontsize=8)
    
    
def segment_and_visualize_vessels(image,debug=True):
    """
    读取图像，进行骨架提取，检测交点，将血管分段，并用不同颜色显示每个段。
    
    参数:
    - image_path: str, 输入图像的路径。
    """
    # 主流程
    # 调用分段和特征提取函数
    skeleton = preprocess_image(image)
    
    junctions = detect_junctions(skeleton)
    segments, G, segment_index_map = build_graph_and_segment(skeleton, junctions)
    
    # 可视化分段结果
    if debug: visualize_segments(segments, skeleton.shape)
        
    # 构建段图
    segment_graph = build_segment_graph(G, segments, junctions, segment_index_map,debug=debug)
    
    if debug: visualize_segment_graph(segment_graph)
    
    return segment_graph,skeleton,segments
