from graphviz import Digraph

def create_network_visualization():
    # 创建有向图
    dot = Digraph(comment='Network Architecture')
    dot.attr(rankdir='LR')  # 从左到右的布局
    dot.attr('node', shape='box', style='rounded,filled', fillcolor='lightblue')
    
    # 设置子图样式
    dot.attr(splines='ortho')  # 使用正交线
    dot.attr(nodesep='0.5')    # 节点间距
    dot.attr(ranksep='0.5')    # 层级间距
    
    # 输入层
    with dot.subgraph(name='cluster_input') as c:
        c.attr(label='Input Layer')
        c.attr(style='rounded')
        c.node('input', 'Input Image\n3 channels')
        c.node('mask1', 'Threshold Mask\n0.5')
        c.node('mask2', 'CP Mask')
        
    # 编码器层
    with dot.subgraph(name='cluster_encoder') as c:
        c.attr(label='Encoder Path')
        c.node('enc1', 'Encoder Block 1\n64 channels')
        c.node('pool1', 'MaxPool 2x2')
        c.node('enc2', 'Encoder Block 2\n128 channels')
        c.node('pool2', 'MaxPool 2x2')
        c.node('enc3', 'Encoder Block 3\n256 channels')
        c.node('pool3', 'MaxPool 2x2')
        c.node('enc4', 'Encoder Block 4\n512 channels')
        
    # 注意力模块
    dot.attr('node', shape='diamond', style='filled', fillcolor='lightpink')
    dot.node('am1', 'AM 1')
    dot.node('am2', 'AM 2')
    dot.node('am3', 'AM 3')
    dot.node('am4', 'AM 4')
    
    # 下采样掩码
    dot.attr('node', shape='box', style='rounded,filled', fillcolor='lightyellow')
    dot.node('ds1', 'Downsample 1/2')
    dot.node('ds2', 'Downsample 1/2')
    dot.node('ds3', 'Downsample 1/4')
    dot.node('ds4', 'Downsample 1/4')
    dot.node('ds5', 'Downsample 1/8')
    dot.node('ds6', 'Downsample 1/8')
    
    # 解码器层
    with dot.subgraph(name='cluster_decoder') as c:
        c.attr(label='Decoder Path')
        c.attr('node', shape='box', style='rounded,filled', fillcolor='lightgreen')
        c.node('up1', 'Upsample x2')
        c.node('dec4', 'Decoder Block 4\n256 channels')
        c.node('up2', 'Upsample x2')
        c.node('dec3', 'Decoder Block 3\n128 channels')
        c.node('up3', 'Upsample x2')
        c.node('dec2', 'Decoder Block 2\n64 channels')
        c.node('dec1', 'Output Conv\n1 channel')
        
    # 输出层
    dot.attr('node', shape='box', style='rounded,filled', fillcolor='lightblue')
    dot.node('output', 'Final Output')
    
    # 添加连接
    # 主路径
    dot.edge('input', 'enc1')
    dot.edge('enc1', 'pool1')
    dot.edge('pool1', 'enc2')
    dot.edge('enc2', 'pool2')
    dot.edge('pool2', 'enc3')
    dot.edge('enc3', 'pool3')
    dot.edge('pool3', 'enc4')
    
    # 掩码路径
    dot.edge('mask1', 'am1')
    dot.edge('mask2', 'am1')
    dot.edge('mask1', 'ds1')
    dot.edge('mask2', 'ds2')
    dot.edge('ds1', 'ds3')
    dot.edge('ds2', 'ds4')
    dot.edge('ds3', 'ds5')
    dot.edge('ds4', 'ds6')
    
    # 注意力连接
    dot.edge('ds1', 'am2')
    dot.edge('ds2', 'am2')
    dot.edge('ds3', 'am3')
    dot.edge('ds4', 'am3')
    dot.edge('ds5', 'am4')
    dot.edge('ds6', 'am4')
    
    # 解码器路径
    dot.edge('enc4', 'am4')
    dot.edge('am4', 'up1')
    dot.edge('up1', 'dec4')
    dot.edge('dec4', 'am3')
    dot.edge('am3', 'up2')
    dot.edge('up2', 'dec3')
    dot.edge('dec3', 'am2')
    dot.edge('am2', 'up3')
    dot.edge('up3', 'dec2')
    dot.edge('dec2', 'am1')
    dot.edge('am1', 'dec1')
    dot.edge('dec1', 'output')
    
    return dot

def save_network_visualization(filename='network_architecture'):
    dot = create_network_visualization()
    # 保存为PDF（矢量格式，适合放大）
    dot.render(filename, format='pdf', cleanup=True)
    # 保存为PNG（位图格式，适合直接使用）
    dot.render(filename, format='png', cleanup=True)
    print(f"Visualization saved as {filename}.pdf and {filename}.png")

if __name__ == "__main__":
    save_network_visualization()