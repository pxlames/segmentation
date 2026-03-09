import os
import glob
from PIL import Image
import numpy as np
import sys

# 指定源文件夹和结果文件夹
source_folder = '/home/xkw/pxlames/segmentation/论文图/裁剪最终结果图/结果'
output_path = '/home/xkw/pxlames/segmentation/论文图/裁剪最终结果图/拼接结果.png'

# 获取所有图片文件并按照编号排序
image_files = glob.glob(os.path.join(source_folder, '图片*.png'))
image_files.sort(key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x)))))
print(f"找到 {len(image_files)} 个图片文件")

# 首先检查所有图片尺寸是否相同
reference_size = None
for img_path in image_files:
    with Image.open(img_path) as img:
        size = img.size
        if reference_size is None:
            reference_size = size
            print(f"参考图片尺寸: {reference_size}")
        elif size != reference_size:
            print(f"错误: 图片 {os.path.basename(img_path)} 的尺寸 {size} 与参考尺寸 {reference_size} 不一致")
            sys.exit(1)

print("所有图片尺寸一致，继续处理...")

# 设置固定布局：8列5行
rows = 5
cols = 8

print(f"使用 {rows}x{cols} 的网格布局")

# 定义布局参数
dpi = 300  # 设置DPI为300
cm_to_pixels = dpi / 2.54  # 将厘米转换为像素
img_size_cm = 3.28  # 每张图片3.28厘米
img_size_px = int(img_size_cm * cm_to_pixels)  # 图片尺寸（像素）
gap_px = 5  # 图片之间的间隙（像素）

# 计算画布大小
canvas_width = cols * img_size_px + (cols - 1) * gap_px
canvas_height = rows * img_size_px + (rows - 1) * gap_px

# 创建空白画布（白色背景）
canvas = Image.new('RGB', (canvas_width, canvas_height), (255, 255, 255))

# 遍历图片文件并放置到画布上
count = 0
for r in range(rows):
    for c in range(cols):
        if count >= len(image_files):
            break
            
        # 计算当前图片在画布上的位置
        x = c * (img_size_px + gap_px)
        y = r * (img_size_px + gap_px)
        
        # 打开图片并调整大小
        img_path = image_files[count]
        img = Image.open(img_path)
        img = img.resize((img_size_px, img_size_px), Image.LANCZOS)
        
        # 将图片放置到画布上
        canvas.paste(img, (x, y))
        
        count += 1
        print(f"处理第 {count} 张图片: {os.path.basename(img_path)}")

# 保存结果
canvas.save(output_path, dpi=(dpi, dpi))
print(f"拼接完成，结果保存至: {output_path}")
print(f"画布大小: {canvas_width}x{canvas_height} 像素, 约 {canvas_width/cm_to_pixels:.2f}x{canvas_height/cm_to_pixels:.2f} 厘米")
