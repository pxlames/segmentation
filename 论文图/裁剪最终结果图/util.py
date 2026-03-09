import os
from PIL import Image
import glob

# 指定源文件夹和目标文件夹
source_folder = '/home/xkw/pxlames/segmentation/论文图/裁剪最终结果图'
target_folder = '/home/xkw/pxlames/segmentation/论文图/裁剪最终结果图/结果'

# 确保目标文件夹存在
os.makedirs(target_folder, exist_ok=True)

# 获取所有图片文件
image_files = glob.glob(os.path.join(source_folder, '图片*.png'))
print(f"找到 {len(image_files)} 个图片文件")

# 查找所有图片的最小边长
min_side = float('inf')
for img_path in image_files:
    # 打开图片并获取尺寸
    with Image.open(img_path) as img:
        width, height = img.size
        # 计算最小边长
        min_dim = min(width, height)
        if min_dim < min_side:
            min_side = min_dim

print(f"所有图片的最小边长为 {min_side} 像素")
print(f"将从每张图片中心裁剪 {min_side}x{min_side} 的正方形")

# 对每张图片进行处理
for i, img_path in enumerate(image_files):
    # 获取文件名
    filename = os.path.basename(img_path)
    
    # 打开图片
    with Image.open(img_path) as img:
        width, height = img.size
        
        # 判断是否为最后8张图片
        if i >= len(image_files) - 8 and i < len(image_files) - 1:
            # 最后8张中的前7张从左上角开始裁剪
            left = 0
            top = 0
            right = min_side
            bottom = min_side
            print(f"处理 {filename} 完成，从左上角裁剪了 {min_side}x{min_side} 的区域")
        else:
            # 其他图片（包括最后一张）从中心裁剪
            left = (width - min_side) // 2
            top = (height - min_side) // 2
            right = left + min_side
            bottom = top + min_side
            print(f"处理 {filename} 完成，从中心裁剪了 {min_side}x{min_side} 的区域")
        
        # 裁剪图片
        cropped_img = img.crop((left, top, right, bottom))
        
        # 保存裁剪后的图片到目标文件夹
        output_path = os.path.join(target_folder, filename)
        cropped_img.save(output_path)

print("所有图片处理完成！")
