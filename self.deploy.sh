
# 训练
CUDA_VISIBLE_DEVICES=1 python /home/pxl/myProject/血管分割/molong-深度插值/molong-work/segmentation/main.py --params /home/pxl/myProject/血管分割/molong-深度插值/molong-work/segmentation/datalists/DRIVE/train.json

# 测试

CUDA_VISIBLE_DEVICES=1 python /home/pxl/myProject/血管分割/molong-深度插值/molong-work/segmentation/main.py --params /home/pxl/myProject/血管分割/molong-深度插值/molong-work/segmentation/datalists/DRIVE/test.json

# 跑分

python /home/pxl/myProject/血管分割/molong-深度插值/molong-work/segmentation/lib/compute-eval-metrics.py

