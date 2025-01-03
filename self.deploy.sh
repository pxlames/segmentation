
# 训练
CUDA_VISIBLE_DEVICES=1 python /home/xkw/pxlames/segmentation/main.py --params /home/xkw/pxlames/segmentation/datalists/DRIVE/train.json
CUDA_VISIBLE_DEVICES=0 python /home/xkw/pxlames/segmentation/main.py --params /home/xkw/pxlames/segmentation/datalists/FIVE/train.json --saveFlag FIVE-BceDice_0.0001_32 --configName FIVE-BceDice
CUDA_VISIBLE_DEVICES=2 python /home/xkw/pxlames/segmentation/main.py --params /home/xkw/pxlames/segmentation/datalists/FIVE/train.json --saveFlag FIVE-BceDicePD_0.0001_4 --configName FIVE-BceDicePD
CUDA_VISIBLE_DEVICES=3 python /home/xkw/pxlames/segmentation/main.py --params /home/xkw/pxlames/segmentation/datalists/FIVE/train.json --saveFlag FIVE-BceDiceSkeleton_0.0001_4 --configName FIVE-BceDiceSkeleton
CUDA_VISIBLE_DEVICES=2 python /home/xkw/pxlames/segmentation/main.py --params /home/xkw/pxlames/segmentation/datalists/FIVE/train.json --saveFlag FIVE-BceDiceSmooth_V3_0.5_0.2_0.0001_4 --configName FIVE-BceDiceSmooth
CUDA_VISIBLE_DEVICES=3 python /home/xkw/pxlames/segmentation/main.py --params /home/xkw/pxlames/segmentation/datalists/FIVE/train.json --saveFlag FIVE-BceDiceSmooth_V4_0.2_0.0001_4 --configName FIVE-BceDiceSmooth
# 测试

CUDA_VISIBLE_DEVICES=0 python /home/xkw/pxlames/segmentation/test.py --params /home/xkw/pxlames/segmentation/datalists/DRIVE/test.json --saveFlag
CUDA_VISIBLE_DEVICES=0 python /home/xkw/pxlames/segmentation/test.py --params /home/xkw/pxlames/segmentation/datalists/FIVE/test.json --saveFlag FIVE-BceDice_0.0001_32
CUDA_VISIBLE_DEVICES=0 python /home/xkw/pxlames/segmentation/test.py --params /home/xkw/pxlames/segmentation/datalists/FIVE/test.json --saveFlag FIVE-BceDiceSmooth_V3_0.5_0.2_0.0001_4

# 跑分

python /home/xkw/pxlames/segmentation/lib/compute-eval-metrics.py


# 提前分数据集
/home/xkw/anaconda3/envs/segmentation/bin/python /home/xkw/pxlames/segmentation/data/FIVE/dataSplit.py