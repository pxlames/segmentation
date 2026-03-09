
# 训练
CUDA_VISIBLE_DEVICES=1 python /home/xkw/pxlames/segmentation/main.py --params /home/xkw/pxlames/segmentation/datalists/DRIVE/train.json
CUDA_VISIBLE_DEVICES=0 python /home/xkw/pxlames/segmentation/main.py --params /home/xkw/pxlames/segmentation/datalists/FIVE/train.json --saveFlag FIVE-BceDice_ResUnet_test --configName FIVE-BceDice
CUDA_VISIBLE_DEVICES=2 python /home/xkw/pxlames/segmentation/main.py --params /home/xkw/pxlames/segmentation/datalists/FIVE/train.json --saveFlag FIVE-BceDicePD_0.0001_4 --configName FIVE-BceDicePD
CUDA_VISIBLE_DEVICES=3 python /home/xkw/pxlames/segmentation/main.py --params /home/xkw/pxlames/segmentation/datalists/FIVE/train.json --saveFlag FIVE-BceDiceSkeleton_0.0001_4 --configName FIVE-BceDiceSkeleton
CUDA_VISIBLE_DEVICES=2 python /home/xkw/pxlames/segmentation/main.py --params /home/xkw/pxlames/segmentation/datalists/FIVE/train.json --saveFlag FIVE-BceDiceSmooth_V3_0.5_0.2_0.0001_4 --configName FIVE-BceDiceSmooth
CUDA_VISIBLE_DEVICES=2 python /home/xkw/pxlames/segmentation/main.py --params /home/xkw/pxlames/segmentation/datalists/FIVE/train.json --saveFlag FIVE-BceDiceLsRecall_V1_0.0001_4_1 --configName FIVE-BceDiceLsRecall
CUDA_VISIBLE_DEVICES=0 python /home/xkw/pxlames/segmentation/main.py --params /home/xkw/pxlames/segmentation/datalists/DRIVE/train.json --saveFlag DRIVE-unet-softIouLoss_256_16_0.001_2000 --configName DRIVE-softIouLoss
CUDA_VISIBLE_DEVICES=0 python /home/xkw/pxlames/segmentation/main.py --params /home/xkw/pxlames/segmentation/datalists/DRIVE/train.json --saveFlag DRIVE-unetplusplus-BceDice_256_16_0.001_2000_sgd1 --configName DRIVE-BceDice-unetplusplus
CUDA_VISIBLE_DEVICES=1 python /home/xkw/pxlames/segmentation/main.py --params /home/xkw/pxlames/segmentation/datalists/DRIVE/train.json --saveFlag DRIVE-unetplusplus-BceDice_256_16_0.001_2000_adam1 --configName DRIVE-BceDice-unetplusplus
CUDA_VISIBLE_DEVICES=3 python /home/xkw/pxlames/segmentation/main.py --params /home/xkw/pxlames/segmentation/datalists/DRIVE/train.json --saveFlag DRIVE-trans_unet-BceDice_256_16_0.001_2000_sgd --configName DRIVE-BceDice-trans_unet
CUDA_VISIBLE_DEVICES=3 python /home/xkw/pxlames/segmentation/main.py --params /home/xkw/pxlames/segmentation/datalists/DRIVE/train.json --saveFlag FIVE-BceDiceSmooth_0.0001_4 --configName FIVE—SECOND-BceDice
CUDA_VISIBLE_DEVICES=3 python /home/xkw/pxlames/segmentation/main.py --params /home/xkw/pxlames/segmentation/datalists/FIVE/train.json --saveFlag FIVE-BceDiceSmooth_0.0001_4_Unet_second_modified_img+C0.5+0U  --configName FIVE—SECOND-BceDice
CUDA_VISIBLE_DEVICES=3 python /home/xkw/pxlames/segmentation/main.py --params /home/xkw/pxlames/segmentation/datalists/FIVE/train.json --saveFlag FIVE-BceDicecldice_0.0001_4_Unet++  --configName FIVE-BceDice-Unet++

# 测试

CUDA_VISIBLE_DEVICES=0 python /home/xkw/pxlames/segmentation/test.py --params /home/xkw/pxlames/segmentation/datalists/DRIVE/test.json --saveFlag
CUDA_VISIBLE_DEVICES=0 python /home/xkw/pxlames/segmentation/test.py --params /home/xkw/pxlames/segmentation/datalists/FIVE/test.json --saveFlag FIVE-BceDice_0.0001_32
CUDA_VISIBLE_DEVICES=0 python /home/xkw/pxlames/segmentation/test.py --params /home/xkw/pxlames/segmentation/datalists/FIVE/test.json --saveFlag FIVE-BceDiceSmooth_V3_0.5_0.2_0.0001_4
CUDA_VISIBLE_DEVICES=0 python /home/xkw/pxlames/segmentation/test.py --params /home/xkw/pxlames/segmentation/datalists/FIVE/test.json --saveFlag FIVE-BceDiceLsRecall_V1_0.0001_4_1
CUDA_VISIBLE_DEVICES=0 python /home/xkw/pxlames/segmentation/test.py --params /home/xkw/pxlames/segmentation/datalists/FIVE/test.json --saveFlag FIVE-BceDiceSmooth_0.0001_4
CUDA_VISIBLE_DEVICES=0 python /home/xkw/pxlames/segmentation/test.py --params /home/xkw/pxlames/segmentation/datalists/DRIVE/test.json --saveFlag DRIVE-trans_unet-BceDice_256_16_0.001_2000
CUDA_VISIBLE_DEVICES=2 python /home/xkw/pxlames/segmentation/test.py --params /home/xkw/pxlames/segmentation/datalists/FIVE/test-second.json --saveFlag FIVE-BceDiceSmooth_0.0001_4_Unet_second_img+C+0U

# 跑分

python /home/xkw/pxlames/segmentation/lib/compute-eval-metrics.py
python /home/xkw/pxlames/segmentation/lib/compute-eval-metrics.py --srcdir /home/xkw/pxlames/segmentation/outputs/secondStageResults/testResults/FIVE-BceDiceSmooth_0.0001_4_Unet_second_C0.5_Utopo

# 提前分数据集
/home/xkw/anaconda3/envs/segmentation/bin/python /home/xkw/pxlames/segmentation/data/FIVE/dataSplit.py