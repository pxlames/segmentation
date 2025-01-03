import numpy as np
import scipy.io

cal_masks_path = '/home/xkw/pxlames/segmentation/outputs/testResults/FIVE_BceDice_0.0001_4/cal_gt_masks.npy'
cal_scores_path = '/home/xkw/pxlames/segmentation/outputs/testResults/FIVE_BceDice_0.0001_4/cal_scores.npy'
test_masks_path = '/home/xkw/pxlames/segmentation/outputs/testResults/FIVE_BceDice_0.0001_4/gt_masks.npy'
test_scores_path = '/home/xkw/pxlames/segmentation/outputs/testResults/FIVE_BceDice_0.0001_4/scores.npy'
cal_masks = np.load(cal_masks_path)
cal_scores = np.load(cal_scores_path)
test_masks = np.load(test_masks_path)
test_scores = np.load(test_scores_path)

# 重塑数组
cal_masks = np.transpose(cal_masks, (1, 2, 0))
cal_scores = np.transpose(cal_scores, (1, 2, 0))
test_masks = np.transpose(test_masks, (1, 2, 0))
test_scores = np.transpose(test_scores, (1, 2, 0))
print("成功加载gt_masks和scores文件")