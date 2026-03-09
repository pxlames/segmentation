

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from PIL import Image

## 误差和不确定性区域联系在一起了！
def CI_fwer(scores, masks, alpha=0.05):
    """
    Calculate the threshold for family-wise error rate (FWER) in conformal inference.
    Parameters:
    scores (ndarray): 3D array of statistical scores where the last dimension represents different images.
    masks (ndarray): 3D binary array indicating the mask for each image, matching the scores array.
    alpha (float): Significance level for FWER.
    Returns:
    tuple: threshold for FWER, array of maximum values per image after masking.
    """
    nimages = scores.shape[-1]
    max_vals = np.zeros(nimages) # max_vals 用来存储每个样本在非掩码区域的最大分数。

    for i in range(nimages):
        # masked_image = scores[..., i] * (1 - masks[..., i].astype(float)) # 原始代码有问题,可能是python不能兼容的原因
        masked_image = np.nan_to_num(scores[..., i] * (1 - masks[..., i].astype(float)), nan=0.0, posinf=0.0, neginf=0.0)
        #这种方式将 NaN 和极端值（inf 和 -inf）替换为 0，从而避免了绘图或计算过程中出现问题。
        max_vals[i] = masked_image.max()
        # print(max_vals[i])
    threshold = np.percentile(max_vals, 100 * (1 - alpha))  
    
    # 绘制 max_vals 的分布和阈值
    plt.figure(figsize=(10, 6))
    plt.hist(max_vals, bins=30, color='skyblue', edgecolor='black', density=True)
    plt.axvline(threshold, color='red', linestyle='--', label=f'{(1 - alpha) * 100}% Threshold: {threshold:.2f}')
    plt.xlabel('Max values')
    plt.ylabel('Density')
    plt.title(f'Distribution of Maximum Values (Threshold at {100 * (1 - alpha)}th percentile)')
    plt.legend()
    plt.show()
    
    return threshold, max_vals

def CI_fwer_topo(scores, masks, alpha1, alpha2):
    """
    Calculate the threshold for family-wise error rate (FWER) in conformal inference.
    """
    nimages = scores.shape[-1]
    min_vals = np.zeros(nimages)
    all_topo_probs = []  # 用于收集所有样本的topo_mask区域的概率值

    for i in range(nimages):  # 遍历所有图像
        # 计算漏检区域
        gt_mask = masks[..., i].astype(float)
        pred_mask = (scores[..., i] > 0.6).astype(float)
        missed_regions = np.clip(gt_mask - pred_mask, 0, 1)
        
        # 提取骨架
        from skimage.morphology import skeletonize
        gt_mask_binary = (gt_mask > 0).astype(np.uint8)
        # skeleton = skeletonize(gt_mask_binary)
        
        # 获取missed_regions和skeleton的重叠区域
        topo_mask = (missed_regions > 0) & gt_mask_binary
        
        # 收集当前样本topo_mask区域的概率值
        # 收集当前样本topo_mask区域的非零概率值
        mask_probs = scores[..., i][topo_mask]
        # 只保留非零的概率值
        non_zero_probs = mask_probs[mask_probs>=0] # 只有当有非零概率值时才添加
        if len(non_zero_probs) > 0: 
            all_topo_probs.extend(non_zero_probs)
        
        # 计算最终的masked_image
        masked_image = np.nan_to_num(scores[..., i] * topo_mask, nan=0.0, posinf=0.0, neginf=0.0)
        min_vals[i] = masked_image.min()
    print(f'all_topo_probs最小值: {min(all_topo_probs)}')
    # 将所有概率值转换为numpy数组
    all_topo_probs = np.array(all_topo_probs)
    
    # 绘制所有样本topo_mask区域的概率分布
    plt.figure(figsize=(10, 6))
    plt.hist(all_topo_probs, bins=100, color='skyblue', edgecolor='black', density=True)
    plt.xlabel('Probability')
    plt.ylabel('Density')
    plt.title('Probability Distribution in All Topology Mask Regions')
    plt.show()

    return all_topo_probs

# 风险控制最初版本
def false_positive_rate(pred_masks, true_masks):
    """
    Calculate the False Positive Rate (FPR)
    
    Parameters:
    - pred_masks: binary predicted masks, shape (H, W, N)
    - true_masks: ground truth masks, shape (H, W, N)
    
    Returns:
    - float: the average False Positive Rate
    """
    assert pred_masks.shape == true_masks.shape
    
    # 计算FPR：FP / (FP + TN)
    fp = ((pred_masks == 1) & (true_masks == 0)).sum(axis=(0, 1))
    tn = ((pred_masks == 0) & (true_masks == 0)).sum(axis=(0, 1))
    fpr = fp / (fp + tn)
    return fpr.mean()

def false_negative_rate(pred_masks, true_masks):
    """
    Calculate the False Negative Rate (FNR)
    
    Parameters:
    - pred_masks: binary predicted masks, shape (H, W, N)
    - true_masks: ground truth masks, shape (H, W, N)
    
    Returns:
    - float: the average False Negative Rate
    """
    assert pred_masks.shape == true_masks.shape
    
    # 计算FNR：FN / (FN + TP)
    fn = ((pred_masks == 0) & (true_masks == 1)).sum(axis=(0, 1))
    tp = ((pred_masks == 1) & (true_masks == 1)).sum(axis=(0, 1))
    fnr = fn / (fn + tp)
    return fnr.mean()

# Define lamhat threshold function（闭包形式）
def create_lamhat_threshold_FNR(cal_scores, cal_gt_masks, n, alpha):
    def lamhat_threshold(lam):
        pred_masks = cal_scores >= lam
        return false_negative_rate(pred_masks, cal_gt_masks) - ((n + 1) / n * alpha - 1 / n)
    return lamhat_threshold

# Define lamhat threshold function（闭包形式）
def create_lamhat_threshold_FPR(cal_scores, cal_gt_masks, n, alpha):
    def lamhat_threshold(lam):
        pred_masks = cal_scores >= lam
        return false_positive_rate(pred_masks, cal_gt_masks) - ((n + 1) / n * alpha - 1 / n)
    return lamhat_threshold

import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

def remove_edge_lines(image):
    """
    对图像进行轻微腐蚀，去除边缘细线
    Args:
        image: 输入图像
    Returns:
        处理后的图像
    """
    # 使用小的结构元素
    kernel = np.array([[0,1,0],
                  [1,1,1],
                  [0,1,0]], np.uint8)    
    # 进行单次腐蚀操作
    eroded = cv2.erode(image, kernel, iterations=2)
    
    return eroded

# 加载测试数据
cal_masks_path = '/home/xkw/pxlames/segmentation/outputs/testResults/FIVE-BceDiceSmooth_0.0001_4/cal_gt_masks.npy'
cal_scores_path = '/home/xkw/pxlames/segmentation/outputs/testResults/FIVE-BceDiceSmooth_0.0001_4/cal_scores.npy'
test_masks_path = '/home/xkw/pxlames/segmentation/outputs/testResults/FIVE-BceDiceSmooth_0.0001_4/gt_masks.npy'
test_scores_path = '/home/xkw/pxlames/segmentation/outputs/testResults/FIVE-BceDiceSmooth_0.0001_4/scores.npy'
train_scores_path = '/home/xkw/pxlames/segmentation/outputs/testResults/FIVE-BceDiceSmooth_0.0001_4/train_scores.npy'
train_masks_path = '/home/xkw/pxlames/segmentation/outputs/testResults/FIVE-BceDiceSmooth_0.0001_4/train_gt_masks.npy'
val_scores_path = '/home/xkw/pxlames/segmentation/outputs/testResults/FIVE-BceDiceSmooth_0.0001_4/val_scores.npy'
val_masks_path = '/home/xkw/pxlames/segmentation/outputs/testResults/FIVE-BceDiceSmooth_0.0001_4/val_gt_masks.npy'

cal_filenames_path = '/home/xkw/pxlames/segmentation/outputs/testResults/FIVE-BceDiceSmooth_0.0001_4/cal_filenames.npy'
test_filenames_path = '/home/xkw/pxlames/segmentation/outputs/testResults/FIVE-BceDiceSmooth_0.0001_4/test_filenames.npy'
train_filenames_path = '/home/xkw/pxlames/segmentation/outputs/testResults/FIVE-BceDiceSmooth_0.0001_4/train_filenames.npy'
val_filenames_path = '/home/xkw/pxlames/segmentation/outputs/testResults/FIVE-BceDiceSmooth_0.0001_4/val_filenames.npy'

val_filenames = np.load(val_filenames_path)
cal_masks = np.load(cal_masks_path)
cal_scores = np.load(cal_scores_path)
test_masks = np.load(test_masks_path)
test_scores = np.load(test_scores_path)
train_masks = np.load(train_masks_path)
train_scores = np.load(train_scores_path)
val_masks = np.load(val_masks_path)
val_scores = np.load(val_scores_path)
# 重塑数组
cal_masks = np.transpose(cal_masks, (1, 2, 0))
cal_scores = np.transpose(cal_scores, (1, 2, 0))
test_masks = np.transpose(test_masks, (1, 2, 0))
test_scores = np.transpose(test_scores, (1, 2, 0))
train_masks = np.transpose(train_masks, (1, 2, 0))
train_scores = np.transpose(train_scores, (1, 2, 0))
val_masks = np.transpose(val_masks, (1, 2, 0))
val_scores = np.transpose(val_scores, (1, 2, 0))

cal_filenames = np.load(cal_filenames_path)
test_filenames = np.load(test_filenames_path)
train_filenames = np.load(train_filenames_path)
val_filenames = np.load(val_filenames_path)

print("成功加载gt_masks和scores文件")


# 使用方法
# threshold_func = create_lamhat_threshold_FPR(cal_scores, cal_masks, cal_scores.shape[0], 0.005)
# lamhat1 = brentq(threshold_func, 0, 1)
lamhat1 = 0.8868423998361363
print(f"Calculated lamhat threshold: {lamhat1}")
# 使用方法
# threshold_func = create_lamhat_threshold_FNR(cal_scores, cal_masks, cal_scores.shape[0], 0.05)
# lamhat2 = brentq(threshold_func, 0, 1)
lamhat2 = 0.013467237820238523
print(f"Calculated lamhat threshold: {lamhat2}")

saveTrainDir = '/home/xkw/pxlames/segmentation/outputs/firstStageResults/FIVE-BceDiceSmooth_0.0001_4/FIVE0/train/Cfolder'
saveValDir = '/home/xkw/pxlames/segmentation/outputs/firstStageResults/FIVE-BceDiceSmooth_0.0001_4/FIVE0/val/Cfolder'
saveTestDir = '/home/xkw/pxlames/segmentation/outputs/firstStageResults/FIVE-BceDiceSmooth_0.0001_4/FIVE0/test/Cfolder'

saveTrainDir2 = '/home/xkw/pxlames/segmentation/outputs/firstStageResults/FIVE-BceDiceSmooth_0.0001_4/FIVE0/train/Ufolder'
saveValDir2 = '/home/xkw/pxlames/segmentation/outputs/firstStageResults/FIVE-BceDiceSmooth_0.0001_4/FIVE0/val/Ufolder'
saveTestDir2 = '/home/xkw/pxlames/segmentation/outputs/firstStageResults/FIVE-BceDiceSmooth_0.0001_4/FIVE0/test/Ufolder'

print(test_scores.shape)
for i in range(test_scores.shape[2]):  
    # 提取单个2D图像 (2048, 2048)
    test_score = test_scores[:, :, i]  
    filename = test_filenames[i]
    test_result_certain = (test_score > lamhat1).astype(np.uint8) * 255
    test_result_U = (test_score > lamhat2).astype(np.uint8) * 255
    
    # 保存图像
    img = Image.fromarray(test_result_certain)
    save_path = os.path.join(saveTestDir, f"{filename}.png")
    img.save(save_path)
    
     #! +++++++++++++后续这一步可以略去试试:++++++++++++++
    # 计算差值图像
    diff_img = cv2.subtract(test_result_U, test_result_certain)
    # 轻微腐蚀处理去除边缘细线
    # processed_img = remove_edge_lines(diff_img)
    img = Image.fromarray(diff_img)
    save_path = os.path.join(saveTestDir2, f"{filename}.png")
    img.save(save_path)
    
for i in range(train_scores.shape[2]):  
    # 提取单个2D图像 (2048, 2048)
    train_score = train_scores[:, :, i]  
    filename = train_filenames[i]
    train_result_certain = (train_score > lamhat1).astype(np.uint8) * 255
    train_result_U = (train_score > lamhat2).astype(np.uint8) * 255
    
    # 保存图像
    img = Image.fromarray(train_result_certain)
    save_path = os.path.join(saveTrainDir, f"{filename}.png")
    img.save(save_path)
    
    #! +++++++++++++后续这一步可以略去试试:++++++++++++++
    # 计算差值图像
    diff_img = cv2.subtract(train_result_U, train_result_certain)
    # # 轻微腐蚀处理去除边缘细线
    # processed_img = remove_edge_lines(diff_img)
    img = Image.fromarray(diff_img)
    save_path = os.path.join(saveTrainDir2, f"{filename}.png")
    img.save(save_path)
    
for i in range(val_scores.shape[2]):  
    # 提取单个2D图像 (2048, 2048)
    val_score = val_scores[:, :, i]  
    filename = val_filenames[i]
    val_result_certain = (val_score > lamhat1).astype(np.uint8) * 255
    val_result_U = (val_score > lamhat2).astype(np.uint8) * 255
    
    # 保存图像
    img = Image.fromarray(val_result_certain)
    save_path = os.path.join(saveValDir, f"{filename}.png")
    img.save(save_path)

    #! +++++++++++++后续这一步可以略去试试:++++++++++++++
    # 计算差值图像
    diff_img = cv2.subtract(test_result_U, test_result_certain)
    # 轻微腐蚀处理去除边缘细线
    # processed_img = remove_edge_lines(diff_img)
    img = Image.fromarray(diff_img)
    save_path = os.path.join(saveValDir2, f"{filename}.png")
    img.save(save_path)
    
