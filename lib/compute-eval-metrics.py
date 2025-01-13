import datetime
import numpy as np
import os
import cc3d
from skimage.morphology import skeletonize, skeletonize_3d
import numpy as np
import os, glob
from PIL import Image
import cv2
from vessel_salience import salience

srcdir = "/home/xkw/pxlames/segmentation/outputs/testResults/FIVE-BceDiceLsRecall_V1_0.0001_4_1"
logfile = os.path.join(srcdir,"metrics.csv") # will log to this file

def main():

    filepaths = glob.glob(srcdir + "/pred*png")
    filepaths.sort()

    print("Saving results to {}".format(logfile))

    metrics = {'accuracy':[], 'dice':[], 'cldice':[], '0betti':[], 'recall':[], 'ls_recall':[]}
    with open(logfile, 'a') as wfile:
        for i, fpath in enumerate(filepaths):
            writestr = fpath.split('/')[-1]
            gtpath = fpath.replace("pred","gt")
            imgpath = fpath.replace("pred","img")

            pred = interpolate(np.array(Image.open(fpath)))
            pred = np.squeeze(pred)

            gt = interpolate(np.array(Image.open(gtpath)))
            gt = np.squeeze(gt[:,:,0])

            img_gray = np.array(Image.open(imgpath), dtype=np.uint8)
            if len(img_gray.shape) > 2:
                img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)

            cldice_acc = clDice(pred, gt)
            betti_acc = get_betti_error(pred, gt)
            dice_acc = dice_score(pred, gt)
            acc = accuracy_score(pred, gt)
            recall = recall_score(pred, gt)
            ls_recall = ls_recall_score(img_gray, gt, pred)

            metrics['accuracy'].append(acc)
            metrics['dice'].append(dice_acc)
            metrics['cldice'].append(cldice_acc)
            metrics['0betti'].append(betti_acc)
            metrics['recall'].append(recall)
            metrics['ls_recall'].append(ls_recall)

            writestr += "; Acc {}; Dice {}; clDice {}; 0-dim Betti {}; Recall {}; LSRecall {}\n".format(
                acc, dice_acc, cldice_acc, betti_acc, recall, ls_recall)

            wfile.write(writestr)
        wfile.write("\nAverage:\n")
        for key, val in metrics.items():
            avg = np.array(val).mean()
            wfile.write("{}: {}\n".format(key, avg))

# logic okay for binary images 
def interpolate(arr):
    return arr/np.max(arr)

def dice_score(image1, image2):
    # Flatten the images to 1D arrays
    image1_flat = image1.flatten()
    image2_flat = image2.flatten()

    # Calculate the intersection and the sum of white pixels in each image
    intersection = np.sum(image1_flat * image2_flat)
    sum_values = np.sum(image1_flat) + np.sum(image2_flat)

    # Compute the Dice score
    dice = 2.0 * intersection / sum_values

    return dice

def cl_score(v, s):
    """[this function computes the skeleton volume overlap]

    Args:
        v ([bool]): [image]
        s ([bool]): [skeleton]

    Returns:
        [float]: [computed skeleton volume intersection]
    """
    return np.sum(v*s)/np.sum(s)

def clDice(v_p, v_l):
    """[this function computes the cldice metric]

    Args:
        v_p ([bool]): [predicted image]
        v_l ([bool]): [ground truth image]

    Returns:
        [float]: [cldice metric]
    """
    if len(v_p.shape)==2:
        tprec = cl_score(v_p,skeletonize(v_l))
        tsens = cl_score(v_l,skeletonize(v_p))
    elif len(v_p.shape)==3:
        tprec = cl_score(v_p,skeletonize_3d(v_l))
        tsens = cl_score(v_l,skeletonize_3d(v_p))
    return 2*tprec*tsens/(tprec+tsens)

# return : 表示输入图像中找到的连通分量的数量，即0维贝蒂数。
def conn_comp(arr):
    ## 疑问：这里明明是二维图像，用的确实三维的连通性分析。
    labels_out, numcomp = cc3d.connected_components(arr, connectivity=8, return_N=True) # 26-connected
    return numcomp

def get_betti_error(arr1, arr2, patchsize=[64,64], stepsize=[64,64]):
    arrsize = arr1.shape
    all_betti = []
    
    for x in range(0,arrsize[0],stepsize[0]):
        for y in range(0,arrsize[1],stepsize[1]):
            newidx = [x+patchsize[0],y+patchsize[1]]
            if(check_bounds([x,y],arrsize) and check_bounds(newidx,arrsize)):
                minivol1 = arr1[x:newidx[0],y:newidx[1]]
                minians1 = conn_comp(minivol1)

                minivol2 = arr2[x:newidx[0],y:newidx[1]]
                minians2 = conn_comp(minivol2)

                all_betti.append(np.abs(minians1-minians2))

    avg_betti = np.asarray(all_betti).mean()
    return avg_betti

def check_bounds(idx, volsize):
    if idx[0] < 0 or idx[0] > volsize[0]:
        return False
    if idx[1] < 0 or idx[1] > volsize[1]:
        return False
    return True

def accuracy_score(image1, image2):
    """Calculate pixel-wise accuracy between two images."""
    correct = np.sum(image1 == image2)
    total = image1.size  # total number of pixels
    accuracy = correct / total
    return accuracy

def recall_score(pred, gt):
    """计算召回率
    
    Args:
        pred: 预测图像
        gt: 真实标签图像
        
    Returns:
        float: 召回率分数
    """
    # 将图像展平为一维数组
    pred_flat = pred.flatten()
    gt_flat = gt.flatten()
    
    # 计算真阳性(TP)和假阴性(FN)
    true_positives = np.sum(pred_flat * gt_flat)
    false_negatives = np.sum((1 - pred_flat) * gt_flat)
    
    # 计算召回率
    recall = true_positives / (true_positives + false_negatives)
    
    return recall

def ls_recall_score(img_gray, gt, pred, threshold=0.01, radius=8):
    """计算LSRecall分数
    
    Args:
        img_gray: 原始灰度图像
        gt: 真实标签图像
        pred: 预测图像
        threshold: 显著性阈值
        radius: 轮廓点周围包含背景值的半径
        
    Returns:
        float: LSRecall分数
    # """
    gt = gt.astype(np.uint8)

    img_lvs, img_skel, img_lvs_skel = salience.lvs(
        img_gray,
        gt,
        radius=radius,
        return_skel=True
    )
    
    # 计算LSRecall
    ls_recall = salience.ls_recall(
        img_lvs,
        gt,
        pred,
        threshold=threshold
    )
    
    # 将img_gray和img_lvs_skel合并到一张图片
    combined_img = np.zeros((img_gray.shape[0], img_gray.shape[1], 3), dtype=np.uint8)
    combined_img[:,:,0] = img_gray  # 灰度图放在红色通道
    combined_img[:,:,1] = ((img_lvs<threshold) & (gt>0)) * 255  # 骨架图放在绿色通道
    
    # 保存合并后的图片
    cv2.imwrite('combined_ls_recall.png', combined_img)
    
    # 保存单独的img_lvs_skel图片
    cv2.imwrite('img_gray.png', img_gray)
    
    gt = gt * 255
    cv2.imwrite('gt.png', gt)
    pred = (pred * 255).astype(np.uint8)[:,:,np.newaxis]
    cv2.imwrite('pred.png', pred)
    
    return ls_recall

if __name__ == "__main__":
    main()