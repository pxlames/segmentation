# load all into cpu
# do cropping to patch size
# normalize range
# test code by outputting few patches
# training, testing, val
import torch
import os, glob, sys
import numpy as np
from PIL import Image
from os.path import join as pjoin
from torchvision import transforms
from torch.utils import data
from skimage import io
import pdb
import SimpleITK as sitk
from LIOT import distance_weight_binary_pattern_faster

class DRIVE(data.Dataset):
    def __init__(self, listpath, folderpaths, task, crop_size = 128):

        self.listpath = listpath
        self.imgfolder = folderpaths[0]
        self.gtfolder = folderpaths[1]
        self.crop_size = crop_size

        self.dataCPU = {}
        self.dataCPU['image'] = []
        self.dataCPU['label'] = []
        self.dataCPU['filename'] = []

        self.task = task
 
        self.to_tensor = transforms.ToTensor() # converts HWC in [0,255] to CHW in [0,1]

        self.loadCPU()

    def loadCPU(self):
        with open(self.listpath, 'r') as f:
            mylist = f.readlines()
        mylist = [x.rstrip('\n') for x in mylist]

        for i, entry in enumerate(mylist):

            components = entry.split('.')
            filename = components[0]

            if self.task == "test":
                im_path = pjoin(self.imgfolder, filename) + '_test.tif'
            else:
                im_path = pjoin(self.imgfolder, filename) + '_training.tif'

            gt_path = pjoin(self.gtfolder, filename) + '_manual1.gif'
            img = Image.open(im_path)
            gt = Image.open(gt_path)

            # 转换为numpy数组进行LION处理
            img_np = np.array(img)  # [H, W, 3]
            
            # 计算LION特征
            lion_features = distance_weight_binary_pattern_faster(img_np)  # [H, W, 4]
            
            # 将原始图像和LION特征拼接
            img_combined = np.concatenate([img_np, lion_features], axis=2)  # [H, W, 7]
            
            # 转换为tensor
            img_combined = torch.from_numpy(img_combined).permute(2, 0, 1).float() / 255.0  # [7, H, W]
            gt = self.to_tensor(gt)
            
            img = img_combined
            # normalize within a channel
            for j in range(img.shape[0]):
                meanval = img[j].mean()
                stdval = img[j].std()
                img[j] = (img[j] - meanval) / stdval
            # cpu store
            self.dataCPU['image'].append(img)
            self.dataCPU['label'].append(gt)
            self.dataCPU['filename'].append(filename)

    def __len__(self): # total number of 2D slices
        return len(self.dataCPU['filename'])

    def __getitem__(self, index): # select random crop and return CHW torch tensor

        torch_img = self.dataCPU['image'][index] #HW
        torch_gt = self.dataCPU['label'][index] #HW

        if self.task == "train":
            # crop: compute top-left corner first
            _, H, W = torch_img.shape
            corner_h = np.random.randint(low=0, high=H-self.crop_size)
            corner_w = np.random.randint(low=0, high=W-self.crop_size)

            torch_img = torch_img[:, corner_h:corner_h+self.crop_size, corner_w:corner_w+self.crop_size]
            torch_gt = torch_gt[:, corner_h:corner_h+self.crop_size, corner_w:corner_w+self.crop_size]

        return torch_img, torch_gt, self.dataCPU['filename'][index]
    
    
       
class FIVE(data.Dataset):
    def __init__(self, folderpaths, task, crop_size = 128):

        self.imgfolder = folderpaths[0]
        self.gtfolder = folderpaths[1]
        self.crop_size = crop_size

        self.dataCPU = {}
        self.dataCPU['image'] = []
        self.dataCPU['label'] = []
        self.dataCPU['filename'] = []

        self.task = task
 
        self.to_tensor = transforms.ToTensor() # converts HWC in [0,255] to CHW in [0,1]

        self.loadCPU()

    def loadCPU(self):
        # 获取所有png图像文件路径
        img_list = glob.glob(pjoin(self.imgfolder, '*.png'))
        print(f"图像文件数量: {len(img_list)}")
        for im_path in img_list:
            # 从文件路径中提取文件名（不含扩展名）
            filename = os.path.splitext(os.path.basename(im_path))[0]
            
            # 构建对应的ground truth路径
            gt_path = pjoin(self.gtfolder, filename + '.png')
            
            # 读取图像
            img = Image.open(im_path)
            gt = Image.open(gt_path)
            
            # 转换为numpy数组进行LION处理
            img = np.array(img)  # [H, W, 3]
            
            # 计算LION特征
            lion_features = distance_weight_binary_pattern_faster(img)  # [H, W, 4]
            
            # 将原始图像和LION特征拼接
            img_combined = np.concatenate([img, lion_features], axis=2)  # [H, W, 7]
            
            # 转换为tensor
            img_combined = torch.from_numpy(img_combined).permute(2, 0, 1).float() / 255.0  # [7, H, W]
            gt = self.to_tensor(gt)
            
            print('test:========')
            print(gt)
            # normalize within a channel
            for j in range(img.shape[0]):
                meanval = img[j].mean()
                stdval = img[j].std()
                img[j] = (img[j] - meanval) / stdval
                
            # cpu store
            self.dataCPU['image'].append(img)
            self.dataCPU['label'].append(gt)
            self.dataCPU['filename'].append(filename)

    def __len__(self): # total number of 2D slices
        return len(self.dataCPU['filename'])

    def __getitem__(self, index): # select random crop and return CHW torch tensor

        torch_img = self.dataCPU['image'][index] #HW
        torch_gt = self.dataCPU['label'][index] #HW
        torch_gt = torch_gt[0].unsqueeze(0)  # 方法2：只取第一个通道

        if self.task == "train":
            # crop: compute top-left corner first
            _, H, W = torch_img.shape
            corner_h = np.random.randint(low=0, high=H-self.crop_size)
            corner_w = np.random.randint(low=0, high=W-self.crop_size)

            torch_img = torch_img[:, corner_h:corner_h+self.crop_size, corner_w:corner_w+self.crop_size]
            torch_gt = torch_gt[:, corner_h:corner_h+self.crop_size, corner_w:corner_w+self.crop_size]

        return torch_img, torch_gt, self.dataCPU['filename'][index]



if __name__ == "__main__":
    # flag = "training"

    dst = DRIVE("/home/pxl/myProject/血管分割/molong-深度插值/molong-work/segmentation/datalists/DRIVE/list-train.csv",
                 ["/home/pxl/myProject/血管分割/molong-深度插值/molong-work/segmentation/data/DRIVE/training/images",
                 "/home/pxl/myProject/血管分割/molong-深度插值/molong-work/segmentation/data/DRIVE/training/1st_manual"], 
                task="train", 
                crop_size=128) 
    training_generator = data.DataLoader(dst, shuffle=False, batch_size=2, num_workers=8)
    # 测试读取一个图片路径
    for step, (patch, mask, filename) in enumerate(training_generator):
        print(f"图片文件名: {filename}")
        break

    # for step, (patch, mask, _) in enumerate(training_generator):
    #     pass
    # print("One epoch done; steps: {}".format(step))
    
    
    
    # # 测试FIVE数据集加载
    # print("测试FIVE数据集加载...")
    
    # # 创建验证集
    # val_dst = FIVE(["/home/pxl/myProject/血管分割/molong-深度插值/molong-work/segmentation/data/FIVE/train/images",
    #                 "/home/pxl/myProject/血管分割/molong-深度插值/molong-work/segmentation/data/FIVE/train/1st_manual"],
    #                task="train",
    #                crop_size=128)
    # val_loader = data.DataLoader(val_dst,
    #                            batch_size=4, 
    #                            shuffle=False,
    #                            num_workers=4)
    
    # print(f"验证集大小: {len(val_dst)} 张图片")