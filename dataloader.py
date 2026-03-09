# load all into cpu
# do cropping to patch size
# normalize range
# test code by outputting few patches
# training, testing, val
import cv2
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
from torchvision.utils import save_image
import sys
sys.path.append('/home/xkw/pxlames/segmentation/lib/Prior/fundus-vessels-toolkit/src')  # 修改为正确的路径

# from lib.LION import distance_weight_binary_pattern_faster

class DRIVE(data.Dataset):
    def __init__(self, listpath, folderpaths, task, crop_size = 128):

        self.listpath = listpath
        self.imgfolder = folderpaths[0]
        self.gtfolder = folderpaths[1]
        self.crop_size = crop_size

        self.dataCPU = {}
        self.dataCPU['image'] = []
        self.dataCPU['image_lvs'] = []
        self.dataCPU['label'] = []
        self.dataCPU['filename'] = []

        self.task = task
 
        self.to_tensor = transforms.ToTensor() # converts HWC in [0,255] to CHW in [0,1]
        self.is_lvs=False
        self.loadCPU()

    def cal_img_lvs(self,img,gt):
        from lib.vessel_salience import salience
        img = np.array(img, dtype=np.uint8)
        gt = np.array(gt, dtype=np.uint8)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gt = (gt[:,:,0] > 127).astype(np.uint8)
        img_lvs = salience.lvs(img,gt,radius=8,return_skel=False)
        return img_lvs
    
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

            if(self.task == "train"):
                if self.is_lvs:
                    img_lvs = self.cal_img_lvs(img,gt)
                    self.dataCPU['image_lvs'].append(img_lvs)
                    # 保存img_lvs到logs文件夹
                    save_path = 'lvs.png'
                    cv2.imwrite(save_path, (img_lvs * 255).astype(np.uint8))
                    print('计算完lvs一次')
                else:
                    self.dataCPU['image_lvs'].append(torch.tensor([]))  # 使用空列表创建tensor,可以用.numel()==0判断是否为空
                 
            img = np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
               
            gt = self.to_tensor(gt)
            img = self.to_tensor(img) 
            
            # savePath = '/home/xkw/pxlames/segmentation/data/DRIVE/train未归一化结果'
            # 确保保存路径存在
            # os.makedirs(savePath, exist_ok=True)
            # 将归一化后的图像保存为png文件
            # save_filename = os.path.join(savePath, filename + '_normalized.png')
            # 将tensor转换为PIL图像并保存
            # save_image(img,save_filename)
            # cpu store
   
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
            torch_img_lvs = self.dataCPU['image_lvs'][index] 

            # crop: compute top-left corner first
            _, H, W = torch_img.shape
           
            corner_h = np.random.randint(low=0, high=H-self.crop_size)
            corner_w = np.random.randint(low=0, high=W-self.crop_size)

            torch_img = torch_img[:, corner_h:corner_h+self.crop_size, corner_w:corner_w+self.crop_size]
            torch_gt = torch_gt[:, corner_h:corner_h+self.crop_size, corner_w:corner_w+self.crop_size]

            # 只有在torch_img_lvs不为空时才进行切片
            if self.is_lvs:  # 修改这里的判断条件
                # 将NumPy数组转换为PyTorch张量
                if isinstance(torch_img_lvs, np.ndarray):
                    torch_img_lvs = torch.from_numpy(torch_img_lvs)
                
                if torch_img_lvs.ndim == 2:
                    torch_img_lvs = torch_img_lvs.unsqueeze(0)
                    
                torch_img_lvs = torch_img_lvs[:, corner_h:corner_h+self.crop_size, corner_w:corner_w+self.crop_size]
        else:
            # 获取原始图像和标签
            torch_img = self.dataCPU['image'][index] #HW
            torch_gt = self.dataCPU['label'][index] #HW
            
            # 确保尺寸是32的倍数
            _, H, W = torch_img.shape
            
            # 计算最大的2的幂次方大小
            max_size = 1
            min_dim = min(H, W)
            while max_size * 2 <= min_dim:
                max_size *= 2
                
            # 裁剪到最大的2的幂次方大小
            torch_img = torch_img[:, :max_size, :max_size]
            torch_gt = torch_gt[:, :max_size, :max_size]
            
            
        if self.task == "train":
            return torch_img, torch_gt, torch_img_lvs, self.dataCPU['filename'][index]
        else:
            return torch_img, torch_gt, self.dataCPU['filename'][index]
    
    
       
class FIVE(data.Dataset):
    def __init__(self, listpath,  folderpaths, task, crop_size=128):

        self.imgfolder = folderpaths[0]
        self.gtfolder = folderpaths[1]
        self.crop_size = crop_size

        self.dataCPU = {}
        self.dataCPU['image'] = []
        self.dataCPU['image_lvs'] = []
        self.dataCPU['label'] = []
        self.dataCPU['filename'] = []

        self.task = task
 
        self.to_tensor = transforms.ToTensor() # converts HWC in [0,255] to CHW in [0,1]
        self.is_lvs=False
        self.loadCPU()

    def augment(self,img,gt):
        import numpy as np
        from lib.vessel_salience import augmentation

        # 增强参数设置
        rqi_len_interv = (30,40)    # 增强区域可能的长度范围(论文中的参数 l)
        min_len_interv = (5,10)     # 不连续区域可能的长度范围(论文中的参数 l_d) 
        n_rqi_interv = (50,55)        # 要增强的片段数量
        back_threshold = 30          # 搜索有效背景时的相似度阈值

        ##############################
        img = np.array(img, dtype=np.uint8)
        gt = np.array(gt, dtype=np.uint8)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gt = gt[:,:,0]

        # print(img.max())
        # print(gt.max())
        gt = (gt > 127).astype(np.float32)

        img_aug, visualizations, proto_graph, img_augmented_segs = augmentation.create_image(
            img_origin = img,
            img_label = gt,
            rqi_len_interv = rqi_len_interv, 
            min_len_interv = min_len_interv, 
            n_rqi_interv = n_rqi_interv,
            back_threshold = back_threshold,
            rng_seed = 1,
            highlight_center = False)
        
        return img_aug
    
    def cal_img_lvs(self,img,gt):
        from lib.vessel_salience import salience
        img = np.array(img, dtype=np.uint8)
        gt = np.array(gt, dtype=np.uint8)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gt = (gt[:,:,0] > 127).astype(np.uint8)
        img_lvs = salience.lvs(img,gt,radius=8,return_skel=False)
        return img_lvs
    
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
            if(self.task == "train"):
                if self.is_lvs:
                    img_lvs = self.cal_img_lvs(img,gt)
                    self.dataCPU['image_lvs'].append(img_lvs)
                    # 保存img_lvs到logs文件夹
                    save_path = 'lvs.png'
                    cv2.imwrite(save_path, (img_lvs * 255).astype(np.uint8))
                    print('计算完lvs一次')
                else:
                    self.dataCPU['image_lvs'].append(torch.tensor([]))  # 使用空列表创建tensor,可以用.numel()==0判断是否为空
                # img_aug = self.augment(img,gt)
                # img = img_aug
                # print("完成增强")
            img = np.array(img)
            # 是否转为一个通道！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gt = np.array(gt)
            gt = gt[:,:,0]
            

            img = self.to_tensor(img)     
            gt = self.to_tensor(gt)
            
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

        torch_img = self.dataCPU['image'][index] 
        torch_gt = self.dataCPU['label'][index] 
        torch_gt = torch_gt[0].unsqueeze(0)

        if self.task == "train":
            torch_img_lvs = self.dataCPU['image_lvs'][index] 
            
            _, H, W = torch_img.shape
            corner_h = np.random.randint(low=0, high=H-self.crop_size)
            corner_w = np.random.randint(low=0, high=W-self.crop_size)

            torch_img = torch_img[:, corner_h:corner_h+self.crop_size, corner_w:corner_w+self.crop_size]
            torch_gt = torch_gt[:, corner_h:corner_h+self.crop_size, corner_w:corner_w+self.crop_size]
            
            # 只有在torch_img_lvs不为空时才进行切片
            if self.is_lvs:  # 修改这里的判断条件
                # 将NumPy数组转换为PyTorch张量
                if isinstance(torch_img_lvs, np.ndarray):
                    torch_img_lvs = torch.from_numpy(torch_img_lvs)
                
                if torch_img_lvs.ndim == 2:
                    torch_img_lvs = torch_img_lvs.unsqueeze(0)
                    
                torch_img_lvs = torch_img_lvs[:, corner_h:corner_h+self.crop_size, corner_w:corner_w+self.crop_size]
        if self.task == "train":
            return torch_img, torch_gt, torch_img_lvs, self.dataCPU['filename'][index]
        else:
            return torch_img, torch_gt, self.dataCPU['filename'][index]
            

class FIVE_SECOND(data.Dataset):
    def __init__(self, listpath,  folderpaths, task, crop_size=128):

        self.imgfolder = folderpaths[0]
        self.gtfolder = folderpaths[1]
        self.Cfolder = folderpaths[2]
        self.Ufolder = folderpaths[3]
        self.crop_size = crop_size

        self.dataCPU = {}
        self.dataCPU['image'] = []
        self.dataCPU['image_lvs'] = []
        self.dataCPU['label'] = []
        self.dataCPU['filename'] = []
        self.dataCPU['torch_C_mask'] = []
        self.dataCPU['torch_U_mask'] = []

        self.task = task
 
        self.to_tensor = transforms.ToTensor() # converts HWC in [0,255] to CHW in [0,1]
        self.is_lvs=False
        self.loadCPU()

    def augment(self,img,gt):
        import numpy as np
        from lib.vessel_salience import augmentation

        # 增强参数设置
        rqi_len_interv = (30,40)    # 增强区域可能的长度范围(论文中的参数 l)
        min_len_interv = (5,10)     # 不连续区域可能的长度范围(论文中的参数 l_d) 
        n_rqi_interv = (50,55)        # 要增强的片段数量
        back_threshold = 30          # 搜索有效背景时的相似度阈值

        ##############################
        img = np.array(img, dtype=np.uint8)
        gt = np.array(gt, dtype=np.uint8)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gt = gt[:,:,0]

        # print(img.max())
        # print(gt.max())
        gt = (gt > 127).astype(np.float32)

        img_aug, visualizations, proto_graph, img_augmented_segs = augmentation.create_image(
            img_origin = img,
            img_label = gt,
            rqi_len_interv = rqi_len_interv, 
            min_len_interv = min_len_interv, 
            n_rqi_interv = n_rqi_interv,
            back_threshold = back_threshold,
            rng_seed = 1,
            highlight_center = False)
        
        return img_aug
    
    def cal_img_lvs(self,img,gt):
        from lib.vessel_salience import salience
        img = np.array(img, dtype=np.uint8)
        gt = np.array(gt, dtype=np.uint8)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gt = (gt[:,:,0] > 127).astype(np.uint8)
        img_lvs = salience.lvs(img,gt,radius=8,return_skel=False)
        return img_lvs
    
    def loadCPU(self):
        # 获取所有png图像文件路径
        img_list = glob.glob(pjoin(self.imgfolder, '*.png'))
        print(f"图像文件数量: {len(img_list)}")
        for im_path in img_list:
            # 从文件路径中提取文件名（不含扩展名）
            filename = os.path.splitext(os.path.basename(im_path))[0]
            
            # 构建对应的ground truth路径
            gt_path = pjoin(self.gtfolder, filename + '.png')
            C_path = pjoin(self.Cfolder, filename + '.png')
            U_path = pjoin(self.Ufolder, filename + '.png')
            
            # 读取图像
            img = Image.open(im_path)
            gt = Image.open(gt_path)
            C_mask = Image.open(C_path)
            U_mask = Image.open(U_path)
            if(self.task == "train"):
                if self.is_lvs:
                    img_lvs = self.cal_img_lvs(img,gt)
                    self.dataCPU['image_lvs'].append(img_lvs)
                    # 保存img_lvs到logs文件夹
                    save_path = 'lvs.png'
                    cv2.imwrite(save_path, (img_lvs * 255).astype(np.uint8))
                    print('计算完lvs一次')
                else:
                    self.dataCPU['image_lvs'].append(torch.tensor([]))  # 使用空列表创建tensor,可以用.numel()==0判断是否为空
                # img_aug = self.augment(img,gt)
                # img = img_aug
                # print("完成增强")
            img = np.array(img)
            # 是否转为一个通道！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gt = np.array(gt)
            C_mask = np.array(C_mask)
            U_mask = np.array(U_mask)
            gt = gt[:,:,0]
            
            img = self.to_tensor(img)     
            gt = self.to_tensor(gt)
            C_mask = self.to_tensor(C_mask)
            U_mask = self.to_tensor(U_mask)
            
            # normalize within a channel
            for j in range(img.shape[0]):
                meanval = img[j].mean()
                stdval = img[j].std()
                img[j] = (img[j] - meanval) / stdval
                
            # cpu store
            self.dataCPU['image'].append(img)
            self.dataCPU['label'].append(gt)
            self.dataCPU['torch_C_mask'].append(C_mask)
            self.dataCPU['torch_U_mask'].append(U_mask)
            self.dataCPU['filename'].append(filename)

    def __len__(self): # total number of 2D slices
        return len(self.dataCPU['filename'])

    def __getitem__(self, index): # select random crop and return CHW torch tensor

        torch_img = self.dataCPU['image'][index] 
        torch_gt = self.dataCPU['label'][index] 
        torch_C_mask = self.dataCPU['torch_C_mask'][index] 
        torch_U_mask = self.dataCPU['torch_U_mask'][index] 
        torch_gt = torch_gt[0].unsqueeze(0)

        if self.task == "train":
            torch_img_lvs = self.dataCPU['image_lvs'][index] 
            
            _, H, W = torch_img.shape
            corner_h = np.random.randint(low=0, high=H-self.crop_size)
            corner_w = np.random.randint(low=0, high=W-self.crop_size)

            torch_img = torch_img[:, corner_h:corner_h+self.crop_size, corner_w:corner_w+self.crop_size]
            torch_gt = torch_gt[:, corner_h:corner_h+self.crop_size, corner_w:corner_w+self.crop_size]
            torch_C_mask = torch_C_mask[:, corner_h:corner_h+self.crop_size, corner_w:corner_w+self.crop_size]
            torch_U_mask = torch_U_mask[:, corner_h:corner_h+self.crop_size, corner_w:corner_w+self.crop_size]
            
            # 只有在torch_img_lvs不为空时才进行切片
            if self.is_lvs:  # 修改这里的判断条件
                # 将NumPy数组转换为PyTorch张量
                if isinstance(torch_img_lvs, np.ndarray):
                    torch_img_lvs = torch.from_numpy(torch_img_lvs)
                
                if torch_img_lvs.ndim == 2:
                    torch_img_lvs = torch_img_lvs.unsqueeze(0)
                    
                torch_img_lvs = torch_img_lvs[:, corner_h:corner_h+self.crop_size, corner_w:corner_w+self.crop_size]
                
        network_input = torch.cat([
                # torch_img,
                torch_C_mask,
                torch_U_mask,
            ], dim=0)
        if self.task == "train":
            return torch_img, torch_gt, torch_img_lvs, torch_C_mask, torch_U_mask, network_input, self.dataCPU['filename'][index]
        else:
            return torch_img, torch_gt, torch_C_mask, torch_U_mask, network_input, self.dataCPU['filename'][index]
            

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