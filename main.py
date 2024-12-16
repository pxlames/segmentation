'''
Commands:

Train:
CUDA_VISIBLE_DEVICES=3 python3 main.py --params ./datalists/DRIVE/train.json
Ensure crop_size in json is divisible by 16

Test/Inference:
CUDA_VISIBLE_DEVICES=4 python3 main.py --params ./datalists/DRIVE/test.json

Compute Evaluation Metrics (Quantitative Results):


Dataset properties:
GT: Foreground should be 255 ; Background should be 0
'''
from matplotlib import pyplot as plt
import wandb

import torch
torch.cuda.empty_cache()
from torchvision.utils import save_image

import numpy as np

import argparse, json
import os, glob, sys
from time import time

from dataloader import DRIVE, FIVE
from lib.unet.unet_model import UNet
from lib.unet.iternet_model import UNet_concate


from lib.topoloss_pd import TopoLossMSE2D
from PIL import Image

'''loss选择'''
from lib.metric_utilities import torch_dice_fn_bce,torch_betti_error_loss
from lib.no_three_weightedLossWithPrior import WeightedLossWithPrior,RefineWithPrior
from lib.skeletonloss import SkeletonLoss

def parse_func(args):
    ### Reading the parameters json file
    print("Reading params file {}...".format(args.params))
    with open(args.params, 'r') as f:
        params = json.load(f)

    task = params['common']['task']
    mydict = {}
    mydict['num_classes'] = int(params['common']['num_classes'])
    mydict['folders'] = [params['common']['img_folder'], params['common']['gt_folder']]
    mydict["checkpoint_restore"] = params['common']['checkpoint_restore']

    if task == "train":
        mydict['train_datalist'] = params['train']['train_datalist']
        mydict['validation_datalist'] = params['train']['validation_datalist']
        mydict['output_folder'] = params['train']['output_folder']
        mydict['crop_size'] = params['train']['crop_size']
        mydict['train_batch_size'] = int(params['train']['train_batch_size'])
        mydict['val_batch_size'] = int(params['train']['val_batch_size'])
        mydict['learning_rate'] = float(params['train']['learning_rate'])
        mydict['num_epochs'] = int(params['train']['num_epochs']) + 1
        mydict['save_every'] = params['train']['save_every']
        mydict['topo_weight'] = params['train']['topo_weight'] # If 0 => not training with topoloss
        mydict['topo_window'] = params['train']['topo_window']

    elif task == "test":
        mydict['test_datalist'] = params['test']['test_datalist']
        mydict['output_folder'] = params['test']['output_folder']

    else:
        print("Wrong task chosen")
        sys.exit()


    if not os.path.exists(mydict['output_folder']):
        os.makedirs(mydict['output_folder'])

    print(task, mydict)
    return task, mydict

def set_seed(): # reproductibility 
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

def force_cudnn_initialization():
    s = 32
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))

def train_2d(mydict):
    set_seed()
    device = torch.device("cuda")
    print("CUDA device: {}".format(device))
    if not torch.cuda.is_available():
        print("WARNING!!! You are attempting to run training on a CPU (torch.cuda.is_available() is False). This can be VERY slow!")
    force_cudnn_initialization()
    dataset = 'DRIVE'
    if(dataset == 'DRIVE'):
        # Train Data       
        training_set = DRIVE(mydict['train_datalist'], mydict['folders'], task="train")
        training_generator = torch.utils.data.DataLoader(training_set, batch_size=mydict['train_batch_size'], shuffle=True, num_workers=2, drop_last=True)
        # Validation Data
        validation_set = DRIVE(mydict['validation_datalist'], mydict['folders'], task="val")
        validation_generator = torch.utils.data.DataLoader(validation_set, batch_size=mydict['val_batch_size'], shuffle=False, num_workers=2, drop_last=False)
    elif(dataset == 'FIVE'):
        training_set = FIVE(["/home/pxl/myProject/血管分割/molong-深度插值/molong-work/segmentation/data/FIVE/train/images",
                            "/home/pxl/myProject/血管分割/molong-深度插值/molong-work/segmentation/data/FIVE/train/1st_manual"],
                            task="train",
                            crop_size=128)
        # 将数据集分为训练集和验证集
        total_size = len(training_set)
        indices = list(range(total_size))
        train_size = int(0.8 * total_size)  # 80%用于训练
        val_size = total_size - train_size   # 20%用于验证
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        train_dataset = torch.utils.data.Subset(training_set, train_indices)
        val_dataset = torch.utils.data.Subset(training_set, val_indices)
        
        # 创建训练集和验证集的数据加载器
        training_generator = torch.utils.data.DataLoader(train_dataset, batch_size=mydict['train_batch_size'], 
                                                       shuffle=True, num_workers=2, drop_last=True)
        validation_generator = torch.utils.data.DataLoader(val_dataset, batch_size=mydict['val_batch_size'],
                                                         shuffle=False, num_workers=2, drop_last=False)
    # Network
    # network = UNet(n_channels=3, n_classes=mydict['num_classes'], start_filters=64).to(device)
    network = UNet_concate(n_channels=7, n_classes=mydict['num_classes'])

    # Optimizer
    optimizer = torch.optim.Adam(network.parameters(), lr=mydict['learning_rate'], weight_decay=0)
    # Load checkpoint (if specified)
    if mydict['checkpoint_restore'] != "":
        network.load_state_dict(torch.load(mydict['checkpoint_restore']), strict=True)
        print("loaded checkpoint! {}".format(mydict['checkpoint_restore']))
    # Loss
    bce_loss_func = torch.nn.BCELoss(size_average = False, reduce=False, reduction=None)
    
    # topo_loss_func = TopoLossMSE2D(mydict['topo_weight'], mydict['topo_window'])
    # graph_topo_func = WeightedLossWithPrior(device=device,gradDebug=False)
    
    # Train loop
    best_dict = {}
    best_dict['epoch'] = 0
    best_dict['val_loss'] = None

    print("Let the training begin!")
    num_batches = len(training_generator)
    # experiment = wandb.init(project='U-Net', resume='allow', anonymous='must',mode='online',dir='./')
    # experiment.config.update(
    #     dict(epochs=mydict['num_epochs'], batch_size=mydict['train_batch_size'], learning_rate=mydict['learning_rate'])
    # )
    for epoch in range(mydict['num_epochs']):
        torch.cuda.empty_cache() # cleanup 
        network.to(device).train() # after .eval() in validation

        avg_train_loss = 0.0
        epoch_start_time = time()

        for step, (patch, mask, _) in enumerate(training_generator): 
            # 每几个批次清理一次内存
            if step % 10 == 0:
                torch.cuda.empty_cache()
            optimizer.zero_grad()

            patch = patch.to(device, dtype=torch.float)
            mask = mask.to(device, dtype=torch.float)
            #mask = mask.type(torch.LongTensor).to(device)
            y_pred = torch.sigmoid(network(patch)[2])  # sigmoid needed for BCE (else throws error if not in 0,1 range)
            lamda1 = 1 - (epoch / mydict['num_epochs'])
            lamda3 = epoch / mydict['num_epochs']
            BCE_loss = torch.mean(bce_loss_func(y_pred, mask))  # 1.BCE
            loss_val = BCE_loss
            dice_loss = 1-torch_dice_fn_bce(y_pred, mask)
            loss_val = loss_val + dice_loss # 2.diceloss
            if mydict['topo_weight'] != 0:
                # loss2 = topo_loss_func(y_pred, mask)
                # loss_val += mydict['topo_weight'] * loss2 # 2.topo
                # loss3 = graph_topo_func(y_pred, mask)
                # loss_val += mydict['topo_weight'] * loss3 # 3.topo2
                # print(f'topo: {loss2.item()}  改进loss: {loss3.item()}')
                '''SkeletonLoss'''
                # 将batch也保存
                # 遍历batch中的每一张图片
                '''
                for i in range(patch.shape[0]):
                    img = patch[i].cpu().permute(1, 2, 0).numpy()  # 将通道维度移到最后

                    # 提取前三个通道作为 RGB 图像
                    img_rgb = img[..., :3]  # 只取前三个通道

                    # 归一化到 [0, 1] 范围
                    img_rgb = (img_rgb - img_rgb.min()) / (img_rgb.max() - img_rgb.min())

                    # 确保值在 [0, 1] 范围内
                    img_rgb = np.clip(img_rgb, 0, 1)

                    # 显示并保存前三个通道作为 RGB 图像
                    plt.imshow(img_rgb)
                    plt.axis('off')
                    plt.savefig(f'patch_{i}_rgb.png', bbox_inches='tight', pad_inches=0)  # 保存 RGB 图像
                    plt.close()

                    # 提取第四个通道作为灰度图像
                    img_gray = img[..., 3]  # 只取第四个通道

                    # 归一化到 [0, 1] 范围
                    img_gray = (img_gray - img_gray.min()) / (img_gray.max() - img_gray.min())

                    # 确保值在 [0, 1] 范围内
                    img_gray = np.clip(img_gray, 0, 1)

                    # 显示并保存第四个通道作为灰度图像
                    plt.imshow(img_gray, cmap='gray')  # 显示为灰度图像
                    plt.axis('off')
                    plt.savefig(f'patch_{i}_gray.png', bbox_inches='tight', pad_inches=0)  # 保存灰度图像
                    plt.close()
                    '''
                loss_fn = SkeletonLoss(filter_p=0.5, loss='BCELoss', reduction='mean', normalization='Sigmoid', dilation_iter=0)
                skeletonLoss = loss_fn(y_pred, mask)
                with open('skeleton_loss.txt', 'a') as file:
                    file.write(f'BCE_loss: {BCE_loss}\n')
                    file.write(f'dice_loss: {dice_loss}\n')
                    file.write(f'SkeletonLoss: {skeletonLoss.item()}\n')
                # 方式2：使用余弦退火，后期权重更大
                # max_epochs = mydict['num_epochs']
                # skeleton_weight = lamda3 * 0.001 * (1 - np.cos(2 * epoch / max_epochs * np.pi)) + 0.001
                # print(f' skeleton_weight: {skeleton_weight}, skeletonLoss: {skeleton_weight*skeletonLoss.item()}, loss_val: {loss_val.item()}')
                loss_val += skeletonLoss.item()
                
                ##########
                # experiment.log({
                #     # 'base_ce_loss': base_loss.item(),
                #     # 'weighted_ce_loss': loss2.item(),
                #     # '改进loss': loss3.item(),
                #     '总train loss': loss2.item(),
                #     'epoch': epoch
                # })
                ##########
            avg_train_loss += loss_val.item()

            loss_val.backward()
            optimizer.step()

        avg_train_loss /= num_batches
        epoch_end_time = time()
        print("Epoch {} Average training loss: {}".format(epoch, avg_train_loss))

        validation_start_time = time()
        with torch.no_grad():
            network.eval()
            validation_iterator = iter(validation_generator)
            avg_val_loss = 0.0
            for _ in range(len(validation_generator)):
                x, y_gt, _ = next(validation_iterator)
                x = x.to(device, non_blocking=True)
                y_gt = y_gt.to(device, non_blocking=True)
                
                y_pred = torch.sigmoid(network(x)[2])
                avg_val_loss += torch_dice_fn_bce(y_pred, y_gt)

            avg_val_loss /= len(validation_generator)
        validation_end_time = time()
        print("Average validation dice: {}".format(avg_val_loss))

        # check for best epoch and save it if it is and print
        if epoch == 0:
            best_dict['epoch'] = epoch
            best_dict['val_loss'] = avg_val_loss
        elif best_dict['val_loss'] < avg_val_loss:
                best_dict['val_loss'] = avg_val_loss
                best_dict['epoch'] = epoch

        if epoch == best_dict['epoch']:
            torch.save(network.state_dict(), os.path.join(mydict['output_folder'], "model_best.pth"))
        print("Best epoch so far: {}\n".format(best_dict))
        # save checkpoint for save_every
        if epoch % mydict['save_every'] == 0:
            torch.save(network.state_dict(), os.path.join(mydict['output_folder'], "model_epoch" + str(epoch) + ".pth"))


def test_2d(mydict):
    device = torch.device("cuda")
    print("CUDA device: {}".format(device))

    if not torch.cuda.is_available():
        print("WARNING!!! You are attempting to run training on a CPU (torch.cuda.is_available() is False). This can be VERY slow!")

    # Test Data
    dataset = 'DRIVE'
    if dataset == 'DRIVE':
        test_set = DRIVE(mydict['test_datalist'], mydict['folders'], task="test")
    elif dataset == 'FIVE':
        test_set = FIVE(["/home/pxl/myProject/血管分割/molong-深度插值/molong-work/segmentation/data/FIVE/test/images",
                            "/home/pxl/myProject/血管分割/molong-深度插值/molong-work/segmentation/data/FIVE/test/1st_manual"],
                            task="test",
                            crop_size=256)        
    n_channels = 3

    # 只读取前10个样本
    test_subset = torch.utils.data.Subset(test_set, range(30))
    test_generator = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=2, drop_last=False)

    network = UNet_concate(n_channels=7, n_classes=mydict['num_classes'])

    if mydict['checkpoint_restore'] != "":
        network.load_state_dict(torch.load(mydict['checkpoint_restore']), strict=True)
        print("loaded checkpoint! {}".format(mydict['checkpoint_restore']))
    else:
        print("No model found!")
        sys.exit()

    print("Let the inference begin!")
    print("Todo: {}".format(len(test_generator)))

    with torch.no_grad():
        # 校准添加,测试阶段全部生成即可. 
        scores = []
        gt_masks = []
        network.eval()
        test_iterator = iter(test_generator)
        for _ in range(len(test_generator)):
            x, y_gt, filename = next(test_iterator)
            x = x.to(device, non_blocking=True)
            y_gt = y_gt.to(device, non_blocking=True)

            y_pred_logit = network(x)[2]
            y_pred = torch.sigmoid(y_pred_logit)
            '''补齐:'''
            # y_pred = RefineWithPrior().forward(y_pred)
            filename = filename[0]
            save_image(x, os.path.join(mydict['output_folder'], 'img_' + filename  + '.png'))
            save_image(torch.squeeze(y_gt*255), os.path.join(mydict['output_folder'], 'gt_' + filename + '.png'))  # can be used since using BCE with num_classes=1

            np_gt = torch.squeeze(y_gt).detach().cpu().numpy()

            np_pred = torch.squeeze(y_pred).detach().cpu().numpy()
            # np.save(os.path.join(mydict['output_folder'], 'pred_' + filename + '.npy'), np_pred)
            
            # 加到集合中, 用于校准集的数据
            scores.append(np_pred)  # 将预测结果添加到 scores 列表中
            gt_masks.append(np_gt)  # 将真实标签添加到 gt_masks 列表中
            
            np_pred_result = np_pred.copy()
            np_pred_result = np.where(np_pred_result >= 0.5, 1., 0.) # 0.5 thresholding
            np_pred_result = (np_pred_result*255.).astype(np.uint8)
            img_pred_result = Image.fromarray(np_pred_result)
            img_pred_result.save(os.path.join(mydict['output_folder'], 'pred_' + filename + '.png'))

            # # 定义阈值列表
            # thresholds = [0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0]  # [0.0, 0.1, 0.2, ..., 0.9]

            # # 用于存储每个阈值处理后的图片
            # images = []

            # # 初始化一个变量来存储上一个阈值的结果
            # previous_tp = None

            # # 生成二值化图片并保存
            # for index, threshold in enumerate(thresholds):
            #     # 使用不同的阈值进行二值化
            #     np_pred_binary = np.where(np_pred >= threshold, 1., 0.)
                
            #     # 创建颜色对比图
            #     comparison_image = np.zeros((np_pred.shape[0], np_pred.shape[1], 3), dtype=np.uint8)

            #     # 假阳性：np_pred_binary 为 1 而 np_gt 为 0
            #     comparison_image[(np_pred_binary == 1) & (np_gt == 0)] = [0, 0, 255]  # 蓝色
                
            #     # 假阴性：np_pred_binary 为 0 而 np_gt 为 1
            #     comparison_image[(np_pred_binary == 0) & (np_gt == 1)] = [255, 0, 0]  # 红色
                
            #     # 正确预测：np_pred_binary 与 np_gt 相同
            #     comparison_image[(np_pred_binary == 1) & (np_gt == 1)] = [255, 255, 255]  # 白色
            #     comparison_image[(np_pred_binary == 0) & (np_gt == 0)] = [0, 0, 0]  # 黑色

            #     # 对于最右边的第一张图，直接保存，不用检查新增 TP
            #     if index == 0:
            #         previous_tp = (np_pred_binary == 1) & (np_gt == 1)  # 记录第一张的 TP
            #     else:
            #         # 找到相对于前一个阈值新增的 TP
            #         new_tp = (np_pred_binary == 1) & (np_gt == 1) & (~previous_tp)
            #         comparison_image[new_tp] = [0, 255, 0]  # 绿色，标记新增的 TP

            #         # 更新 previous_tp
            #         previous_tp = previous_tp | ((np_pred_binary == 1) & (np_gt == 1))

            #     # 将数组转换为图片
            #     im_pred_comparison = Image.fromarray(comparison_image)
                
            #     # 收集所有图片到列表中
            #     images.append(im_pred_comparison)

            # # 将所有图片按列排列在一起，生成最终的组合图片
            # width, height = images[0].size
            # combined_image = Image.new('RGB', (width * len(images), height))

            # # 将每张图片粘贴到组合图片中
            # for i, img in enumerate(images):
            #     combined_image.paste(img, (i * width, 0))

            # # 保存最终的组合图片
            # output_folder = mydict['output_folder']
            # combined_image.save(os.path.join(output_folder, 'combined_comparison_' + filename + '.png'))

            # print("图像对比已完成并保存。")
            
            # 将 scores 和 gt_masks 保存到当前文件夹中
            # 将 scores 转为 numpy 数组并保存
            scores_array = np.array(scores)
            np.save(os.path.join(mydict['output_folder'], 'scores.npy'), scores_array)

            # 将 gt_masks 转为 numpy 数组并保存
            gt_masks_array = np.array(gt_masks)
            np.save(os.path.join(mydict['output_folder'], 'gt_masks.npy'), gt_masks_array)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', type=str, help="Path to the JSON parameters file")
    
    if len(sys.argv) == 1:
        print("Path to parameters file not provided. Exiting...")

    else:
        args = parser.parse_args()
        task, mydict = parse_func(args)

    if task == "train":
        train_2d(mydict)
    elif task == "test":
        test_2d(mydict)
