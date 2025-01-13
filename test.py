
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
from lib.model.unet.unet_model import UNet
# from lib.unet.iternet_model import UNet_concate


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
    mydict['dataset'] = params['common']['dataset']

    if task == "train":
        mydict['train_datalist'] = params['train']['train_datalist']
        mydict['validation_datalist'] = params['train']['validation_datalist']
        mydict['output_folder'] = params['train']['output_folder']
        mydict['output_folder_flag'] = params['train']['output_folder_flag']
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
        saveFlag = args.saveFlag
        mydict['output_folder'] = os.path.join(params['test']['output_folder'], saveFlag)

    else:
        print("Wrong task chosen")
        sys.exit()


    if not os.path.exists(mydict['output_folder']):
        os.makedirs(mydict['output_folder'])
        
    print("保存路径:{}".format(mydict['output_folder']))

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


def test_2d(mydict):
    device = torch.device("cuda")
    print("CUDA device: {}".format(device))

    if not torch.cuda.is_available():
        print("WARNING!!! You are attempting to run training on a CPU (torch.cuda.is_available() is False). This can be VERY slow!")

    # Test Data
    dataset = mydict['dataset']
    print("Dataset Name: {}".format(dataset))
    if dataset == 'DRIVE':
        test_set = DRIVE(mydict['test_datalist'], mydict['folders'], task="test")
    elif(dataset == 'FIVE'):
        test_set = FIVE("",
                            ["/home/xkw/pxlames/segmentation/data/FIVE/test/images",
                            "/home/xkw/pxlames/segmentation/data/FIVE/test/1st_manual"],
                            task="test",
                            crop_size=128)    

    # 只读取前10个样本
    # test_subset = torch.utils.data.Subset(test_set, range(30))
    test_generator = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=2, drop_last=False)

    network = UNet(n_channels=1, n_classes=mydict['num_classes'], start_filters=64).to(device)

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

            y_pred_logit = network(x)
            y_pred = torch.sigmoid(y_pred_logit)
            '''补齐:'''
            np_gt = torch.squeeze(y_gt).detach().cpu().numpy()
            np_pred = torch.squeeze(y_pred).detach().cpu().numpy()
            
            # 加到集合中, 用于校准集的数据
            scores.append(np_pred)  # 将预测结果添加到 scores 列表中
            gt_masks.append(np_gt)  # 将真实标签添加到 gt_masks 列表中
            
            
            #是否保存图像：
            y_pred = RefineWithPrior().forward(y_pred)
            filename = filename[0]
            save_image(x, os.path.join(mydict['output_folder'], 'img_' + filename  + '.png'))
            save_image(torch.squeeze(y_gt*255), os.path.join(mydict['output_folder'], 'gt_' + filename + '.png')) 
            
            np_pred_result = np_pred.copy()
            np_pred_result = np.where(np_pred_result >= 0.5, 1., 0.) # 0.5 thresholding
            np_pred_result = (np_pred_result*255.).astype(np.uint8)
            img_pred_result = Image.fromarray(np_pred_result)
            img_pred_result.save(os.path.join(mydict['output_folder'], 'pred_' + filename + '.png'))

        scores_array = np.array(scores)
        np.save(os.path.join(mydict['output_folder'], 'scores.npy'), scores_array)
        gt_masks_array = np.array(gt_masks)
        np.save(os.path.join(mydict['output_folder'], 'gt_masks.npy'), gt_masks_array)


def test_2d_cal(mydict):
    device = torch.device("cuda")
    print("CUDA device: {}".format(device))

    if not torch.cuda.is_available():
        print("WARNING!!! You are attempting to run training on a CPU (torch.cuda.is_available() is False). This can be VERY slow!")

    # Test Data
    dataset = mydict['dataset']
    print("Dataset Name: {}".format(dataset))
    if dataset == 'DRIVE':
        test_set = DRIVE(mydict['test_datalist'], mydict['folders'], task="test")
    elif(dataset == 'FIVE'):
        test_set = FIVE("",
                            ["/home/xkw/pxlames/segmentation/data/FIVE/cal/images",
                            "/home/xkw/pxlames/segmentation/data/FIVE/cal/1st_manual"],
                            task="test",
                            crop_size=128)    

    # 只读取前10个样本
    # test_subset = torch.utils.data.Subset(test_set, range(30))
    test_generator = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=2, drop_last=False)

    network = UNet(n_channels=1, n_classes=mydict['num_classes'], start_filters=64).to(device)

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

            y_pred_logit = network(x)
            y_pred = torch.sigmoid(y_pred_logit)
            '''补齐:'''
            # y_pred = RefineWithPrior().forward(y_pred)
            filename = filename[0]
            np_gt = torch.squeeze(y_gt).detach().cpu().numpy()
            np_pred = torch.squeeze(y_pred).detach().cpu().numpy()
            scores.append(np_pred)  # 将预测结果添加到 scores 列表中
            gt_masks.append(np_gt)  # 将真实标签添加到 gt_masks 列表中
            
        # 将 scores 和 gt_masks 保存到当前文件夹中
        scores_array = np.array(scores)
        np.save(os.path.join(mydict['output_folder'], 'cal_scores.npy'), scores_array)
        gt_masks_array = np.array(gt_masks)
        np.save(os.path.join(mydict['output_folder'], 'cal_gt_masks.npy'), gt_masks_array)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', type=str, help="Path to the JSON parameters file")
    parser.add_argument('--saveFlag', type=str, help="")
    
    if len(sys.argv) == 1:
        print("Path to parameters file not provided. Exiting...")

    else:
        args = parser.parse_args()
        task, mydict = parse_func(args)

    test_2d(mydict)
    print('测试完成')
    test_2d_cal(mydict)
    print('cal测试完成')

