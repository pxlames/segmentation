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

from dataloader import DRIVE, FIVE,FIVE_SECOND
from lib.model.unet.unet_model import UNet
from lib.model.unet_second.unet_model1 import UNet_SECOND
from lib.model.ResUnet.res_unet import ResUnet
from lib.model.ResUnet.res_unet_plus import ResUnetPlusPlus
from lib.model.ResUnet.res_unet_plus2 import build_resunetplusplus
from lib.model.unetplusplus.unetplusplus import ResNet34UnetPlus
from lib.model.TransUnet.vit_seg_modeling import VisionTransformer as ViT_seg
from lib.model.TransUnet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
# from lib.unet.iternet_model import UNet_concate

from calculate_loss import LossCalculator
from optimizer import create_optimizer

from PIL import Image

'''loss选择'''
from lib.metric_utilities import torch_dice_fn_bce,torch_betti_error_loss
from lib.no_three_weightedLossWithPrior import WeightedLossWithPrior,RefineWithPrior

def parse_func(args):
    ### Reading the parameters json file
    print("Reading params file {}...".format(args.params))
    with open(args.params, 'r') as f:
        params = json.load(f)
    # 选择不同任务的配置参数：
    configName = args.configName
    params = params[configName]
    task = params['common']['task']
    mydict = {}
    mydict['num_classes'] = int(params['common']['num_classes'])
    mydict['folders'] = [params['common']['img_folder'], params['common']['gt_folder'], params['common']['Cfolder'], params['common']['Ufolder']]
    mydict["checkpoint_restore"] = params['common']['checkpoint_restore']
    mydict['dataset'] = params['common']['dataset']
    mydict['loss'] = params['common']['loss_config']
    mydict['model'] = params['common']['model']
    
    if task == "train":
        mydict['train_datalist'] = params['train']['train_datalist']
        mydict['validation_datalist'] = params['train']['validation_datalist']
        mydict['output_folder'] = params['train']['output_folder']
        mydict['crop_size'] = params['train']['crop_size']
        mydict['train_batch_size'] = int(params['train']['train_batch_size'])
        mydict['val_batch_size'] = int(params['train']['val_batch_size'])
        mydict['learning_rate'] = float(params['train']['learning_rate'])
        mydict['mode'] = params['train']['mode']
        mydict['momentum'] = float(params['train']['momentum'])
        mydict['weight_decay'] = float(params['train']['weight_decay'])
        mydict['num_epochs'] = int(params['train']['num_epochs']) + 1
        mydict['save_every'] = params['train']['save_every']
        mydict['topo_weight'] = params['train']['topo_weight'] # If 0 => not training with topoloss
        mydict['topo_window'] = params['train']['topo_window']
        saveFlag = args.saveFlag
        mydict['output_folder'] = os.path.join(params['train']['output_folder'], saveFlag)

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
    print("Batch Size: {}".format(['train_batch_size']))
    dataset = mydict['dataset']
    print("Dataset Name: {}".format(dataset))
    if(dataset == 'DRIVE'):
        # Train Data       
        training_set = DRIVE(mydict['train_datalist'], mydict['folders'], task="train",crop_size=mydict['crop_size'])
        training_generator = torch.utils.data.DataLoader(training_set, batch_size=mydict['train_batch_size'], shuffle=True, num_workers=0, drop_last=True)
        # Validation Data
        validation_set = DRIVE(mydict['validation_datalist'], mydict['folders'], task="val",crop_size=mydict['crop_size'])
        validation_generator = torch.utils.data.DataLoader(validation_set, batch_size=mydict['val_batch_size'], shuffle=False, num_workers=0, drop_last=False)
    elif(dataset == 'FIVE'):
        training_set = FIVE("",
                            ["/home/xkw/pxlames/segmentation/data/FIVE/train/images",
                            "/home/xkw/pxlames/segmentation/data/FIVE/train/1st_manual"],
                            task="train",
                            crop_size=mydict['crop_size'])
        training_generator = torch.utils.data.DataLoader(training_set, batch_size=mydict['train_batch_size'], 
                                                       shuffle=True, num_workers=0, drop_last=True)
        
        validation_set = FIVE("",
                            ["/home/xkw/pxlames/segmentation/data/FIVE/val/images",
                            "/home/xkw/pxlames/segmentation/data/FIVE/val/1st_manual"],
                            task="val",
                            crop_size=mydict['crop_size'])
        validation_generator = torch.utils.data.DataLoader(validation_set, batch_size=mydict['val_batch_size'],
                                                         shuffle=False, num_workers=0, drop_last=False)
    # Network
    model = mydict['model']
    if(model == 'unet'):
        network = UNet(n_channels=1, n_classes=mydict['num_classes'], start_filters=64).to(device)
    elif(model == 'res_unet'):
        network = ResUnet(channel=1).to(device)
    elif(model == 'res_unet_plus'):
        network = ResUnetPlusPlus(channel=1).to(device)
        # network = build_resunetplusplus()
    elif(model == 'unetplusplus'):
        network = ResNet34UnetPlus(3,1).to(device)
    elif(model == 'trans_unet'):
        config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
        config_vit['n_skip'] = 3
        network = ViT_seg(config_vit, img_size=mydict['crop_size'], num_classes=1).cuda()
        config_vit = CONFIGS_ViT_seg['val']
        config_vit['n_skip'] = 3
        network_val = ViT_seg(config_vit, img_size=512, num_classes=1).cuda()

    # network = UNet_concate(n_channels=7, n_classes=mydict['num_classes'])
    # 从配置文件中获取损失函数配置
    lossCalculator = LossCalculator(mydict['loss'])
    # Optimizer
    optimizer = create_optimizer(network.parameters(),mydict['mode'],lr=mydict['learning_rate'],weight_decay=mydict['weight_decay'])
    # Load checkpoint (if specified)
    if mydict['checkpoint_restore'] != "":
        network.load_state_dict(torch.load(mydict['checkpoint_restore']), strict=True)
        print("loaded checkpoint! {}".format(mydict['checkpoint_restore']))
    
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

        for step, (patch, mask, patch_lvs, _) in enumerate(training_generator): 
            # 每几个批次清理一次内存
            if step % 10 == 0:
                torch.cuda.empty_cache()
            optimizer.zero_grad()
            patch = patch.to(device, dtype=torch.float)
            mask = mask.to(device, dtype=torch.float)
            patch_lvs = patch_lvs.to(device, dtype=torch.float)
            #mask = mask.type(torch.LongTensor).to(device)
            y_pred = torch.sigmoid(network(patch))  # sigmoid needed for BCE (else throws error if not in 0,1 range)

            loss_val = lossCalculator.compute_loss(y_pred, mask, patch_lvs, epoch)
            # 使用总损失进行反向传播
            loss_val = loss_val['total']
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
                # 每个epoch结束后绘制损失曲线
        if (epoch + 1) % 10 == 0:
            save_dir = mydict['output_folder']
            lossCalculator.plot_loss_history(save_dir=save_dir)
            print(f"损失曲线已保存至: {save_dir}")
            
            
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
                if(model == 'trans_unet'):
                    y_pred = torch.sigmoid(network_val(x))
                else:
                    y_pred = torch.sigmoid(network(x))
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
            os.makedirs(mydict['output_folder'], exist_ok=True)
            save_path = os.path.join(mydict['output_folder'], "model_best.pth")
            torch.save(network.state_dict(), save_path)
        print("Best epoch so far: {}\n".format(best_dict))
        # save checkpoint for save_every
        # if epoch % mydict['save_every'] == 0:
        #     torch.save(network.state_dict(), os.path.join(mydict['output_folder'], "model_epoch" + str(epoch) + ".pth"))

def train_2d_second(mydict):
    set_seed()
    device = torch.device("cuda")
    print("CUDA device: {}".format(device))
    if not torch.cuda.is_available():
        print("WARNING!!! You are attempting to run training on a CPU (torch.cuda.is_available() is False). This can be VERY slow!")
    force_cudnn_initialization()
    print("Batch Size: {}".format(['train_batch_size']))
    dataset = mydict['dataset']
    print("Dataset Name: {}".format(dataset))
    if(dataset == 'DRIVE'):
        # Train Data       
        training_set = DRIVE(mydict['train_datalist'], mydict['folders'], task="train",crop_size=mydict['crop_size'])
        training_generator = torch.utils.data.DataLoader(training_set, batch_size=mydict['train_batch_size'], shuffle=True, num_workers=0, drop_last=True)
        # Validation Data
        validation_set = DRIVE(mydict['validation_datalist'], mydict['folders'], task="val",crop_size=mydict['crop_size'])
        validation_generator = torch.utils.data.DataLoader(validation_set, batch_size=mydict['val_batch_size'], shuffle=False, num_workers=0, drop_last=False)
    elif(dataset == 'FIVE'):
        training_set = FIVE_SECOND("",
                            ["/home/xkw/pxlames/segmentation/data/FIVE/train/images",
                            "/home/xkw/pxlames/segmentation/data/FIVE/train/1st_manual",
                            "/home/xkw/pxlames/segmentation/outputs/firstStageResults/FIVE-BceDiceSmooth_0.0001_4/FIVE3/train/Cfolder",
                            "/home/xkw/pxlames/segmentation/outputs/firstStageResults/FIVE-BceDiceSmooth_0.0001_4/FIVE3/train/Ufolder"],
                            task="train",
                            crop_size=mydict['crop_size'])
        training_generator = torch.utils.data.DataLoader(training_set, batch_size=mydict['train_batch_size'], 
                                                       shuffle=True, num_workers=0, drop_last=True)
        
        validation_set = FIVE_SECOND("",
                            ["/home/xkw/pxlames/segmentation/data/FIVE/val/images",
                            "/home/xkw/pxlames/segmentation/data/FIVE/val/1st_manual",
                            "/home/xkw/pxlames/segmentation/outputs/firstStageResults/FIVE-BceDiceSmooth_0.0001_4/FIVE3/val/Cfolder",
                            "/home/xkw/pxlames/segmentation/outputs/firstStageResults/FIVE-BceDiceSmooth_0.0001_4/FIVE3/val/Ufolder"],
                            task="val",
                            crop_size=mydict['crop_size'])
        validation_generator = torch.utils.data.DataLoader(validation_set, batch_size=mydict['val_batch_size'],
                                                         shuffle=False, num_workers=0, drop_last=False)
    # Network
    model = mydict['model']
    if(model == 'unet'):
        network = UNet(n_channels=2, n_classes=mydict['num_classes'], start_filters=64).to(device)
    elif(model == 'unet_second'):
        network = UNet_SECOND(n_channels=2, n_classes=mydict['num_classes'], start_filters=64).to(device)
    elif(model == 'res_unet'):
        network = ResUnet(channel=1).to(device)
    elif(model == 'res_unet_plus'):
        network = ResUnetPlusPlus(channel=1).to(device)
        # network = build_resunetplusplus()
    elif(model == 'unetplusplus'):
        network = ResNet34UnetPlus(1,1).to(device)
    elif(model == 'trans_unet'):
        config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
        config_vit['n_skip'] = 3
        network = ViT_seg(config_vit, img_size=mydict['crop_size'], num_classes=1).cuda()
        config_vit = CONFIGS_ViT_seg['val']
        config_vit['n_skip'] = 3
        network_val = ViT_seg(config_vit, img_size=512, num_classes=1).cuda()

    # network = UNet_concate(n_channels=7, n_classes=mydict['num_classes'])
    # 从配置文件中获取损失函数配置
    lossCalculator = LossCalculator(mydict['loss'])
    # Optimizer
    optimizer = create_optimizer(network.parameters(),mydict['mode'],lr=mydict['learning_rate'],weight_decay=mydict['weight_decay'])
    # Load checkpoint (if specified)
    if mydict['checkpoint_restore'] != "":
        network.load_state_dict(torch.load(mydict['checkpoint_restore']), strict=True)
        print("loaded checkpoint! {}".format(mydict['checkpoint_restore']))
    
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

        for step, (patch, mask, patch_lvs, torch_C_mask, torch_U_mask, network_input,filename) in enumerate(training_generator): 
            # 每几个批次清理一次内存
            if step % 10 == 0:
                torch.cuda.empty_cache()
            optimizer.zero_grad()
            patch = patch.to(device, dtype=torch.float)
            mask = mask.to(device, dtype=torch.float)
            patch_lvs = patch_lvs.to(device, dtype=torch.float)
            torch_C_mask = torch_C_mask.to(device, dtype=torch.float)
            torch_U_mask = torch_U_mask.to(device, dtype=torch.float)
            network_input = network_input.to(device, dtype=torch.float)
            #mask = mask.type(torch.LongTensor).to(device)
            y_pred = torch.sigmoid(network(network_input))  # sigmoid needed for BCE (else throws error if not in 0,1 range)

            loss_val = lossCalculator.compute_loss(y_pred, mask, patch_lvs, epoch, torch_C_mask, torch_U_mask)
            # 使用总损失进行反向传播
            loss_val = loss_val['total']
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
                # 每个epoch结束后绘制损失曲线
        if (epoch + 1) % 10 == 0:
            save_dir = mydict['output_folder']
            lossCalculator.plot_loss_history(save_dir=save_dir)
            print(f"损失曲线已保存至: {save_dir}")
            
            
        avg_train_loss /= num_batches
        epoch_end_time = time()
        print("Epoch {} Average training loss: {}".format(epoch, avg_train_loss))

        validation_start_time = time()
        with torch.no_grad():
            network.eval()
            validation_iterator = iter(validation_generator)
            avg_val_loss = 0.0
            for _ in range(len(validation_generator)):
                x, y_gt, torch_C_mask, torch_U_mask, network_input, filename = next(validation_iterator)
                x = x.to(device, non_blocking=True)
                y_gt = y_gt.to(device, non_blocking=True)
                torch_C_mask = torch_C_mask.to(device, non_blocking=True)
                torch_U_mask = torch_U_mask.to(device, non_blocking=True)
                network_input = network_input.to(device, non_blocking=True)
                if(model == 'trans_unet'):
                    y_pred = torch.sigmoid(network_val(x))
                else:
                    y_pred = torch.sigmoid(network(network_input))
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
            os.makedirs(mydict['output_folder'], exist_ok=True)
            save_path = os.path.join(mydict['output_folder'], "model_best.pth")
            torch.save(network.state_dict(), save_path)
        print("Best epoch so far: {}\n".format(best_dict))
        # save checkpoint for save_every
        # if epoch % mydict['save_every'] == 0:
        #     torch.save(network.state_dict(), os.path.join(mydict['output_folder'], "model_epoch" + str(epoch) + ".pth"))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', type=str, help="Path to the JSON parameters file")
    parser.add_argument('--configName', type=str, help="")
    parser.add_argument('--saveFlag', type=str, help="")
    
    if len(sys.argv) == 1:
        print("Path to parameters file not provided. Exiting...")

    else:
        args = parser.parse_args()
        task, mydict = parse_func(args)

    train_2d(mydict)
    # train_2d_second(mydict)
