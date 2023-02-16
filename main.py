import torch.distributed as dist
from tqdm import tqdm
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
import time

from Utils.args import Affine_args
from Utils.criterion import  load_criterion
from Utils.basic_utils import __gamma_adjust
from Utils.train import Affine_rotation_train
from Utils.validation import Affine_rotation_validation

from Utils.dataset import load_dataloader
from model import load_network

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


# parser
args = Affine_args()
a = 0.002


def main(args):
    print(args)
    Flag = f'{args.exp}_{args.t1}_{args.t2}_{args.batch_size}_{args.lr}'
    log_dir = f'log_file/{Flag}/'
    if not os.path.exists(f'checkpoints/{Flag}'):
        os.mkdir(f'checkpoints/{Flag}')
    # save config file
    torch.save(args,f'checkpoints/{Flag}/args.dic')
    tensorboard_path = log_dir
    Writer = SummaryWriter(tensorboard_path)
    save_step = 20


    # model, optimizer and loss
    model = load_network(args)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)
    cudnn.benchmark = True
    criterion = load_criterion('NCC')
    criterion2 = nn.CrossEntropyLoss()


    # Load datasets and transform
    transform2 = transforms.Compose([transforms.Lambda(lambda img: __gamma_adjust(img)),
                                     transforms.ToPILImage(),
                                     transforms.RandomChoice([transforms.ColorJitter(brightness=0.5),
                                                              transforms.ColorJitter(contrast=0.5),
                                                              transforms.ColorJitter(saturation=0.5),
                                                              transforms.ColorJitter(hue=0.5)]),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])


    if args.t1 == 0:
        transform = None
    if args.t2 == 0:
        transform2 = None
    if args.mp:
        Train_dataset, Train_dataloader,Train_sampler = load_dataloader(args, 'train', transform, transform2)
    else:
        Train_dataset, Train_dataloader = load_dataloader(args, 'train', transform, transform2)

    Val_dataset, Val_dataloader = load_dataloader(args, 'validation', transform, transform2)

    # loop
    best_cri = 1
    train_loss = 1
    val_loss = 1

    for epoch in range(args.purge_step, args.epoch):
        try:
            Train_sampler.set_epoch(epoch) # shuffle 结果
        except:
            pass
        args.SS = 1
        train_losses , train_wrong_num, sources_visi, targets_visi, transform_sources_visi,optimizer = Affine_rotation_train(args,model,Train_dataloader,criterion,criterion2,optimizer,scheduler)

        args.SS = 0
        val_losses, val_wrong_num, val_sources_visi, val_targets_visi, val_transform_sources_visi = Affine_rotation_validation(args,model,Val_dataloader,criterion,criterion2)

        for param_group in optimizer.param_groups:
            lr_latest=param_group['lr']

        train_loss,train_ncc_loss,train_CE_true_loss, train_CE_fake_loss = train_losses
        val_loss, val_ncc_loss, val_CE_true_loss = val_losses




        if args.mp == 0 or (args.mp == 1 and args.local_rank == 0 ):
            """ Write in tensorboard"""
            Writer.add_scalar('scalar/train_loss', train_loss, epoch )
            Writer.add_scalar('scalar/train_ncc_loss', train_ncc_loss, epoch)
            Writer.add_scalar('scalar/train_CE_true', train_CE_true_loss, epoch)
            Writer.add_scalar('scalar/train_CE_fake', train_CE_fake_loss, epoch)
            Writer.add_scalar('scalar/train_wrong_num', train_wrong_num,epoch)
            Writer.add_scalar('scalar/val_loss', val_loss, epoch )
            Writer.add_scalar('scalar/val_ncc_loss', val_ncc_loss, epoch)
            Writer.add_scalar('scalar/val_CE', val_CE_true_loss, epoch)
            Writer.add_scalar('scalar/val_wrong_num', val_wrong_num, epoch)


            Writer.add_scalar('scalar/lr', lr_latest, epoch)
            # Writer.add_scalar('scalar/val_loss', val_loss.item(), epoch )
            print("\r[Exp:{}][Train][Epoch {}/{}][lr{}][train_loss:{:.8f}][learning rate:{}]".format(args.exp,
                                                                                                epoch + 1,
                                                                                                args.epoch,
                                                                                                lr_latest,
                                                                                                train_loss,
                                                                                                args.lr))
            if epoch % save_step == 0 or epoch == 0:

                Writer.add_image('image/sources', make_grid(sources_visi[:10, ...]), global_step = epoch)
                Writer.add_image('image/targets', make_grid(targets_visi[:10, ...]), global_step = epoch)
                Writer.add_image('image/transformed sources', make_grid(transform_sources_visi[:10, ...]), global_step = epoch)

                Writer.add_image('image/val_sources', make_grid(val_sources_visi[:10, ...]), global_step = epoch)
                Writer.add_image('image/val_targets', make_grid(val_targets_visi[:10, ...]), global_step = epoch)
                Writer.add_image('image/val_transformed sources', make_grid(val_transform_sources_visi[:10, ...]), global_step = epoch)





        cri = train_loss
        '''save checkpoints'''
        if args.local_rank == 0 and args.mp:

            if  epoch != args.epoch - 1 and cri <= best_cri:
                best_cri = cri
                path = f'checkpoints/{Flag}/{epoch}_{train_loss}_{val_loss}_{best_cri}_{lr_latest}.pth'
                torch.save(model.module.state_dict(), path)
            if epoch == args.epoch - 1:
                path = f'checkpoints/{Flag}/{epoch}_{train_loss}_{val_loss}_{cri}_{lr_latest}.pth'
                torch.save(model.module.state_dict(),path)
        elif args.mp == 0:
            if  epoch != args.epoch - 1 and cri <= best_cri:
                best_cri = cri
                path = f'checkpoints/{Flag}/{epoch}_{train_loss}_{val_loss}_{best_cri}_{lr_latest}.pth'
                torch.save(model.state_dict(), path)
            if epoch == args.epoch - 1:
                path = f'checkpoints/{Flag}/{epoch}_{train_loss}_{val_loss}_{cri}_{lr_latest}.pth'
                torch.save(model.state_dict(),path)




if __name__ == '__main__':
    # args = parser.parse_args()
    # initial

    args.t1 = 0
    args.t2 = 0
    args.save_criterion = 'NCC'
    args.seg = 1
    args.pad = 'BG'
    args.Pair_dic_path =  ' '
    args.Pair_dic_path_eval = ' '


    args.modelphase = 'Affine'
    args.modelname = 'Affine_rot'

    args.cpt = ' '
    args.exp = ' '
    args.epoch = 1000
    args.device = 'cuda:0'
    phase = 'train'
    args.SS = 1

    if args.mp:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)

    main(args)


