import os

import torch
import time
from tqdm import tqdm
from Utils.basic_utils import FixedNum



def Affine_rotation_train(args, model, loader, criterion, criterion2, optimizer,scheduler):
    # Train
    model.train()
    if args.mp:
        loc = 'cuda:{}'.format(args.local_rank)
    else:
        loc = args.device
    train_loss = 0
    NCC_losses = 0
    train_true_CE = 0
    train_fake_CE = 0

    num = 0
    save_num = 20
    targets_visi, sources_visi, transform_sources_visi = [], [], []



    with tqdm(total=len(loader)) as pbar:
        right_tensor = torch.zeros(len(loader.dataset)).cuda(args.local_rank, non_blocking=True)  # 用于存放真实标签true label
        indices_tensor = torch.zeros(len(loader.dataset)).cuda(args.local_rank,
                                                               non_blocking=True)  # 用于分类得到的预测标签，并用于指导配准theta的选择
        log_matrix = torch.zeros((4, 4)).cuda(args.local_rank,non_blocking=True).type(torch.cuda.FloatTensor)
        for i, inputs in enumerate(loader):
            if args.mp:
                sources = inputs['source'].cuda(args.local_rank, non_blocking=True).type(
                    torch.cuda.FloatTensor) + torch.rand(inputs['source'].size()).cuda(args.local_rank,
                                                                                       non_blocking=True) * 0.001
                targets = inputs['target'].cuda(args.local_rank, non_blocking=True).type(
                    torch.cuda.FloatTensor) + torch.rand(inputs['source'].size()).cuda(args.local_rank,
                                                                                       non_blocking=True) * 0.001
            else:
                sources = inputs['source'].to(loc).type(torch.cuda.FloatTensor)
                targets = inputs['target'].to(loc).type(torch.cuda.FloatTensor)

            img_id = inputs['id']
            true_label = inputs['true_label'].to(loc)


            CrossEntropy_fake = torch.zeros((4)).to(loc).type(torch.cuda.FloatTensor)

            transformed_sources, theta, transformed_grid,\
            pred_fake, preds_true, indices_true = model(sources, targets)

            '''compute loss and log matrix'''
            # NCC loss
            NCC_loss = torch.mean(criterion(transformed_sources, targets))
            # fake loss
            for j in range(4):
                fake_label = torch.zeros((sources.size(0), 4), dtype=torch.long).to(loc)
                fake_label[:, j] = j
                CrossEntropy_fake[j] = criterion2(pred_fake[:, j, :], fake_label[:, j])
            CE_fake_loss = torch.mean(CrossEntropy_fake)
            # true loss
            CE_true_loss = torch.mean(criterion2(preds_true, true_label))  # 弱监督任务 true_label分类任务损失

            # log matrix
            for j in range(sources.size(0)):
                log_matrix[true_label[j], indices_true[j]] += 1

            right_tensor[i * args.batch_size:(i + 1) * args.batch_size] = true_label  # 按照采样顺续的true label，用于计算旋转角预测错误的数量
            indices_tensor[i * args.batch_size:(i + 1) * args.batch_size] = indices_true

            if args.local_rank == 0:
                print(
                    f'NCC is:{FixedNum(NCC_loss.item())}, CE_True is:{FixedNum(CE_true_loss.item())}, CE_Fake is:{FixedNum(CE_fake_loss.item())}.')
                print('theta:', theta[0:5, ...])

            # total loss
            if not args.SS:
                total_loss = CE_true_loss +  NCC_loss
            else:
                total_loss = CE_fake_loss + CE_true_loss + NCC_loss


            train_loss += total_loss.item()
            NCC_losses += NCC_loss.item()
            train_true_CE += CE_true_loss.item()
            train_fake_CE += CE_fake_loss.item()

            if num <= save_num:
                sources_visi.append(sources)
                targets_visi.append(targets)
                transform_sources_visi.append(transformed_sources)
                num += sources.shape[0]

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            pbar.update(1)
    if not scheduler == None:
        scheduler.step()

    train_loss = train_loss / len(loader)
    NCC_losses = NCC_losses / len(loader)
    train_true_CE = train_true_CE / len(loader)
    train_fake_CE = train_fake_CE / len(loader)
    wrong_num = len(right_tensor) - int(torch.sum((indices_tensor == right_tensor)))

    if args.local_rank == 0:
        print('*' * 10, 'train', '*' * 10, '\n')
        print(f'NCC is:{NCC_losses}, CE_True is:{train_true_CE}, CE_Fake is:{train_fake_CE}.')
    Losses = [train_loss,NCC_losses,train_true_CE,train_fake_CE]
    sources_visi = torch.cat(sources_visi,dim=0)
    targets_visi = torch.cat(targets_visi,dim=0)
    transform_sources_visi = torch.cat(transform_sources_visi,dim=0)

    return Losses, wrong_num, sources_visi, targets_visi, transform_sources_visi,optimizer


