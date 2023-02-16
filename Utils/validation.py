import os

import torch
import time
from tqdm import tqdm
from Utils.basic_utils import FixedNum



def Affine_rotation_validation(args, model, loader, criterion, criterion2):
    
    model.eval()
    loc = f'cuda:{args.local_rank}'
    val_loss = 0
    NCC_losses = 0
    val_true_CE = 0


    num = 0
    save_num = 20
    targets_visi, sources_visi, transform_sources_visi = [], [], []

    with torch.no_grad():
        with tqdm(total=len(loader)) as pbar:
            right_tensor = torch.zeros(len(loader.dataset)).to(loc) # 用于存放真实标签true label
            indices_tensor = torch.zeros(len(loader.dataset)).to(loc)  # 用于分类得到的预测标签，并用于指导配准theta的选择
            log_matrix = torch.zeros((4, 4)).to(loc).type(torch.cuda.FloatTensor)
            for i, inputs in enumerate(loader):
                sources = inputs['source'].to(loc).type(torch.cuda.FloatTensor)
                targets = inputs['target'].to(loc).type(torch.cuda.FloatTensor)

                img_id = inputs['id']
                true_label = inputs['true_label'].to(loc)

                transformed_sources, theta, transformed_grid,_,\
                preds_true, indices_true = model(sources, targets)

                '''compute loss and log matrix'''
                # NCC loss
                NCC_loss = torch.mean(criterion(transformed_sources, targets))

                # true loss
                CE_true_loss = torch.mean(criterion2(preds_true, true_label))  # 弱监督任务 true_label分类任务损失

                # log matrix
                for j in range(sources.size(0)):
                    log_matrix[true_label[j], indices_true[j]] += 1

                right_tensor[i * args.batch_size_eval:(i + 1) * args.batch_size_eval] = true_label  # 按照采样顺续的true label，用于计算旋转角预测错误的数量
                indices_tensor[i * args.batch_size_eval:(i + 1) * args.batch_size_eval] = indices_true

                if args.local_rank == 0:
                    print(
                        f'NCC is:{FixedNum(NCC_loss.item())}, CE_True is:{FixedNum(CE_true_loss.item())}, CE_Fake is:{FixedNum(CE_true_loss.item())}.')
                    print('theta:', theta[0:5, ...])


                val_loss = val_loss + NCC_loss.item() + CE_true_loss.item()
                NCC_losses += NCC_loss.item()
                val_true_CE += CE_true_loss.item()


                if num <= save_num:
                    sources_visi.append(sources)
                    targets_visi.append(targets)
                    transform_sources_visi.append(transformed_sources)
                    num += sources.shape[0]

                pbar.update(1)


        val_loss = val_loss / len(loader)
        NCC_losses = NCC_losses / len(loader)
        val_true_CE = val_true_CE / len(loader)
        wrong_num = len(right_tensor) - int(torch.sum((indices_tensor == right_tensor)))


        print('*' * 10, 'Validation', '*' * 10, '\n')
        print(f'NCC is:{NCC_losses}, CE_True is:{val_true_CE}')
        Losses = [FixedNum(val_loss),FixedNum(NCC_losses),FixedNum(val_true_CE)]
        sources_visi = torch.cat(sources_visi,dim=0)
        targets_visi = torch.cat(targets_visi,dim=0)
        transform_sources_visi = torch.cat(transform_sources_visi,dim=0)
    return Losses, wrong_num, sources_visi, targets_visi, transform_sources_visi


