import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.utils.data as data
from torchvision import transforms
import torch.nn.functional as F
import random



def get_pad(img, output_shape=[512, 512]):
    """[C,H,W]."""
    img = img.view(1, img.size(0), img.size(1), img.size(2))  # 3 channel
    img = F.interpolate(img, output_shape)
    img = img.view(img.size(1), img.size(2), img.size(3))
    return img

def seg_img(img, rec):
    x0, y0, w, h = rec
    img_seg = img[y0:y0 + h, x0:x0 + w]
    return img_seg


def normalize(img):
    for i in range(img.shape[2]):
        img[..., i] = (img[..., i] - np.min(img[..., i])) / (np.max(img[..., i]) - np.min(img[..., i]))
    return img
def tensor_affine_transform(tensor, tensor_transform, device, sample_mode='border'):
    affine_grid = F.affine_grid(tensor_transform,
                                tensor.size(), align_corners=True)  # affine_grid [N,H,W,2]
    affine_grid = affine_grid.to(device)
    transformed_tensor = F.grid_sample(tensor, affine_grid, padding_mode=sample_mode, align_corners=True)  # 重采样 /zeros/border/reflection
    return transformed_tensor
def rotation_single(img, angle, device='cpu'):
    '''[C, H, W]'''
    rotation = angle * np.pi / 180
    rigid_matrix = torch.from_numpy(np.array([
        [np.cos(rotation), np.sin(rotation), 0],
        [-np.sin(rotation), np.cos(rotation), 0],
    ])).to(torch.float32)
    transformed_source = tensor_affine_transform(img.view(1, img.size(0), img.size(1), img.size(2)),
                                                    rigid_matrix.unsqueeze(0), device)[0,...]
    return transformed_source

class ACROBATDataset(data.Dataset):
    def __init__(self, args, phase, transform=None, transform2=None, level=5, sobel=False):
        self.abs_dir = args.abs_dir
        self.transform = transform
        self.transform2 = transform2
        self.channel = args.channel
        self.patch_size = args.patch_size
        if phase == 'train':
            self.Pair_dic = torch.load(args.Pair_dic_path)
        elif phase == 'validation':
            self.Pair_dic = torch.load(args.Pair_dic_path_eval)
        elif phase == 'test':
            self.Pair_dic = torch.load(args.Pair_dic_path_eval)
        self.TrainID = args.TrainID
        self.ValidID = args.ValidID
        self.TestID = args.TestID
        self.device = args.device

        self.pad = args.pad
        self.seg = args.seg
        self.level = level
        self.phase = phase
        self.batch_size_eval = args.batch_size_eval



        if phase == 'train':
            # self.Indicator_ID = self.TrainID
            self.Indicator_ID = list(self.Pair_dic.keys())
        elif phase == 'test':
            self.Indicator_ID = list(self.Pair_dic.keys())
        elif phase == 'validation':
            self.Indicator_ID = list(self.Pair_dic.keys())

    def __len__(self):
        return len(self.Indicator_ID)

    def __getitem__(self, index):
        indicator_id = self.Indicator_ID[index]
        if self.phase == 'train':
            source_path = self.Pair_dic[indicator_id]['source']
            target_path = self.Pair_dic[indicator_id]['target']
        if self.phase == 'validation':
            source_path = self.Pair_dic[indicator_id]['source']
            target_path = self.Pair_dic[indicator_id]['target']
        # transfer landmarks:
        if self.batch_size_eval == 1:
            Landmark = True
        else:
            Landmark = False
        t_lamk = []
        if self.phase == 'test' or self.phase == 'validation':
            if Landmark:
                t_lamk = np.array(self.Pair_dic[indicator_id]['ihc_xs_ys'])
                mmp_t = self.Pair_dic[indicator_id]['mpp_ihc_10X']
                t_lamk[:, 0] = t_lamk[:, 0] / (mmp_t * (2 ** (self.level)))
                t_lamk[:, 1] = t_lamk[:, 1] / (mmp_t * (2 ** (self.level)))

                rect = self.Pair_dic[indicator_id]['target_x0_y0_w_h']
                padding = self.Pair_dic[indicator_id]['target_seg_pad_h_top_t_pad_h_bottom_t_pad_w_left_t_pad_w_right_t']
                x0_t, y0_t, w_t, h_t = rect
                pad_h_top_t, pad_h_bottom_t, pad_w_left_t, pad_w_right_t = padding
                t_lamk[:, 0] = t_lamk[:, 0] - x0_t + pad_w_left_t
                t_lamk[:, 1] = t_lamk[:, 1] - y0_t + pad_h_top_t

        if self.channel == 3:
            source = cv2.imread(source_path)
            target = cv2.imread(target_path)
        # 输入用单灰度图 [H, W, C]， C=3训练使用三通道灰度图 [H, W, C]， C=1
        if self.channel == 1:
            source = cv2.imread(source_path, 0)
            target = cv2.imread(target_path, 0)
            source = np.stack((source, source, source), axis=2)
            target = np.stack((target, target, target), axis=2)

        ori_s_size = source.shape[:2]
        ori_t_size = target.shape[:2]

        seg_source_info = self.Pair_dic[indicator_id]['source_x0_y0_w_h']
        seg_target_info = self.Pair_dic[indicator_id]['target_x0_y0_w_h']
        spad_h_top, spad_h_bottom, spad_w_left, spad_w_right = self.Pair_dic[indicator_id][
            'source_seg_pad_h_top_s_pad_h_bottom_s_pad_w_left_s_pad_w_right_s']
        tpad_h_top, tpad_h_bottom, tpad_w_left, tpad_w_right = self.Pair_dic[indicator_id][
            'target_seg_pad_h_top_t_pad_h_bottom_t_pad_w_left_t_pad_w_right_t']
        source_color = self.Pair_dic[indicator_id]['source_color']
        target_color = self.Pair_dic[indicator_id]['target_color']
        if self.seg:
            source = seg_img(source, seg_source_info)
            target = seg_img(target, seg_target_info)
        if self.pad == 'border':
            source = cv2.copyMakeBorder(source, spad_h_top, spad_h_bottom, spad_w_left, spad_w_right,
                                        cv2.BORDER_REPLICATE)
            target = cv2.copyMakeBorder(target, tpad_h_top, tpad_h_bottom, tpad_w_left, tpad_w_right,
                                        cv2.BORDER_REPLICATE)
        if self.pad == 'BG':
            source = cv2.copyMakeBorder(source, spad_h_top, spad_h_bottom, spad_w_left, spad_w_right,
                                        cv2.BORDER_CONSTANT,
                                        value=source_color)
            target = cv2.copyMakeBorder(target, tpad_h_top, tpad_h_bottom, tpad_w_left, tpad_w_right,
                                        cv2.BORDER_CONSTANT,
                                        value=target_color)

        sp_s_size = source.shape[:2]
        sp_t_size = target.shape[:2]

        assert sp_s_size == sp_t_size

        if self.phase == 'test' or self.phase == 'validation':
            if Landmark:
                t_lamk[:, 0] = self.patch_size * t_lamk[:, 0] / target.shape[1]
                t_lamk[:, 1] = self.patch_size * t_lamk[:, 1] / target.shape[0]

        source = torch.from_numpy(source).permute(2, 0, 1)
        target = torch.from_numpy(target).permute(2, 0, 1)

        # 调整大小
        if source.shape[1] != self.patch_size:
            source = get_pad(source, [self.patch_size, self.patch_size]).permute(1, 2, 0).detach().cpu().numpy()
            target = get_pad(target, [self.patch_size, self.patch_size]).permute(1, 2, 0).detach().cpu().numpy()

        if self.transform:
            source, target, _ = self.transform(source, target)  # 数据增强返回的target为numpy格式
            source = source.astype(np.uint8)
            target = target.astype(np.uint8)
        if self.transform2:
            source = self.transform2(source)
            target = self.transform2(target)
        else:
            transform = transforms.Compose([transforms.ToPILImage(),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
            source = transform(source)
            target = transform(target)
        if self.phase == 'train' or self.phase == 'validation':
            label = self.Pair_dic[indicator_id]['label']
        else:
            label = None
        # random rotaion
        if self.phase == 'train':
            random_label = random.randint(0,4)
            label = (label - random_label + 4 ) % 4
            source = rotation_single(source,random_label*90)
        return {'source': source, 'target': target, 'id': indicator_id, 'true_label':label,'t_lamk': t_lamk, \
                's_size': {'ori': ori_s_size, 'seg_pad': sp_s_size},
                't_size': {'ori': ori_t_size, 'seg_pad': sp_t_size}}






def load_dataloader(args, phase, transform=None, transform2=None, sobel=False):
    if phase == 'train':
        if args.mp:
            Train_dataset = ACROBATDataset(args, phase, transform, transform2)
            Train_sampler = torch.utils.data.distributed.DistributedSampler(Train_dataset)
            Train_dataloader = data.DataLoader(Train_dataset, batch_size=args.batch_size,
                                               shuffle=(Train_sampler is None), num_workers=args.num_workers,
                                               pin_memory=True,
                                               sampler=Train_sampler, drop_last=True)
            return Train_dataset, Train_dataloader, Train_sampler
        else:
            Train_dataset = ACROBATDataset(args, phase, transform, transform2)
            Train_dataloader = data.DataLoader(Train_dataset, batch_size=args.batch_size,
                                               shuffle=True, num_workers=args.num_workers, drop_last=True)
            return Train_dataset, Train_dataloader

    elif phase == 'validation':

        Val_dataset = ACROBATDataset(args, phase, None, None)
        Val_dataloader = data.DataLoader(Val_dataset, batch_size=args.batch_size_eval,
                                         shuffle=False, num_workers=args.num_workers)
        return Val_dataset, Val_dataloader
    elif phase == 'test':
        Test_dataset = ACROBATDataset(args, phase, None, transform2)
        Test_dataloader = data.DataLoader(Test_dataset, batch_size=args.batch_size_eval,
                                          shuffle=False, num_workers=args.num_workers)
        return Test_dataset, Test_dataloader
