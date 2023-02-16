import os
import cv2
import math
import time
import shutil
import numpy as np
import random
import scipy.ndimage as nd
from skimage.exposure import exposure
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

def __gamma_adjust(img, phase='train'):
    '''
    gamma变换
    :param img:
    :return:
    '''
    if phase == 'train':
        # return exposure.adjust_gamma(img, random.uniform(0.5,1.5))
        return exposure.adjust_gamma(img, 0.3 + random.random() * 4.5)
    if phase == 'test':
        return exposure.adjust_gamma(img, 3)

def FixedNum(number):
    return round(number, 6)

def affine_augmentation(affine_generation_params, augment_both=True):
    def augmentation(source, target):

        transform = generate_random_affine_transform(source.shape, **affine_generation_params)
        if augment_both:
            if random.random() > 0.5:
                transformed_source = numpy_affine_transform(source, transform)
                transformed_target = target
            else:
                transformed_source = source
                transformed_target = numpy_affine_transform(target, transform)
                transform = numpy_inv_transform(transform)
        else:
            transformed_source = numpy_affine_transform(source, transform)
            transformed_target = target
        transformed_source = (transformed_source - np.min(transformed_source)) / (
                    np.max(transformed_source) - np.min(transformed_source))
        transformed_target = (transformed_target - np.min(transformed_target)) / (
                    np.max(transformed_target) - np.min(transformed_target))

        return transformed_source, transformed_target, numpy_inv_transform(transform)

    return augmentation


def generate_random_affine_transform(shape, **params):
    min_translation = params['min_translation']
    max_translation = params['max_translation']
    min_rotation = params['min_rotation']
    max_rotation = params['max_rotation']
    min_shear = params['min_shear']
    max_shear = params['max_shear']
    min_scale = params['min_scale']
    max_scale = params['max_scale']

    min_rotation = min_rotation * np.pi / 180
    max_rotation = max_rotation * np.pi / 180
    min_translation = min_translation * min(shape[0:2])
    max_translation = max_translation * max(shape[0:2])

    x_translation = random.uniform(min_translation, max_translation)
    y_translation = random.uniform(min_translation, max_translation)
    rotation = random.uniform(min_rotation, max_rotation)

    x_shear = random.uniform(min_shear, max_shear)
    y_shear = random.uniform(min_shear, max_shear)

    x_scale = random.uniform(min_scale, max_scale)
    y_scale = random.uniform(min_scale, max_scale)

    rigid_matrix = np.array([
        [np.cos(rotation), -np.sin(rotation), x_translation],
        [np.sin(rotation), np.cos(rotation), y_translation],
        [0, 0, 1],
    ])
    cm1 = np.array([
        [1, 0, ((shape[0] - 1) / 2)],
        [0, 1, ((shape[1] - 1) / 2)],
        [0, 0, 1],
    ])
    cm2 = np.array([
        [1, 0, -((shape[0] - 1) / 2)],
        [0, 1, -((shape[1] - 1) / 2)],
        [0, 0, 1],
    ])
    rigid_matrix = cm1 @ rigid_matrix @ cm2

    shear_matrix = np.array([
        [1, x_shear, 0],
        [y_shear, 1, 0],
        [0, 0, 1],
    ])
    scale_matrix = np.array([
        [x_scale, 0, 0],
        [0, y_scale, 0],
        [0, 0, 1],
    ])

    all_matrices = [rigid_matrix, shear_matrix, scale_matrix]
    random.shuffle(all_matrices)
    transform = np.eye(3)
    for i in range(len(all_matrices)):
        transform = transform @ all_matrices[i]
    final_transform = transform[0:2, :]
    return final_transform

def generate_rotation_matrix(angle, x0, y0):
    """theta(图像坐标系):逆时针旋转."""
    angle = angle * np.pi / 180
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    cm1 = np.array([
        [1, 0, x0],
        [0, 1, y0],
        [0, 0, 1]
    ])
    cm2 = np.array([
        [1, 0, -x0],
        [0, 1, -y0],
        [0, 0, 1]
    ])
    transform = cm1 @ rotation_matrix @ cm2
    return transform[0:2, :]

def affine2theta(affine, shape):
    """坐标变换 affine：起始坐标为图片左上角顶点，theta：起始坐标为图片对角线交点（中心点）"""
    h, w = shape[0], shape[1]
    temp = affine
    theta = torch.zeros([2, 3])
    theta[0, 0] = temp[0, 0]
    theta[0, 1] = temp[0, 1] * h / w
    theta[0, 2] = temp[0, 2] * 2 / w + theta[0, 0] + theta[0, 1] - 1
    theta[1, 0] = temp[1, 0] * w / h
    theta[1, 1] = temp[1, 1]
    theta[1, 2] = temp[1, 2] * 2 / h + theta[1, 0] + theta[1, 1] - 1
    return theta

def tensor_affine_transform(tensor, tensor_transform, device, sample_mode='border'):
    """
    根据tensor_transform,将tensor->warped.其中tensor_transform为图像坐标系中的theta类型.
    tensor:[B,C,H,W]
    tensor_transform:[B,2,3]
    """
    affine_grid = F.affine_grid(tensor_transform,
                                tensor.size(), align_corners=True)  # affine_grid [N,H,W,2]
    affine_grid = affine_grid.to(device)
    transformed_tensor = F.grid_sample(tensor, affine_grid, padding_mode=sample_mode, align_corners=True)  # 重采样 /zeros/border/reflection
    return transformed_tensor

def compose_transforms(t1, t2, device="cpu"):
    """将单个图像的仿射变换参数进行叠加,t1,t2均为theta类型."""
    tr1 = torch.zeros((3, 3)).to(device)
    tr2 = torch.zeros((3, 3)).to(device)
    tr1[0:2, :] = t1
    tr2[0:2, :] = t2
    tr1[2, 2] = 1
    tr2[2, 2] = 1
    result = torch.mm(tr1, tr2)
    return result[0:2, :]


def generate_rotation_matrix(angle, x0, y0):
    """theta(图像坐标系):逆时针旋转."""
    angle = angle * np.pi / 180
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    cm1 = np.array([
        [1, 0, x0],
        [0, 1, y0],
        [0, 0, 1]
    ])
    cm2 = np.array([
        [1, 0, -x0],
        [0, 1, -y0],
        [0, 0, 1]
    ])
    transform = cm1 @ rotation_matrix @ cm2
    return transform[0:2, :]

def warp_images( img, trans):
    sampling_grid = trans
    if len(img.shape) == 3:
        img = img.unsqueeze(0)
    warped_img = F.grid_sample(img, sampling_grid, padding_mode='border', align_corners=True)
    return warped_img


def transform_landmarks(landmarks, transformation_field, device='cpu'):
    """
    根据致密变形场(transformation_field)将target的landmarks转换至对应的source图像中.
    transformation_field:[2,H,W]
    """
    # displacement_field = transformation_to_displacement_field(transformation_field, device)[0]  # [1,2,H,W]
    # print(displacement_field.shape)
    displacement_field = transformation_field
    u_x = displacement_field[0, :, :].detach().cpu().numpy()
    u_y = displacement_field[1, :, :].detach().cpu().numpy()
    landmarks_x = landmarks[:, 0]
    landmarks_y = landmarks[:, 1]
    ux = nd.map_coordinates(u_x.copy(), [landmarks_y, landmarks_x])  # 根据原坐标在变形场上查找x轴，y轴查找对应的偏移量（dx，dy）
    uy = nd.map_coordinates(u_y.copy(), [landmarks_y, landmarks_x])
    ux = (ux / 2 + 0.5) * (displacement_field.shape[2] - 1)  # 将坐标由像素坐标系转为图像坐标系
    uy = (uy / 2 + 0.5) * (displacement_field.shape[1] - 1)
    new_landmarks = np.stack((ux, uy), axis=1)
    return new_landmarks

def warp_landmarks(t_lmark, pp_theta,affine_trans, loc):

    if affine_trans is not None:
        if len(affine_trans.shape) == 4:
            affine_trans = affine_trans.squeeze(0)
        t_lmark = transform_landmarks(t_lmark, affine_trans.permute(2, 0, 1), loc)

    if pp_theta is not None:
        # pp_sampling_grid = gridGen(pp_theta)[0,...].permute(2, 0, 1)
        if len(pp_theta.shape) == 4:
            pp_theta = pp_theta.squeeze(0)
        t_lmark = transform_landmarks(t_lmark, pp_theta.permute(2, 0, 1), loc)
    return t_lmark

def warp_landmarks_2(lamk,imgsize,s_size,t_size,device,\
                     rect_t=None,padding_t=None,rect_s=None,padding_s=None,\
                        pp_theta=None,affine_trans=None):
    ori_s_size,ori_t_size = s_size['ori'], t_size['ori']
    sp_s_size, sp_t_size = s_size['seg_pad'], t_size['seg_pad']
    # 分割
    if rect_t and rect_s and padding_t and padding_s:
        x0_t, y0_t, w_t, h_t = rect_t
        pad_h_top_t, pad_h_bottom_t, pad_w_left_t, pad_w_right_t = padding_t
        x0_s, y0_s, w_s, h_s = rect_s
        pad_h_top_s, pad_h_bottom_s, pad_w_left_s, pad_w_right_s = padding_s

        lamk[:,0] = imgsize*(lamk[:,0] - x0_t + pad_w_left_t)/sp_t_size[1]
        lamk[:,1] = imgsize*(lamk[:,1] - y0_t + pad_h_top_t)/sp_t_size[0]


    lamk = warp_landmarks(lamk,pp_theta,affine_trans,device)
    lamk[:,0] = sp_s_size[1]*lamk[:,0]/imgsize
    lamk[:,1] = sp_s_size[0]*lamk[:,1]/imgsize

    if rect_t and rect_s and padding_t and padding_s:
        lamk[:,0] = lamk[:,0] + x0_s - pad_w_left_s
        lamk[:,1] = lamk[:,1] + y0_s - pad_h_top_s
    return lamk



