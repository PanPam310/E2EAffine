from __future__ import print_function, division
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.activation import PReLU
import torchvision.models as models


class AffineGridGen(nn.Module):
    def __init__(self, out_h=240, out_w=240, out_ch=3):
        super(AffineGridGen, self).__init__()
        self.out_h = out_h
        self.out_w = out_w
        self.out_ch = out_ch

    def forward(self, theta):
        theta = theta.contiguous()
        batch_size = theta.size()[0]
        out_size = torch.Size((batch_size, self.out_ch, self.out_h, self.out_w))
        return F.affine_grid(theta, out_size, align_corners=False)
def rotation_img(img, search_angle, device='cpu'):
    '''
    inputs:[N,C,H,W]
    outputs:[N,S,C,H,W] S:search.
    顺时针
    '''
    angle_area_number = int(360 / search_angle)
    rigid_matrix = torch.zeros((angle_area_number, 2, 3)).to(device).to(torch.float32)
    transformed_img = torch.zeros((img.size(0), angle_area_number, img.size(1), img.size(2), img.size(3))).to(
        device).to(torch.float32)
    for i in range(angle_area_number):
        rotation = i * search_angle * np.pi / 180
        rigid_matrix[i, ...] = torch.from_numpy(np.array([
            [np.cos(rotation), np.sin(rotation), 0],
            [-np.sin(rotation), np.cos(rotation), 0],
        ])).to(torch.float32)
    for i in range(img.size(0)):
        _img = torch.zeros((angle_area_number, img.size(1), img.size(2), img.size(3))).to(device).to(torch.float32)
        _img[:, ...] = img[i, ...]
        grid = F.affine_grid(rigid_matrix,_img.size(),align_corners=True).to(device)
        transformed_img[i, ...] = F.grid_sample(_img, grid, align_corners=True, padding_mode='border')
    return transformed_img, rigid_matrix




class FeatureExtraction(torch.nn.Module):
    def __init__(self, device='cuda:0', feature_extraction_cnn='resnet34', last_layer='',
                 pretrained=True):
        super(FeatureExtraction, self).__init__()
        self.device = device


        if feature_extraction_cnn == 'vgg':
            self.model = models.vgg16(pretrained=pretrained)
            # keep feature extraction network up to indicated layer
            vgg_feature_layers = ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1',
                                  'relu2_1', 'conv2_2', 'relu2_2', 'pool2', 'conv3_1', 'relu3_1',
                                  'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'pool3', 'conv4_1',
                                  'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'pool4',
                                  'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'pool5']
            if last_layer == '':
                last_layer = 'pool4'
            last_layer_idx = vgg_feature_layers.index(last_layer)
            self.model = nn.Sequential(*list(self.model.features.children())[:last_layer_idx + 1])

        if feature_extraction_cnn == 'resnet18':
            self.model = models.resnet18(pretrained=pretrained)
            resnet_feature_layers = ['conv1',
                                     'bn1',
                                     'relu',
                                     'maxpool',
                                     'layer1',
                                     'layer2',
                                     'layer3',
                                     'layer4']
            if last_layer == '':
                last_layer = 'layer3'
            last_layer_idx = resnet_feature_layers.index(last_layer)
            resnet_module_list = [self.model.conv1,
                                  self.model.bn1,
                                  self.model.relu,
                                  self.model.maxpool,
                                  self.model.layer1,
                                  self.model.layer2,
                                  self.model.layer3,
                                  self.model.layer4]

            self.model = nn.Sequential(*resnet_module_list[:last_layer_idx + 1])

        if feature_extraction_cnn == 'resnet34':
            self.model = models.resnet34(pretrained=pretrained)
            resnet_feature_layers = ['conv1',
                                     'bn1',
                                     'relu',
                                     'maxpool',
                                     'layer1',
                                     'layer2',
                                     'layer3',
                                     'layer4']
            if last_layer == '':
                last_layer = 'layer3'
            resnet_module_list = [self.model.conv1,
                                  self.model.bn1,
                                  self.model.relu,
                                  self.model.maxpool,
                                  self.model.layer1,
                                  self.model.layer2,
                                  self.model.layer3,
                                  self.model.layer4]
            self.model = nn.Sequential(*resnet_module_list[:7])


        if feature_extraction_cnn == 'resnet101':
            self.model = models.resnet101(pretrained=pretrained)
            resnet_feature_layers = ['conv1',
                                     'bn1',
                                     'relu',
                                     'maxpool',
                                     'layer1',
                                     'layer2',
                                     'layer3',
                                     'layer4']
            if last_layer == '':
                last_layer = 'layer3'
            last_layer_idx = resnet_feature_layers.index(last_layer)
            resnet_module_list = [self.model.conv1,
                                  self.model.bn1,
                                  self.model.relu,
                                  self.model.maxpool,
                                  self.model.layer1,
                                  self.model.layer2,
                                  self.model.layer3,
                                  self.model.layer4]

            self.model = nn.Sequential(*resnet_module_list[:last_layer_idx + 1])

        # freeze parameters
        for param in self.model.parameters():
            # save lots of memory
            # param.requires_grad = False
            param.requires_grad = True
        # move to GPU
        if True:
            self.model.to(self.device)

    def forward(self, image_batch):
        return self.model(image_batch)


class FeatureL2Norm(torch.nn.Module):
    def __init__(self):
        super(FeatureL2Norm, self).__init__()

    def forward(self, feature):
        epsilon = 1e-6
        norm = torch.pow(torch.sum(torch.pow(feature, 2), dim=1) + epsilon, 0.5).unsqueeze(1).expand_as(feature)
        return torch.div(feature, norm)

class FeatureCorrelation(torch.nn.Module):
    def __init__(self):
        super(FeatureCorrelation, self).__init__()

    def forward(self, feature_A, feature_B):
        b, c, h, w = feature_A.size()
        # reshape features for matrix multiplication
        feature_A = feature_A.transpose(2, 3).contiguous().view(b, c, h * w)
        feature_B = feature_B.view(b, c, h * w).transpose(1, 2)  # [b, h*w, c]
        # perform matrix mult.
        feature_mul = torch.bmm(feature_B, feature_A)  # batch matrix multiplication->(b, h*w, h*w)
        correlation_tensor = feature_mul.view(b, h, w, h * w).transpose(2, 3).transpose(1, 2)
        return correlation_tensor
class FeatureRegression(nn.Module):
    '''用于回归仿射变换参数.'''

    def __init__(self, output_dim=6, device='cuda:0'):
        super(FeatureRegression, self).__init__()
        self.device = device
        self.conv = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=7, padding=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # 回归仿射变换参数
        self.linear = nn.Linear(64 * 16 * 16, output_dim)

        if True:
            self.conv.to(self.device)
            self.linear.to(self.device)

    def forward(self, x):
        x = self.conv(x)
        x = x.contiguous().view(x.size(0), -1)
        theta = self.linear(x)
        return theta



class Affine_rot(nn.Module):
    def __init__(self, args, device='cuda:0', geometric_model='affine', normalize_features=True,
                 normalize_matches=True, batch_normalization=True):
        super(Affine_rot, self).__init__()
        self.feature_extraction_cnn = 'resnet34'
        self.angle = 90
        self.device = device
        self.alpha = 0.1
        self.SS = args.SS
        self.gridGen = AffineGridGen(args.patch_size, args.patch_size)

        self.normalize_features = normalize_features
        self.normalize_matches = normalize_matches
        self.FeatureExtraction = FeatureExtraction(device=self.device,
                                                   feature_extraction_cnn=self.feature_extraction_cnn, pretrained=False)
        self.adpPool = nn.AdaptiveAvgPool2d((16, 16))  # 为适应不同尺寸的图像，添加自适应池化层（ProsRegNet中没有）
        self.FeatureL2Norm = FeatureL2Norm()
        self.FeatureCorrelation = FeatureCorrelation()
        if geometric_model == 'affine':
            output_dim = 6
        elif geometric_model == 'tps':
            output_dim = 72
        self.FeatureRegression = FeatureRegression(output_dim, device=self.device)
        self.FeatureExtraction_ss = FeatureRegression(int(360 / self.angle), device=self.device)
        self.ReLU = nn.ReLU(inplace=True)


    def forward(self, source_img, target_img):
        """
        use self-supervised mode
        when inference， SS mode is suppressed
        """

        source_img, affine_matrix = rotation_img(source_img, self.angle, self.device)
        pred_fake = torch.zeros((source_img.size(0), source_img.size(1), int(360 / self.angle))).to(self.device).type(
            torch.cuda.FloatTensor)  #   [N,S,4]
        pred_true = torch.zeros((source_img.size(0), source_img.size(1), int(360 / self.angle))).to(self.device).type(
            torch.cuda.FloatTensor)  #   [N,S,4]

        # compute fake label
        if self.SS :
            for i in range(source_img.size(1)):
                # compute fake label
                org_feature_A = self.FeatureExtraction(source_img[:, 0, ...])
                ss_feature_A = self.FeatureExtraction(source_img[:, i, ...])
                org_feature_A = self.adpPool(org_feature_A)
                ss_feature_A = self.adpPool(ss_feature_A)  # normalize
                if self.normalize_features:
                    org_feature_A = self.FeatureL2Norm(org_feature_A)
                    ss_feature_A = self.FeatureL2Norm(ss_feature_A)

                correlation = self.FeatureCorrelation(org_feature_A, ss_feature_A)
                pred_fake[:, i, :] = self.FeatureExtraction_ss(correlation)

        # compute true label
        feature_A = self.FeatureExtraction(source_img[:, 0, ...])
        feature_B = self.FeatureExtraction(target_img)
        feature_A = self.adpPool(feature_A)
        feature_B = self.adpPool(feature_B)
        if self.normalize_features:
            feature_A = self.FeatureL2Norm(feature_A)
            feature_B = self.FeatureL2Norm(feature_B)
        correlation = self.FeatureCorrelation(feature_A, feature_B)
        pred_true = self.FeatureExtraction_ss(correlation)
        values_true, indices_true = torch.max(pred_true, dim=1)

        # compute theta according to minimum prediction loss of angle
        # feature extraction
        source_A = torch.zeros(target_img.size()).to(self.device).type(torch.cuda.FloatTensor)
        for i in range(len(indices_true)):
            source_A[i, ...] = source_img[i, indices_true[i], ...]

        true_feature_A = self.FeatureExtraction(source_A)
        true_feature_B = self.FeatureExtraction(target_img)
        true_feature_A = self.adpPool(true_feature_A)
        true_feature_B = self.adpPool(true_feature_B)
        if self.normalize_features:
            true_feature_A = self.FeatureL2Norm(true_feature_A)
            true_feature_B = self.FeatureL2Norm(true_feature_B)
        true_correlation = self.FeatureCorrelation(true_feature_A, true_feature_B)
        theta = self.FeatureRegression(true_correlation)

        if theta.shape[1] == 6:
            temp = torch.tensor([1.0, 0, 0, 0, 1.0, 0])
            adjust = temp.repeat(theta.shape[0], 1)
            adjust = adjust.to(self.device)
            theta = self.alpha * theta + adjust  # identity affine transformation
            theta = theta.reshape(theta.size()[0], 2, 3)
            theta = theta.to(self.device)

        sampling_grid = self.gridGen(theta)  # affine/tps [B,H,W,2] 形变场
        warped_source = F.grid_sample(source_A, sampling_grid, padding_mode='border', align_corners=False)
        return warped_source, theta, sampling_grid, pred_fake, pred_true, indices_true




def load_network(args):
    if args.mp:
        loc = 'cuda:{}'.format(args.local_rank)
    else:
        loc = args.device
    if args.modelname == 'Affine_rot':
        model = Affine_rot(args,device=loc)
    model.to(loc)
    if args.cpt != None:
        try:
            model.load_state_dict(torch.load(args.cpt, map_location=loc)['state_dict'])
        except:
            model.load_state_dict(torch.load(args.cpt, map_location=loc))
    if args.mp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], broadcast_buffers=False, find_unused_parameters=True)
    return model


if __name__ == '__main__':
    from Utils.args import Affine_args
    from torchsummary import summary
    args = Affine_args()
    args.SS = 0
    device = 'cuda:0'
    print(args)
    model = Affine_rot(args,device = device).to(device)
    summary(model,[(3,256,256),(3,256,256)])
    print(model)