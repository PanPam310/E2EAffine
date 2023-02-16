import argparse

def Affine_args():
    parser = argparse.ArgumentParser('Affine')
    # multiprocess
    parser.add_argument('--local_rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--mp', default=0, type=int,
                        help='whether multiprocessing')
    parser.add_argument('--exp', type=str, default='0',
                        help = 'experiments name')

    # Data
    parser.add_argument('--Pair_dic_path', default=None,
                        help='Dic file of image pairs')
    parser.add_argument('--abs_dir',default='/data_sdd/lyh/ACROBAT', help = 'absolute path of data')
    parser.add_argument('--TrainID', type=str, default=None,
                        help='train dataset ids')
    parser.add_argument('--ValidID', type=str, default=None,
                        help='valid dataset ids')
    parser.add_argument('--TestID', type=str, default=None,
                        help='test dataset ids')
    parser.add_argument('--patch_size', type=int, default=256)

    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size for dataloader(default=32)')
    parser.add_argument('--batch_size_eval', type=int, default=32,
                        help='batch size for dataloader(default=32)')

    parser.add_argument('--num_workers', type=int, default=0,
                        help='number workers for dataloader(default=8)')
    parser.add_argument('--channel', type=int, default=3,
                        help='image channels(3/1)')
    parser.add_argument('--DatasetLandmark_path', type=str, default=None,
                        help='landmark path')
    parser.add_argument('--rtre', type=int, default=1,
                        help='compute rtre ')
    parser.add_argument('--pad', type=str, default='border',
                        help='padding mode : border/zeros/background ')
    parser.add_argument('--seg', type=int, default=1,
                        help='segment images')


    # model
    parser.add_argument('--modelname', type=str, default='Affine_rot',
                        help='model name for train(default=Unet/)')
    parser.add_argument('--SS', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='device for load model and data(default=cuda:0)')
    parser.add_argument('--output_dim', type=int, default=6,
                        help='dimensions of output(default=6)')
    parser.add_argument('--normalization', type=str, default='BatchNorm',
                        help='Normalization of mini-batch:BatchNorm/GroupNorm')


    # loss
    parser.add_argument('--criterion', type=str, default='NCC',
                        help='loss function for weight update.(default=NCC/NGF/)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='weight for curvature_regularization loss')
    parser.add_argument('--alpha', type=float, default=1.0,
                            help='weight for curv loss')
    parser.add_argument('--beta', type=float, default=0.01,
                            help='weight for bend loss')
    # log
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--purge_step', type=int, default=0)
    parser.add_argument('--cpt', type=str, default=None,help='path for checkpoints')


    # train
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--t1', type=int, default=0,
                        help='whether use transform, 1:True, 0:False')
    parser.add_argument('--t2', type=int, default=0,
                        help='whether use transform2, 1:True, 0:False')
    args = parser.parse_args()
    return args


