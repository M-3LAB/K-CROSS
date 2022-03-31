import argparse
from random import choices

__all__ = ['parse_arguments_kaid']


def parse_arguments_kaid():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', '-d', type=str, default='brats2021', choices=['ixi', 'brats2021'])
    parser.add_argument('--model', '-m', type=str, default='cyclegan', choices=['cyclegan', 'munit', 'unit'])
    parser.add_argument('--source-domain', '-s', type=str, default='t1', choices=['t1', 't2', 'pd', 'flair'])
    parser.add_argument('--target-domain', '-t', type=str, default='t2', choices=['t1', 't2', 'pd', 'flair'])
    parser.add_argument('--data-num', '-dn', type=int, default=None)
    parser.add_argument('--data-path', '-dp', type=str, default=None)
    parser.add_argument('--valid-path', '-vp', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=None)

    parser.add_argument('--noise-type', type=str, default='gaussian', choices=['normal', 'gaussian', 'slight', 'severe', 'kaid'])
    parser.add_argument('--gpu-id', '-g', type=str, default=None)
    parser.add_argument('--debug', action='store_true', default=None)
    parser.add_argument('--radius-stats', action='store_true', help='mask stastical learning')
    parser.add_argument('--radius-assigned', action='store_true', help='mask assigned flag')
    parser.add_argument('--radius-assigned-value', type=float, help='msl assigned value')
    parser.add_argument('--radius-path', type=str, default=None, help='mask side length storage path')
    parser.add_argument('--delta-diff', type=float, default=None, help='mask side length difference vairation thereshold value')
    parser.add_argument('--normalized-method', type=str, default=None, choices=['forward', 'backward', 'ortho'])
    parser.add_argument('--num-epochs', type=int, default=None)
    parser.add_argument('--lambda-recon', type=float, default=None, help='weight for reconstruction loss')
    parser.add_argument('--lambda-contrastive', type=float, default=None, help='weight for contrastive loss')
    parser.add_argument('--lambda-hf', type=float, default=None, help='weight for high frequency part')
    parser.add_argument('--lambda-lf', type=float, default=None, help='weight for low frequency part')
    #parser.add_argument('--mode', type=str, default='train', choices=['train', 'pred', 'trainpred'])
    parser.add_argument('--diff-method', type=str, default=None, choices=['l1', 'l2', 'cos'])
    parser.add_argument('--lr', type=float, default=None, help='learning rate')
    parser.add_argument('--step-size', type=int, default=None, help='learning rate will be adjust for epoch numbers')
    parser.add_argument('--gamma', type=float, default=None, help='Multiplicative factor of learning rate decay')
    parser.add_argument('--beta1', type=float, default=None, help='Adam Optimizer parameter')
    parser.add_argument('--beta2', type=float, default=None, help='Adam Optimizer parameter')
    parser.add_argument('--fid', action='store_true', default=True)
    parser.add_argument('--resume', action='store_true', default=None, help='resume training or not')
    parser.add_argument('--train', action='store_true', default=None)
    parser.add_argument('--validate', action='store_true', default=None)
    parser.add_argument('--load-latest', action='store_true', default=None, help='indiciate whether load the latest kaid model or not')
    parser.add_argument('--assigned-epoch', type=int, default=None, help='load which epoch of kaid model')
    #segmentation
    parser.add_argument('--segment', action='store_true', default=None, help='segmentation method evaluation or not')
    parser.add_argument('--seg-method', type=str, choices=['2d', '3d'])
    parser.add_argument('--seg-model', type=str, choices=['ex_unet', 'unet'])
    parser.add_argument('--max-filters-2d', type=int, default=999)
    # visualize check
    parser.add_argument('--vis-method', type=str, choices=['np', 'torch', 'torch_2d'])
    parser.add_argument('--vis-check', action='store_true', default=None, help='indiciate whether need to check visually')
    # method
    parser.add_argument('--method', type=str, choices=['combined', 'complex', 'normal'])
    # inference 
    parser.add_argument('--test-model', type=str, default=None, choices=['cyclegan','munit','unit'])
    parser.add_argument('--start-epoch', type=int, default=None)
    parser.add_argument('--end-epoch', type=int, default=None)

    args = parser.parse_args()
    return args

