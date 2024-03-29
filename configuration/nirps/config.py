import argparse

__all__ = ['parse_arguments_nirps']


def parse_arguments_nirps():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', '-d', type=str, default='brats2021', choices=['ixi', 'brats2021'])
    parser.add_argument('--model', '-m', type=str, default='cyclegan', choices=['cyclegan', 'munit', 'unit'])
    parser.add_argument('--num-epoch', type=int, default=None)
    parser.add_argument('--num-img-save', type=int, default=None)
    parser.add_argument('--general-evaluation', '-ge', action='store_true', default=None, help='indicate whether the evaluation for total images need to be done or not')
    parser.add_argument('--source-domain', '-s', default='flair', choices=['t1', 't2', 'pd', 'flair'])
    parser.add_argument('--target-domain', '-t', default='t2', choices=['t1', 't2', 'pd', 'flair'])
    parser.add_argument('--data-path', '-dp', type=str, default=None)
    parser.add_argument('--valid-path', '-vp', type=str, default=None)
    parser.add_argument('--data-mode', '-dm', type=str, default='mixed', choices=['mixed', 'paired', 'unpaired'])
    parser.add_argument('--data-num', type=int, default=None, help='slices number for GAN training')
    parser.add_argument('--gpu-id', '-g', type=str, default=None)
    parser.add_argument('--debug', action='store_true', default=None)
    #parser.add_argument('--nirps-structure', '-ns', action='store_true', help='flag for nirps structure')

    args = parser.parse_args()
    return args
