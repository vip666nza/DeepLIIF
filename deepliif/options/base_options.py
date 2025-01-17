import os
import argparse
import sys

import torch

from deepliif.util import util


def create_base_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # added by Zeeon: formatter_class=argparse.ArgumentDefaultsHelpFormatter
    #                 在使用help时，同时会给出default信息
    
    # basic parameters
    parser.add_argument('--dataroot', required=True,
                        help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
    parser.add_argument('--name', type=str, default='experiment_name',
                        help='name of the experiment. It decides where to store samples and models')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
    parser.add_argument('--targets_no', type=int, default=5, help='number of targets')
    
    # model parameters
    parser.add_argument('--model', type=str, default='DeepLIIF', help='chooses which model to use. [DeepLIIF]')
    parser.add_argument('--input_nc', type=int, default=3,
                        help='# of inpu t image channels: 3 for RGB and 1 for grayscale')
    parser.add_argument('--output_nc', type=int, default=3,
                        help='# of output image channels: 3 for RGB and 1 for grayscale')
    parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
    parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
    parser.add_argument('--netD', type=str, default='n_layers',
                        help='specify discriminator architecture [basic | n_layers | pixel]. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator')
    parser.add_argument('--netG', type=str, default='resnet_9blocks',
                        help='specify generator architecture [resnet_9blocks | resnet_6blocks | unet_512 | unet_256 | unet_128]')
    parser.add_argument('--n_layers_D', type=int, default=4, help='only used if netD==n_layers')
    parser.add_argument('--norm', type=str, default='batch',
                        help='instance normalization or batch normalization [instance | batch | none]')
    parser.add_argument('--init_type', type=str, default='normal',
                        help='network initialization [normal | xavier | kaiming | orthogonal]')
    parser.add_argument('--init_gain', type=float, default=0.02,
                        help='scaling factor for normal, xavier and orthogonal.')
    parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
    
    # dataset parameters
    parser.add_argument('--dataset_mode', type=str, default='aligned',
                        help='chooses how datasets are loaded. [unaligned | aligned | single | colorization]')
    parser.add_argument('--direction', type=str, default='AtoB', help='AtoB or BtoA')
    parser.add_argument('--serial_batches', action='store_true',
                        help='if true, takes images in order to make batches, otherwise takes them randomly')
    parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--load_size', type=int, default=512, help='scale images to this size')
    parser.add_argument('--crop_size', type=int, default=512, help='then crop to this size')
    parser.add_argument('--max_dataset_size', type=int, default=None,
                        help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
    parser.add_argument('--preprocess', type=str, default='none',
                        help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
    parser.add_argument('--no_flip', action='store_true',
                        help='if specified, do not flip the images for data augmentation')
    parser.add_argument('--display_winsize', type=int, default=512, help='display window size for both visdom and HTML')
   
    # additional parameters
    parser.add_argument('--epoch', type=str, default='latest',
                        help='which epoch to load? set to latest to use latest cached model')
    parser.add_argument('--load_iter', type=int, default='0',
                        help='which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch]')
    parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
    parser.add_argument('--suffix', default='', type=str,
                        help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')
    parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
    parser.add_argument('--lambda_feat', type=float, default=100.0, help='weight for feature loss')
    parser.add_argument('--is_train', action='store_true', default=True)
    parser.add_argument('--seed', type=int, default=None, help='basic seed to be used for deterministic training, default to None (non-deterministic)')
    parser.add_argument('--padding', type=str, default='reflect', help='basic seed to be used for deterministic training, default to None (non-deterministic)')
    parser.add_argument('--remote-transfer-cmd', type=str, default=None,
                        help='module and function to be used to transfer remote files to target storage location, for example mymodule.myfunction')
    parser.add_argument('--remote', type=bool, default=False,
                        help='whether isolate visdom checkpoints or not; if False, you can run a separate visdom server anywhere that consumes the checkpoints')
    return parser


class BaseOptions():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.parser = create_base_parser()
        self.initialize()

    def initialize(self):
        # added by Zeeon:
        # 在train_option, test_option类被实例化的时候，由于它们继承了Base_option，所以会先执行__init__()中的操作
        # 即加载Base_option定义的parser后，调用重构的initialize()，加入train_args, test_args
        # train_option被实例化后，self.is_train = True
        # test_option被实例化后，self.is_train = False
        return NotImplementedError

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """

        # save and return the parser
        return self.parser.parse_args()

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        opt.is_train = self.is_train  # train or test

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt

    def create(self, dataroot, **kwargs):
        args = ['--dataroot', dataroot]
        for k, v in kwargs.items():
            args.append(f'--{k}') # added by Zeeon: f-string, kind of format contorlling
            args.append(v)

        opt = self.parser.parse_args(args)
        # added by Zeeon: why line159 - line164 is similar to self.read_options()?
        
        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        self.print_options(opt)

        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        return opt

    def read_options(self, dataroot, **kwargs):
        args = ['--dataroot', dataroot]
        for k, v in kwargs.items():
            args.append(f'--{k}')
            args.append(v)

        return self.parser.parse_args(args)
