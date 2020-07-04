import os
from util.utils import ensure_dirs
import argparse
import json
import shutil


def get_config(phase):
    config = Config(phase)
    return config


class Config(object):
    """Base class of Config, provide necessary hyperparameters. 
    """
    def __init__(self, phase):
        self.is_train = phase == "train"

        # init hyperparameters and parse from command-line
        parser, args = self.parse()

        # set as attributes
        print("----Experiment Configuration-----")
        for k, v in args.__dict__.items():
            print("{0:20}".format(k), v)
            self.__setattr__(k, v)

        # experiment paths
        self.exp_dir = os.path.join(self.proj_dir, self.exp_name, self.module)
        if phase == "train" and args.cont is not True and os.path.exists(self.exp_dir):
            response = input('Experiment log/model already exists, overwrite to retrain? (y/n) ')
            if response != 'y':
                exit()
            shutil.rmtree(self.exp_dir)

        self.log_dir = os.path.join(self.exp_dir, 'log')
        self.model_dir = os.path.join(self.exp_dir, 'model')
        ensure_dirs([self.log_dir, self.model_dir])

        # GPU usage
        if args.gpu_ids is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)

        # create soft link to experiment log directory
        # if not os.path.exists('train_log'):
        #     os.symlink(self.exp_dir, 'train_log')

        # save this configuration
        if self.is_train:
            with open(os.path.join(self.exp_dir, 'config.txt'), 'w') as f:
                json.dump(args.__dict__, f, indent=2)

    def parse(self):
        """initiaize argument parser. Define default hyperparameters and collect from command-line arguments."""
        parser = argparse.ArgumentParser()
        
        # basic configuration
        self._add_basic_config_(parser)

        # dataset configuration
        self._add_dataset_config_(parser)

        # model configuration
        self._add_network_config_(parser)

        # training configuration
        self._add_training_config_(parser)

        if not self.is_train:
            # testing configuration
            self._add_testing_config_(parser)

        # additional parameters if needed
        pass

        args = parser.parse_args()
        return parser, args

    def _add_basic_config_(self, parser):
        """add general hyperparameters"""
        group = parser.add_argument_group('basic')
        group.add_argument('--proj_dir', type=str, default="proj_log",
                           help="path to project folder where experiment logs/models will be saved")
        group.add_argument('--exp_name', type=str, default=os.getcwd().split('/')[-1],
                           help="name of this experiment")
        group.add_argument('-g', '--gpu_ids', type=str, default=None,
                           help="gpu to use, e.g. 0  0,1,2. CPU not supported.")
        group.add_argument('--module', type=str, choices=['ae', 'vae', 'gan'], required=True,
                           help="which network module to set")

    def _add_dataset_config_(self, parser):
        """add hyperparameters for dataset configuration"""
        group = parser.add_argument_group('dataset')
        group.add_argument('--dataset_name', type=str, choices=['partnet', 'partnet_scan', '3depn'], required=True,
                           help="which dataset to use")
        group.add_argument('--data_root', type=str, default="", help="path to complete shape data")
        group.add_argument('--data_raw_root', type=str, default="", help="path to partial shape data")
        group.add_argument('--batch_size', type=int, default=50, help="batch size")
        group.add_argument('--category', type=str, default="airplane", help="shape category name")
        group.add_argument('--num_workers', type=int, default=8, help="number of workers for data loading")

        group.add_argument('--n_pts', type=int, default=2048,
                           help="number of points sampled for complete shape. Half of it for partial shape")

    def _add_network_config_(self, parser):
        """add hyperparameters for network architecture"""
        group = parser.add_argument_group('network')
        # ae encoder
        self.enc_filters = (64, 128, 128, 256)
        group.add_argument('--latent_dim', type=int, default=128)
        group.add_argument('--enc_bn', type=bool, default=True)

        # ae decoder
        self.dec_features = (256, 256)
        group.add_argument('--dec_bn', type=bool, default=False)

        # GAN
        self.zEnc_features = (256, 512)
        self.G_features = (256, 512)
        self.D_features = (256, 512)
        parser.add_argument('--z_dim', type=int, default=64, help="dimension for mode condition vector z")
        parser.add_argument('--pretrain_ae_path', type=str,
                            help="path for pretrained ae model, only needed when training/testing cGAN")
        parser.add_argument('--pretrain_vae_path', type=str,
                            help="path for pretrained vae model, only needed when training/testing cGAN")

    def _add_training_config_(self, parser):
        """training configuration"""
        group = parser.add_argument_group('training')
        group.add_argument('--nr_epochs', type=int, default=2000, help="total number of epochs to train")
        group.add_argument('--lr', type=float, default=5e-4, help="initial learning rate")
        group.add_argument('--lr_decay', type=float, default=0.9995, help="step size for learning rate decay")
        group.add_argument('--beta1_gan', type=int, default=0.5, help="beta1 for Adam Optimizer when training gan")
        group.add_argument('--parallel', action='store_true', help="use multiple GPU for parallel training")
        group.add_argument('--continue', dest='cont',  action='store_true', help="continue training from checkpoint")
        group.add_argument('--ckpt', type=str, default='latest', required=False, help="desired checkpoint to restore")
        group.add_argument('--vis', action='store_true', default=False, help="visualize output in tensorboard")
        group.add_argument('--save_frequency', type=int, default=100, help="save models every x epochs")
        group.add_argument('--val_frequency', type=int, default=10, help="run validation every x iterations")
        group.add_argument('--vis_frequency', type=int, default=1000, help="visualize output every x iterations")

        # vae loss factor
        parser.add_argument('--weight_kl_vae', type=float, default=10.0,
                           help="weight factor for kl loss")

        # cGAN loss factor
        parser.add_argument('--weight_z_L1', type=float, default=7.5,
                           help="weight factor for latent reconstruction loss")
        parser.add_argument('--weight_partial_rec', type=float, default=6,
                           help="weight factor for partial reconstruction loss")

    def _add_testing_config_(self, parser):
        group = parser.add_argument_group('testing')
        group.add_argument('--num_sample', type=int, default=10, help="number test samples to use, -1 for all")
        group.add_argument('--num_z', type=int, default=5, help="number of completion outputs per sample")


if __name__ == '__main__':
    pass
