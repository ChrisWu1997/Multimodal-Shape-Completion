import torch
import numpy as np
from networks import get_network, set_requires_grad
from util.hausdorff import directed_hausdorff
from agent.base import GANzEAgent


class MainAgent(GANzEAgent):
    def __init__(self, config):
        super(MainAgent, self).__init__(config)

        self.weight_z_L1 = config.weight_z_L1
        self.weight_partial_rec = config.weight_partial_rec

    def build_net(self, config):
        # load pretrained pointAE
        self.pointAE = get_network(config, "pointAE")
        try:
            ae_weights = torch.load(config.pretrain_ae_path)['model_state_dict']
        except Exception as e:
            raise ValueError("Check the path for pretrained model of point AE. \n{}".format(e))
        self.pointAE.load_state_dict(ae_weights)
        self.pointAE = self.pointAE.eval().cuda()
        set_requires_grad(self.pointAE, False)

        # load pretrained pointVAE
        pointVAE = get_network(config, "pointVAE")
        try:
            vae_weights = torch.load(config.pretrain_vae_path)['model_state_dict']
        except Exception as e:
            raise ValueError("Check the path for pretrained model of point VAE. \n{}".format(e))
        pointVAE.load_state_dict(vae_weights)
        self.netE = pointVAE.encoder.eval().cuda()
        self.vaeD = pointVAE.decoder.eval().cuda()
        set_requires_grad(self.netE, False)  # netE remains fixed

        # build G, D
        self.netG = get_network(config, "G").cuda()
        self.netD = get_network(config, "D").cuda()

    def collect_loss(self):
        loss_dict = {"D_GAN": self.loss_D,
                     "G_GAN": self.loss_G_GAN,
                     "z_L1": self.loss_z_L1,
                     "partial_rec": self.loss_partial_rec}
        return loss_dict

    def forward(self, data):
        self.raw_pc = data['raw'].cuda()
        self.real_pc = data['real'].cuda()

        with torch.no_grad():
            self.raw_latent = self.pointAE.encode(self.raw_pc)
            self.real_latent = self.pointAE.encode(self.real_pc)

        self.forward_GE()

    def forward_GE(self):
        self.z_random = self.get_random_noise(self.raw_latent.size(0))

        self.fake_latent = self.netG(self.raw_latent, self.z_random)
        self.fake_pc = self.pointAE.decode(self.fake_latent)
        self.z_rec, z_mu, z_logvar = self.netE(self.fake_pc)

    def backward_D(self):
        # fake
        pred_fake = self.netD(self.fake_latent.detach())
        fake = torch.zeros_like(pred_fake).fill_(0.0).cuda()
        self.loss_D_fake = self.criterionGAN(pred_fake, fake)

        # real
        pred_real = self.netD(self.real_latent.detach())
        real = torch.ones_like(pred_real).fill_(1.0).cuda()
        self.loss_D_real = self.criterionGAN(pred_real, real)

        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def update_D(self):
        set_requires_grad(self.netD, True)

        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

    def backward_EG(self):
        # 1. G(A) fool D
        pred_fake = self.netD(self.fake_latent)
        real = torch.ones_like(pred_fake).fill_(1.0).cuda()
        self.loss_G_GAN = self.criterionGAN(pred_fake, real)

        # 2. noise reconstruction |E(G(A, z)) - z_random|
        self.loss_z_L1 = self.criterionL1(self.z_rec, self.z_random) * self.weight_z_L1

        # 3. partial scan reconstruction
        self.loss_partial_rec = directed_hausdorff(self.raw_pc, self.fake_pc) * self.weight_partial_rec

        self.loss_EG = self.loss_G_GAN + self.loss_z_L1 + self.loss_partial_rec
        self.loss_EG.backward()

    def update_G_and_E(self):
        set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_EG()
        self.optimizer_G.step()

    def optimize_network(self):
        self.update_G_and_E()
        self.update_D()

    def get_point_cloud(self):
        """get real/fake/raw point cloud of current batch"""
        real_pts = self.real_pc.transpose(1, 2).detach().cpu().numpy()
        fake_pts = self.fake_pc.transpose(1, 2).detach().cpu().numpy()
        raw_pts = self.raw_pc.transpose(1, 2).detach().cpu().numpy()
        return real_pts, fake_pts, raw_pts

    def visualize_batch(self, data, mode, **kwargs):
        tb = self.train_tb if mode == 'train' else self.val_tb

        num = 2

        real_pts = data['real'][:num].transpose(1, 2).detach().cpu().numpy()
        fake_pts = self.fake_pc[:num].transpose(1, 2).detach().cpu().numpy()
        raw_pts = self.raw_pc[:num].transpose(1, 2).detach().cpu().numpy()

        fake_pts = np.clip(fake_pts, -0.999, 0.999)

        tb.add_mesh("real", vertices=real_pts, global_step=self.clock.step)
        tb.add_mesh("fake", vertices=fake_pts, global_step=self.clock.step)
        tb.add_mesh("input", vertices=raw_pts, global_step=self.clock.step)
