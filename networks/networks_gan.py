import torch
import torch.nn as nn


# Functions
##############################################################################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# GAN
##############################################################################
class GeneratorFC(nn.Module):
    def __init__(self, latent_dim=128, noise_dim=8, n_features=(256, 512)):
        super(GeneratorFC, self).__init__()
        self.n_features = list(n_features)

        model = []
        prev_nf = latent_dim + noise_dim
        for idx, nf in enumerate(self.n_features):
            model.append(nn.Linear(prev_nf, nf))
            model.append(nn.LeakyReLU(inplace=True))
            prev_nf = nf

        model.append(nn.Linear(self.n_features[-1], latent_dim))

        self.model = nn.Sequential(*model)

        self.apply(weights_init)


    def forward(self, x, noise):
        x = torch.cat([x, noise], dim=1)
        x = self.model(x)
        return x


class DiscriminatorFC(nn.Module):
    def __init__(self, latent_dim=128, n_features=(256, 512)):
        super(DiscriminatorFC, self).__init__()
        self.n_features = list(n_features)

        model = []
        prev_nf = latent_dim
        for idx, nf in enumerate(self.n_features):
            model.append(nn.Linear(prev_nf, nf))
            model.append(nn.LeakyReLU(inplace=True))
            prev_nf = nf

        model.append(nn.Linear(self.n_features[-1], 1))

        self.model = nn.Sequential(*model)

        self.apply(weights_init)

    def forward(self, x):
        x = self.model(x).view(-1)
        return x


# Latent(noise) Regression
##############################################################################
class zEncoderFC(nn.Module):
    def __init__(self, latent_dim=128, z_dim=8, n_features=(256, 512)):
        super(zEncoderFC, self).__init__()
        self.n_features = list(n_features)

        model = []
        prev_nf = latent_dim
        for idx, nf in enumerate(self.n_features):
            model.append(nn.Linear(prev_nf, nf))
            model.append(nn.LeakyReLU(inplace=True))
            prev_nf = nf

        self.model = nn.Sequential(*model)

        self.fc_mu = nn.Linear(self.n_features[-1], z_dim)
        self.fc_logvar = nn.Linear(self.n_features[-1], z_dim)


    def forward(self, x):
        x = self.model(x)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        var = torch.exp(logvar / 2.)
        # N ~ N(0,1)
        z_size = mu.size()
        N = torch.normal(torch.zeros(z_size), torch.ones(z_size)).cuda()
        z = mu + var * N

        return z, mu, logvar


class zEncoderPointNet(nn.Module):
    def __init__(self, n_filters=(64, 128, 128, 256), z_dim=64, bn=True):
        super(zEncoderPointNet, self).__init__()
        self.n_filters = list(n_filters)
        self.noise_dim = z_dim

        model = []
        prev_nf = 3
        for idx, nf in enumerate(self.n_filters):
            conv_layer = nn.Conv1d(prev_nf, nf, kernel_size=1, stride=1)
            model.append(conv_layer)

            if bn:
                bn_layer = nn.BatchNorm1d(nf)
                model.append(bn_layer)

            act_layer = nn.ReLU(inplace=True)
            model.append(act_layer)
            prev_nf = nf

        model.append(nn.Conv1d(prev_nf, prev_nf // 2, kernel_size=1, stride=1))

        self.model = nn.Sequential(*model)

        self.fc_mu = nn.Linear(prev_nf // 2, z_dim)
        self.fc_logvar = nn.Linear(prev_nf // 2, z_dim)

    def forward(self, x):
        x = self.model(x)
        x = torch.max(x, dim=2)[0]

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        var = torch.exp(logvar / 2.)
        # N ~ N(0,1)
        z_size = mu.size()
        N = torch.normal(torch.zeros(z_size), torch.ones(z_size)).cuda()
        z = mu + var * N
        return z, mu, logvar


if __name__ == '__main__':
    pass
