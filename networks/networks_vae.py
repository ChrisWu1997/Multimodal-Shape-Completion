import torch
import torch.nn as nn


# PointNet AutoEncoder
##############################################################################
class EncoderPointNet(nn.Module):
    def __init__(self, n_filters=(64, 128, 128, 256), latent_dim=128, z_dim=64, bn=True):
        super(EncoderPointNet, self).__init__()
        self.n_filters = list(n_filters) + [latent_dim]
        self.latent_dim = latent_dim
        self.z_dim = z_dim

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

        self.model = nn.Sequential(*model)

        self.fc_mu = nn.Linear(latent_dim, z_dim)
        self.fc_logvar = nn.Linear(latent_dim, z_dim)

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


class DecoderFC(nn.Module):
    def __init__(self, n_features=(256, 256), latent_dim=128, z_dim=64, output_pts=2048, bn=False):
        super(DecoderFC, self).__init__()
        self.n_features = list(n_features)
        self.output_pts = output_pts
        self.latent_dim = latent_dim
        self.z_dim = z_dim

        model = []
        prev_nf = self.latent_dim
        for idx, nf in enumerate(self.n_features):
            fc_layer = nn.Linear(prev_nf, nf)
            model.append(fc_layer)

            if bn:
                bn_layer = nn.BatchNorm1d(nf)
                model.append(bn_layer)

            act_layer = nn.ReLU(inplace=True)
            model.append(act_layer)
            prev_nf = nf

        fc_layer = nn.Linear(self.n_features[-1], output_pts*3)
        model.append(fc_layer)

        self.model = nn.Sequential(*model)

        self.expand = nn.Linear(z_dim, latent_dim)

    def forward(self, x):
        x = self.expand(x)
        x = self.model(x)
        x = x.view((-1, 3, self.output_pts))
        return x


class PointVAE(nn.Module):
    def __init__(self, config):
        super(PointVAE, self).__init__()
        self.encoder = EncoderPointNet(config.enc_filters, config.latent_dim, config.z_dim, config.enc_bn)
        self.decoder = DecoderFC(config.dec_features, config.latent_dim, config.z_dim, config.n_pts, config.dec_bn)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        z, mu, logvar = self.encoder(x)
        x = self.decoder(z)
        return x, mu, logvar


if __name__ == '__main__':
    pass
