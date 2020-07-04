from networks.networks_ae import PointAE
from networks.networks_vae import PointVAE
from networks.networks_gan import GeneratorFC, DiscriminatorFC, zEncoderFC, zEncoderPointNet


def get_network(config, name):
    if name == "pointAE":
        return PointAE(config)
    elif name == "pointVAE":
        return PointVAE(config)
    elif name == "G":
        return GeneratorFC(config.latent_dim, config.z_dim, config.G_features)
    elif name == "D":
        return DiscriminatorFC(config.latent_dim, config.D_features)
    elif name == "zE_latent":
        return zEncoderFC(config.latent_dim, config.z_dim, config.zEnc_features)
    elif name == "zE_pc":
        return zEncoderPointNet(config.enc_filters, config.z_dim)
    else:
        raise NotImplementedError("Got name '{}'".format(name))


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad
