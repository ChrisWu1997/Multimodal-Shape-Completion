from dataset.dataset_3depn import get_dataloader_3depn
from dataset.dataset_partnet_scan import get_dataloader_partnet_scan
from dataset.dataset_partnet import get_dataloader_partnet


def get_dataloader(phase, config):
    if config.dataset_name == '3depn':
        return get_dataloader_3depn(phase, config)
    elif config.dataset_name == 'partnet':
        return get_dataloader_partnet(phase, config)
    elif config.dataset_name == 'partnet_scan':
        return get_dataloader_partnet_scan(phase, config)
    else:
        raise ValueError
