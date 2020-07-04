from tqdm import tqdm
import torch
import os
from dataset import get_dataloader
from common import get_config
from agent import get_agent
from util.pc_utils import write_ply
from util.utils import cycle, ensure_dir
import random

random.seed(1856)


def main():
    # create experiment config containing all hyperparameters
    config = get_config('test')

    # create network and training agent
    tr_agent = get_agent(config)

    # load from checkpoint
    tr_agent.load_ckpt(config.ckpt)
    tr_agent.eval()

    # create dataloader
    config.batch_size = 1
    config.num_workers = 1
    test_loader = get_dataloader('test', config)
    num_test = len(test_loader)
    print("total number of test samples: {}".format(num_test))
    used_test = num_test if config.num_sample == -1 else config.num_sample
    print("used number of test samples: {}".format(used_test))
    test_loader = cycle(test_loader)

    save_dir = os.path.join(config.exp_dir, "results/ckpt-{}-n{}-z{}".format(config.ckpt, used_test, config.num_z))
    ensure_dir(save_dir)

    # run
    for i in tqdm(range(used_test)):
        data = next(test_loader)

        for j in range(config.num_z):
            with torch.no_grad():
                tr_agent.forward(data)
            real_pts, fake_pts, raw_pts = tr_agent.get_point_cloud()

            raw_id = data['raw_id'][0].split('.')[0]
            save_sample_dir = os.path.join(save_dir, "{}".format(raw_id))
            ensure_dir(save_sample_dir)

            # save input partial shape
            if j == 0:
                save_path = os.path.join(save_sample_dir, "raw.ply")
                write_ply(raw_pts[0], save_path)
            # save completed shape
            save_path = os.path.join(save_sample_dir, "fake-z{}.ply".format(j))
            write_ply(fake_pts[0], save_path)


if __name__ == '__main__':
    main()
