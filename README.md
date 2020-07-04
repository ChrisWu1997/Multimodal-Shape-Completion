# Multimodal-Shape-Completion

This repository provides PyTorch implementation of our paper:

[Multimodal Shape Completion via Conditional Generative Adversarial Networks](https://arxiv.org/abs/2003.07717)

[Rundi Wu](https://chriswu1997.github.io)\*, [Xuelin Chen](https://xuelin-chen.github.io)\*, [Yixin Zhuang](http://www.yixin.io/), [Baoquan Chen](http://cfcs.pku.edu.cn/baoquan/)

ECCV 2020 (spotlight)

<p align="center">
  <img src='teaser.png' width=400>
</p>

## Prerequisites

- Linux
- NVIDIA GPU + CUDA CuDNN
- Python 3.6



## Dependencies

1. Install python package dependencies through pip:

   ```bash
   pip install -r requirements.txt
   ```

2. Install external dependency [PyTorchEMD](https://github.com/daerduoCarey/PyTorchEMD) for the usage of EMD loss.



## Data

We use three datasets in our paper.

1. 3D-EPN

   Please download the partial scan point cloud data from [their website](http://kaldir.vc.in.tum.de/adai/CNNComplete/shapenet_dim32_sdf_pc.zip) and extract it into `data` folder. For the complete point clouds data, please download it from [PKU disk](https://dev-rc.teamviewer.com/LogOn) and extract it into `data` folder. Or you can follow our [instruction(placeholder)]() to virtually scan ShapeNet objects by yourself.

2. PartNet

   Please first download the [PartNet](https://www.shapenet.org/download/parts) dataset and then run our script to merge the point cloud segmentation label to the first level:

   ```bash
   cd data/
   python pc_merge.py --src {your-path-to-PartNet-dataset}
   ```

   Also, remember to change the data_root path in the corresponding scripts for training/testing on PartNet dataset.

3. PartNet-Scan

   TBA.



## Training

Training scripts can be found in `scripts` folder and please see `common.py` for specific definition of all command line parameters. For example, to train on 3DEPN chair category:

```bash
# 1. pre-train the PointNet AE
sh scripts/3depn/chair/train-3depn-chair-ae.sh

# 2. pre-train the PointNet VAE
sh scripts/3depn/chair/train-3depn-chair-vae.sh

# 3. train the conditional GAN for multimodal shape completion
sh scripts/3depn/chair/train-3depn-chair-gan.sh

```

Training log and model weights will be saved in `proj_log` folder by default. 



## Testing

Testing scripts can also be found in `scripts` folder. For example, to test the model trained on 3DEPN chair category:

```bash
# by default, run over all test examples and output 10 completion results for each
sh scripts/3depn/chair/test-3depn-chair-gan.sh
```

The completion results, along with the input parital shape, will be saved in `proj_log/mpc-3depn-chair/gan/results` by default. 



## Evaluation

Evaluation code can be found in `evaluation` folder. To evaluate the completion *diversity* and *fidelity*:

```bash
cd evaluation/
# calculate Total Mutual Difference (TMD)
python total_mutual_diff.py --src {path-to-saved-testing-results}
# calculate Unidirectional Hausdorff Distance (UHD) and completeness
python completeness.py --src {path-to-saved-testing-results}
```

To evaluate the *quality* of the completed shape, we use the Minimal Matching Distance (MMD). Please use its original [code](https://github.com/optas/latent_3d_points/blob/master/src/evaluation_metrics.py) for the calculation.



## Pre-trained models

TBA.



## Cite

Please cite our work if you find it useful:

```
@misc{wu2020multimodal,
    title={Multimodal Shape Completion via Conditional Generative Adversarial Networks},
    author={Rundi Wu and Xuelin Chen and Yixin Zhuang and Baoquan Chen},
    year={2020},
    eprint={2003.07717},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

