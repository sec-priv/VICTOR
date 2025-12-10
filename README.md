# VICTOR
Implementation of VICTOR

## Requirements

```
numpy >= 1.23.0
pandas >= 1.4.3
scikit-learn >= 0.24.1
scipy >= 1.9.1
python >= 3.8
mmaction2 >= 1.2.0
mmengine >= 0.10.7
mmcv >= 2.1.0
```

This project is based on [mmaction2](https://github.com/open-mmlab/mmaction2), we suggest that users first install mmaction2 by following the [installation instrcution](https://mmaction2.readthedocs.io/en/latest/get_started/installation.html). The users can finish the installation based on this folder. We encourange users to build mmaction2 from source (this folder). The install version can be python3.8, pytorch1.12+cu113 or python3.9, pytorch2.0+cu117.


## Introduction

Compared to the original mmaction2, this project mainly adds relevant code in `./data` folder.

1. data (folder): The datasets and core code are stored in this folder.
2. data/hmdb51 (folder): The HMDB51 dataset is stored in this folder.
3. data/ucf101 (folder): The UCF101 dataset is stored in this folder.
4. data/sthv2 (folder): The Something-something-v2 dataset is stored in this folder.
5. data/gen_noise_video.py (file): This file is used to generate the modified samples.
6. data/cal_post_prob.py (file): This file is used to calculate the probability of samples.
7. data/audit.py (file): The file is used to determine the threshold and conduct hypothesis testing.
8. data/utils.py (file): The file includes the functions that are needed for other files.
9. data/hmdb51/label_path (folder): This folder includes related training and test dataset division.
10. data/hmdb51/videos (folder): This folder is used to store the original and modified video files.


## Running

The users can download the HMDB51 and UCF101 datasets from the Internet, and the video files should be put into `./data/hmdb51/videos/` or `./data/ucf101/videos/` or `./data/sthv2/videos/`. The training config file is in `configs/recongition/{model_name}` folder, and the path can be modified based on the actual requirements. The detailed guide can be found in the [readthedocs](https://mmaction2.readthedocs.io/en/latest/get_started/quick_run.html).


```
######## Example 1: Train a model ########
python tools/train.py configs/recognition/i3d/i3d_imagenet-pretrained-r50_8xb8-32x2x1-100e_hmdb51-rgb_d1.py

######## Example 2: Generate a modified sample ########
python data/gen_noise_video.py --epsilon 10 --seed_n 1

######## Example 3: Calculate the probability difference ########
python data/cal_post_prob.py --epsilon 10 --seed_n 1

######## Example 4: Copyright verification ########
python data/audit.py
```


The audit process for stolen dataset is as follows:

Step 1: Generate modified samples following Example 2;

Step 2: Calculate the probability difference between modified samples and original samples following Example 3. Based on these results, select the reference set and modified set. Then, construct the training set text (which can be obtained by modifying `./data/{dataset_name}/label_path/dataset1_train0.txt`) and update the corresponding training set path (i.e., `ann_file_train`) in the configuration file (see `./configs/reconginiton/{model_path}/{model_config}.py`);

Step 3: Train the suspect model following Example 3;

Step 4: Complete the dataset auditing following Example 4.

The audit process for non-stolen datasets is as follows:

Step 1: Ensure the non-stolen dataset is stored in the relevant path;

Step 2: Construct the text for the non-stolen dataset (by modifying `./data/{dataset_name}/label_path/dataset1_train0.txt`) and modify the training set path (i.e., `ann_file_train`) in the configuration file (see `./configs/reconginiton/{model_path}/{model_config}.py`);

Step 3: Train the suspect model following Example 3;

Step 4: Complete the dataset auditing following Example 4.


## Acknowledgement

We sincerely thank the contributors from mmaction2 for their contributions to the community.


## Citation
```
@inproceedings{yuan2026victor,
  title={{VICTOR: Dataset Copyright Auditing in Video Recognition Systems}},
  author={Yuan, Quan and Zhang, Zhikun and Du, Linkang and Chen, Min and Sun, Mingyang and Gao, Yunjun and He, Shibo and Chen, Jiming},
  booktitle={NDSS},
  year={2026}
}
```


