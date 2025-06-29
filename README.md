# RobustFace: Adaptive Mining of Noise and Hard Samples for Robust Face Recognitions

## Introduction
 we propose an updated deep face recognition model: RobustFace, and, 
 in comparison with the existing models, it has the feature that 
 training is proceeded with an adaptive noise and hard sample mining 
 loss function, which is designed to improve the robustness of deep 
 learning models in the presence of both closed-set and open-set 
 noises, enabling direct learning of more effective facial features 
 on large-scale noisy datasets. This can be 
 verified by observing not only the distribution of the angles between 
 noise and clean samples, but also the distribution of the current 
 class centre at different training stages. 
 
<img src="illustration/loss_work_flow.png" width="80%" height="80%"/>

## Requirements

In order to enjoy the new features of pytorch, we have upgraded the pytorch to 1.9.0.  
Pytorch before 1.9.0 may not work in the future.

- `pip install -r requirement.txt`.

## Download Datasets or Prepare Datasets
[InsightFace](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_) provides a range of preprocessed 
labeled face datasets for  RobustFace.

**Training Datasets:**

We adopt several large-scale face recognition datasets for training:
- MS1MV2 (87K IDs, 5.8M images)
- [MS-Celeb-1M ](https://www.microsoft.com/en-us/research/project/ms-celeb-1m-challenge-recognizing-one-million-celebrities-real-world/) (100k IDs, 10M images)
- Glint360K (360K IDs, 18M images)

🔹We also provide a high-quality cleaned version of MS-Celeb-1M:

- MS-Celeb-C (77K IDs, 4.9M images)
- `The cleaned file list (MS-Celeb-C.txt) is available, and the full cleaned dataset is provided as ./MS-Celeb-C.rar in this repository.`

The cleaning is conducted through a automated framework built upon RobustFace, which effectively 
identifies and removes noisy, mislabeled, and low-quality samples (details provided in the Clean Dataset Framework section).


**Validation Datasets:** 

We evaluate model performance on the following standard face verification benchmarks:

- LFW (5749 IDs, 13,233 images, 6k Pairs)
- AgeDB-30 (570 IDs, 12,240 images, 7k Pairs)
- CFP-FP (500 IDs, 7k images, 7k Pairs)
- CALFW (5749 IDs, 13,233 images, 6k Pairs)
- CPLFW 5749 IDs, 13,233 images, 6k Pairs)

**Testing Datasets:** 

We conduct both face verification (1:1) and identification (1:N) experiments:

1:1 Verification:

- 1:1:   IJB (IJB-B, IJB-C)

1:1 & 1:N Evaluation:

- 1:1 & 1:N   MegaFace

**Synthetic Noise Datasets:** 

To provide a comprehensive evaluation on the proposed RobustFace, we follow the work reported in [1] to apply two types of synthetic noise datasets, `close-set noise` and `open-set noise`, to the training
of the assessed models. While the close-set noise is introduced by randomly changing the labels of the facial images inside the training dataset, the open-set noise  is introduced by changing the label of those facial 
images that are not included inside the training dataset. For close-set noises, the labels of MS1MV2 samples are randomly flipped, and for open-set noises, `Glint360K` is selected as the source, and `MS1MV2` samples are randomly 
replaced. 

- Close-set Noise:
Random label flipping within the original training set (e.g., MS1MV2)

- Open-set Noise:
Injecting unseen identities by replacing MS1MV2 samples with identities from an external dataset (e.g., Glint360K)

These synthetic settings allow controlled benchmarking of model robustness against both in-distribution and out-of-distribution noise.

## How to training
To train a model, run `train.py` with the path to the configs.
The example commands below show how to run distributed training.

```shell
python -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=12581 train.py configs/ms_celeb_1m_r100.py
```
If you want to train on a machine with multiple GPUs, you can achieve this by `--nproc_per_node`. For example, on a machine with 8 GPUs:
```shell
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=12581 train.py configs/ms_celeb_1m_r100.py
```

## Experimental results

**1:1 Verification TAR@FAR on IJB-B and IJB-C**

| Method (%)         | IJB-B @1e-5 | IJB-B @1e-4 | IJB-B @1e-3 | IJB-C @1e-5 | IJB-C @1e-4 | IJB-C @1e-3 |
| ------------------ | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| ArcFace            | 63.35       | 85.03       | 93.26       | 70.77       | 87.73       | 94.65       |
| CurricularFace     | 74.30       | 89.04       | 94.48       | 83.50       | 92.05       | 95.79       |
| Sub-center ArcFace | 74.33       | 90.41       | 94.53       | 83.57       | 92.30       | 95.84       |
| AdaFace            | 72.83       | 89.03       | 94.44       | 83.55       | 91.95       | 95.75       |
| RVFace             | 72.03       | 89.18       | 94.38       | 83.58       | 92.26       | 95.80       |
| BoundaryFace       | 65.37       | 84.80       | 93.36       | 74.55       | 88.13       | 94.65       |
| **The Proposed**   | **74.38**   | **91.30**   | **95.57**   | **85.50**   | **94.02**   | **96.69**   |


**1:1 Verification & 1:N Identification on MegaFace**

| Method (%)         | Large Ident | Large Verif | Large/C Ident | Large/C Verif |
| ------------------ | ----------- | ----------- | ------------- | ------------- |
| ArcFace            | 77.81       | 93.69       | 95.53         | 95.68         |
| CurricularFace     | 78.68       | 94.26       | 95.66         | 95.72         |
| Sub-center ArcFace | 80.43       | 95.72       | 96.74         | 96.81         |
| AdaFace            | 78.94       | 95.18       | 96.69         | 96.89         |
| RVFace             | 81.16       | 96.04       | 96.33         | 96.42         |
| BoundaryFace       | 81.59       | 96.17       | 96.67         | 96.67         |
| **The Proposed**   | **81.83**   | **96.83**   | **97.75**     | **97.81**     |


**Experimental Results Under Close&Open Noise:**

| Method             | Train Data       | LFW       | CALFW     | AgeDB     | CFP-FP    | CPLFW     | AVG       |
| ------------------ | ---------------- | --------- | --------- | --------- | --------- | --------- | --------- |
| ArcFace            | MS1MCON (r=0.20) | 90.03     | 76.85     | 70.25     | 68.60     | 81.18     | 77.78     |
| CurricularFace     |      MS1MCON (r=0.20)             | 92.32     | 82.16     | 74.98     | 70.01     | 81.41     | 80.18     |
| Sub-center ArcFace |      MS1MCON (r=0.20)             | 94.21     | 83.14     | 76.11     | 70.11     | 84.88     | 81.29     |
| AdaFace            |       MS1MCON (r=0.20)            | 91.97     | 81.43     | 72.16     | 71.62     | 84.93     | 80.82     |
| RVFace             |       MS1MCON (r=0.20)            | 95.83     | 84.21     | 83.21     | 74.10     | 84.10     | 84.29     |
| BoundaryFace       |       MS1MCON (r=0.20)            | 91.13     | 81.48     | 73.88     | 69.96     | 82.43     | 79.78     |
| **The Proposed**   |       MS1MCON (r=0.20)            | **99.75** | **96.08** | **97.62** | **96.14** | **91.28** | **96.17** |




The full experimental results will be presented after publication of the paper.


## Clean Datasets Framework

We propose a fully automated cleaning framework based on RobustFace to remove:

- mislabeled identities
- low-resolution or heavily blurred face images
- non-photorealistic faces (e.g., sketches or cartoon-like images)
- non-face images

The picture below shows the examples of noisy images that are removed from MS-Celeb-1M by the proposed cleaning framework:

<img src="illustration/clean_id_img.png" width="80%" height="80%"/>

Images highlighted with red boxes represent samples identified as noisy, while yellow boxes indicate hard samples. 


The cleaning process proceeds as follows:

<img src="illustration/clean_imgs.png" width="80%" height="80%"/>

Overview of the proposed automated dataset cleaning framework based on self-training with our proposed loss. 
The process starts by training a model on MS-Celeb-1M using our proposed loss. Samples identified as noisy are removed. 
The model is then retrained on the cleaned subset, and this cycle of cleaning-retraining is repeated iteratively until 
the dataset is purified.

**Cleaning Effectiveness**

Below we show the comparison results between the original MS-Celeb-1M and our cleaned version **MS-Celeb-C**. All models 
achieve **significant performance improvements** when trained on our cleaned dataset.

**Table: Verification accuracy (%) on AgeDB and CFP-FP before and after MS-Celeb-1M cleaning.**

| Method (%)     | AgeDB Original | AgeDB Cleaned | CFP-FP Original | CFP-FP Cleaned |
|----------------|----------------|----------------|------------------|-----------------|
| ArcFace        | 96.82          | 98.30          | 95.37            | 98.33           |
| CurricularFace | 96.82          | 98.38          | 95.58            | 98.42           |
| **Ours**       | 97.55          | 98.31          | 95.87            | 98.34           |



## Acknowledgements

This code is largely based on [InsightFace](https://github.com/deepinsight/insightface/). We thank the authors a lot for their valuable efforts.

## Reference
[1] Wu, Shijie, and Xun Gong. "BoundaryFace: A mining framework with noise label self-correction for Face Recognition." European Conference on Computer Vision. Cham: Springer Nature Switzerland, 2022.
