# On the Effectiveness of Supervision in asymmetric Non-Contrastive Representation Learning
## Jeongheon Oh, Kibok Lee

This repository contains the official implementation for the paper On the Effectiveness of Supervision in asymmetric Non-Contrastive Representation Learning

This work is accepted in
- [ICML 2024](https://openreview.net/forum?id=iC8l9DI1ZX)

```bibtex
@inproceedings{oh2024supancl,
  title={On the Effectiveness of Supervision in Asymmetric Non-Contrastive Learning},
  author={Jeongheon Oh, Kibok Lee},
  booktitle={ICML},
  year={2024}
}
```
## Installation
To install the necessary dependencies, run:
```python
pip install -r requirements.txt
```

## Data
- For ImageNet, please refer to the [[PyTorch ImageNet example](https://github.com/pytorch/examples/tree/main/imagenet)]. The folder structure should be like 
- CIFAR-10/100 will automatically be downloaded

## Pre-Training
Only multi-gpu, DistributedDataParallel training is supported; single-gpu or DataParallel training is not supported.
```python
python main.py \
 -a resnet50 --framework 'supbyol' --warmup_epoch 0 --lr 0.4 \
 --dim 256 --hid-dim 4096 --K 8192 --m 0.996 --m-cos --T 0.0 \
 --fix-pred-lr --num_positive 0 --alpha 0.5 \
 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
 --dir [your imagenet-folder with train and val folders]
```
```python
python main.py \
 -a resnet50 --framework 'supsiam' --warmup_epoch 40 --lr 0.4 \
 --dim 2048 --hid-dim 512 --K 8192 --m 0.0 --T 0.0 \
 --fix-pred-lr --num_positive 0 --alpha 0.5 \
 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
 --dir [your imagenet-folder with train and val folders]
```

## Linear Evaluation
With a pre-trained model, to train a supervised linear classifier on frozen features/weights run;
```python
python linear.py \
 -a resnet50 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
 --dir [your imagenet-folder with train and val folders] \
 --pretrained [path where pretrained model is saved]
```

## Transfer Learning via Linear Evaluation
With a pre-trained model, to train a supervised linear classifier on frozen features/weights run;
```python
python transfer.py \
 -a resnet50 --data 'CIFAR100' --metric 'top1'  \
 --dir [your dataset folder] \
 --pretrained [path where pretrained model is saved]
```

## Acknowledgement
We appreciate the following github repositories for their valuable code base & datasets:
- [SimSiam](https://github.com/facebookresearch/simsiam/tree/main)
- [MoCo](https://github.com/facebookresearch/moco)
- [AugSelf](https://github.com/hankook/AugSelf/tree/main)
- [MoCo-v3](https://github.com/facebookresearch/moco-v3)
- [SupContrast](https://github.com/HobbitLong/SupContrast)
- [targeted-supcon](https://github.com/LTH14/targeted-supcon)
