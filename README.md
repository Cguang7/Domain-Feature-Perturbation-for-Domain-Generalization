>📋  This is the code repository of the paper Domain Feature Perturbation (DFP) for Domain Generalization. This repository contains the core code for DFP and code modified from the PyTorch suite [DomainBed](https://github.com/facebookresearch/DomainBed) and Benchmark [OoD-Bench](https://github.com/ynysjtu/ood_bench).

# Domain Feature Perturbation for Domain Generalization

This repository is the official implementation of Domain Feature Perturbation for Domain Generalization. 

## Requirements

Python 3.8, Pytorch 1.8
To install requirements:

```anaconda env
conda env create -f environment.yaml
```
please modify the prefix to your anaconda dir

## Dataset
make sure the directory structures are as follows:

```
PACS
└── kfold
    ├── art_painting
    ├── cartoon
    ├── photo
    └── sketch
```

```
office_home
├── Art
├── Clipart
├── Product
├── RealWorld
├── ImageInfo.csv
└── imagelist.txt
```

```
TerraIncognita
├── location_38
├── location_43
├── location_46
└── location_100
```

## Training

To train the model(s) in the paper, run this command:

```PACS
bash sweep/PACS/run.sh launch /root/data 4,5,6,7
```

```OfficeHome
bash sweep/OfficeHome/run.sh launch /root/data 4,5,6,7
```

```TerraIncognita
bash sweep/TerraIncognita/run.sh launch /root/data 0,1,2,3
```

>📋  You need to change "/root/data" to your own dir of corresponding dataset. "4,5,6,7" means the GPU ID you suppose to use.

## Collect Results

To collect the accuracy results, run:

```
python -m domainbed.scripts.collect_results\
       --input_dir=/root/output/PACS
```

>📋  Please change the input_dir to your results dir.

## Pre-trained Models

You can download pretrained models here:

- [Resnet-18 pretrained model](https://download.pytorch.org/models/resnet18-5c106cde.pth) trained on ImageNet. 

