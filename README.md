# MoVNect
Implementation of paper **["Lightweight 3D Human Pose Estimation Network Training Using Teacher-Student Learning"](https://arxiv.org/abs/2001.05097)** in PyTorch

### Environment
- Python 3.6
- pytorch 1.3.1
- torchvision 0.4.1
- CUDA 11.0
- Ubuntu 18.04

### Files
```
.  
|____README.md  
|____model  
  |____student.py  
  |____teacher.py  
  |____utils.py  
|____utils  
  |____loss.py  
  |____dataloader.py  
|____data  
  |____dummy.png  
  |____VNect.png  
  |____MoVNect.png  
|____test.py  
|____train.py  
```

### Networks
1. **VNect** - teacher network
![VNect network](https://raw.githubusercontent.com/ocean-joo/MoVNect/main/data/VNect.png)


2. **MoVNect** - student network
![MoVNect network](https://raw.githubusercontent.com/ocean-joo/MoVNect/main/data/MoVNect.png)



### TODO
I implemented network architecture, but there are some more functionalities that should be implemented to work properly.
1. train.py
    - To student network, teacher network must be pretrained.
    - Pretrain network for 2D pose estimation with 2D pose dataset, and fine-tune 3D pose estimation with 3D pose dataset.
2. test.py
    - inference and various evaluation metric.
3. dataloader.py
    - dataloader should support at least MPII dataset and Human3.6M dataset.
