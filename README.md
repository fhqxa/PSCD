# DPAL-Calib Dual-Pathway Adaptation with Logit Calibration for Long-Tailed Classification
## 📖 Introduction
Long-tailed recognition remains challenging because tail classes have limited training samples, making it difficult to distinguish them from competing classes. Existing methods address class imbalance through discriminative learning and representation learning. However, even empirical prototypes derived from learned representations support the target class, the classifier still tends to assign a higher score to a competing class, revealing a prototype-classifier discrepancy. In this paper, we propose prototype support to classifier decisions (PSCD), a unified framework combines differentiated discriminative supervision with prototype-guided discrepancy reduction for long-tailed recognition. First, the difficulty-balanced discrimination (DBD) module adjusts classification logits on mixed views according to class frequency and instance difficulty, adapting discriminative requirements for rare classes and difficult samples. The adjustments are scaled by the corresponding soft target values, allowing differentiated supervision without changing the original target distribution. Second, the prototype-guided discrepancy reduction (PDR) module compares target advantages from online empirical prototypes and projected classifier directions against the same active competitor on label-preserving clean views. PDR then selectively reduces local discrepancies for samples with strong prototype support but insufficient original logit separation. Experimental results on multiple long-tailed recognition benchmarks demonstrate the effectiveness of PSCD. Compared with representative long-tailed recognition methods, PSCD improves overall classification accuracy and few-group accuracy by 1.9\% and 4.3\%, respectively.

![](./figs/figure.png)

## 📂 Dataset Preparation

---

Please download the benchmark datasets and organize them into the dataset/ directory. The expected file structure is as follows:

dataset/
├── cifar.py                    
├── imagenet.py                   
├── inaturalist.py                                  
├── ImageNet_LT/
│   ├── ImageNet_LT_train.txt
│   ├── ImageNet_LT_val.txt
│   └── ImageNet_LT_test.txt
├── iNaturalist18/
    ├── categories.json
    ├── iNaturalist18_train.txt
    └── iNaturalist18_val.txt

## Run
0. Requirements:
* Python 3.10.19
* pytorch 2.9.1
* torchvision 0.24.1
1. Training:
* Prepare the required images and store them in categories, set up training image folders and validation image folders respectively
* Run
* python run_pscd.py -d inat2018 --data /path/to/iNaturalist18
* python run_pscd.py -d imagenet_lt --data /path/to/ImageNet2012
* python run_pscd.py -d cifar100_ir100 --data /path/to/CIFAR100
* python run_pscd.py -d cifar10_ir100 --data /path/to/CIFAR10






