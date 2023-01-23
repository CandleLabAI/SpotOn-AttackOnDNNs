# SpotOn

This repository contains the code for "SpotOn: A Gradient-based Targeted Data Poisoning Attack on Deep Neural Networks".

SpotOn proposes 3 attacks:
- Only In ROI
- Intense In ROI
- Variable Noise

## Attack Algorithm

![8YyyDdiE](https://user-images.githubusercontent.com/41234408/214020499-1e4fb8ed-a809-4b81-a625-e5ddee4c8259.jpeg)

**Comparison of FGSM with SpotOn-IntenseInRoI (ε = 0.1, λ = 0.5, K = 49.98)**

![9J8eYgCg](https://user-images.githubusercontent.com/41234408/214020946-d71e6561-9e78-41af-bd8f-dd8181e19435.png)

**Original image (top-left) and various perturbed images for different combinations of and SaliencyThreshold (λ)**

![iC8fwzyw](https://user-images.githubusercontent.com/41234408/214021390-97f8348a-3be0-4ffd-802f-698bd011cdc5.png)

## Performing the attacks

All of these attacks can be executed using the [`run_attack.py`](https://github.com/yashk2000/SpotOn/blob/main/run_attack.py) file. 

```python
usage: run_attack.py [-h] 
       -e EPSILONS [EPSILONS ...] 
       -m MEAN [MEAN ...] 
       -s STD [STD ...] 
       -g LOGS 
       -d DATA 
       -c CKPT 
       -a {vgg19,alexnet,googlenet} 
       -t {fgsm,variable_noise,only_in_roi,intense_in_roi} 
       -l THRESH_LAMBDA [-z Z]

optional arguments:
  -h, --help            show this help message and exit
  -e EPSILONS [EPSILONS ...], --epsilons EPSILONS [EPSILONS ...]
                        Set of epsilons to run attack with
  -m MEAN [MEAN ...], --mean MEAN [MEAN ...]
                        Mean of the dataset
  -s STD [STD ...], --std STD [STD ...]
                        Standard Devation of the dataset
  -g LOGS, --logs LOGS  Path to log file
  -d DATA, --data DATA  Path to the caltech dataset
  -c CKPT, --ckpt CKPT  Path to the model checkpoint
  -a {vgg19,alexnet,googlenet}, --arch {vgg19,alexnet,googlenet}
                        Model architechture
  -t {fgsm,variable_noise,only_in_roi,intense_in_roi}, --type_attack {fgsm,variable_noise,only_in_roi,intense_in_roi}
                        Type of attack to be launched
  -l THRESH_LAMBDA, --thresh_lambda THRESH_LAMBDA
                        Value of Lambda
  -z Z, --z Z           Value of Z for variable noise attack
```

The mean for the CalTech data is `(0.485, 0.456, 0.406)` and the standard deviation is `(0.229, 0.224, 0.225)`.

## Code examples

### Running Only In ROI

```python
python run_attack.py 
       --epsilons 0.005 0.006 0.007 
       --mean 0.485 0.456 0.406 
       --std 0.229 0.224 0.225 
       --logs only_roi.txt 
       --data 101_ObjectCategories/ 
       --arch vgg19 
       --ckpt vgg19.pth
       --type_attack only_in_roi 
       --thresh_lambda 0.3
```

### Running Intense In ROI

```python
python run_attack.py 
       --epsilons 0.01 
       --mean 0.485 0.456 0.406 
       --std 0.229 0.224 0.225 
       --logs intense_roi.txt
       --data 101_ObjectCategories/ 
       --arch alexnet 
       --ckpt alexnet.pth 
       --type_attack intense_in_roi
       --thresh_lambda 0.27
```

### Running Variable Noise

```python
python run_attack.py 
       --epsilons 0.3 
       --mean 0.485 0.456 0.406 
       --std 0.229 0.224 0.225 
       --logs variable_noise.txt 
       --data 101_ObjectCategories/ 
       --arch googlenet 
       --ckpt googlenet.pth 
       --type_attack variable_noise 
       --thresh_lambda 0.4 
       --z 4
```

### Running FGSM

```python
python run_attack.py 
       --epsilons 0.3 
       --mean 0.485 0.456 0.406 
       --std 0.229 0.224 0.225 
       --logs fgsm.txt 
       --data 101_ObjectCategories/ 
       --arch googlenet 
       --ckpt googlenet.pth 
       --type_attack fgsm
```

## Attack Implementations

- [Only In ROI](https://github.com/yashk2000/SpotOn/blob/bb0ea8d5eaf8ff5aa211490b1931de6589107af8/attacks.py#L27)
- [Intense In ROI](https://github.com/yashk2000/SpotOn/blob/bb0ea8d5eaf8ff5aa211490b1931de6589107af8/attacks.py#L16)
- [Variable Noise](https://github.com/yashk2000/SpotOn/blob/bb0ea8d5eaf8ff5aa211490b1931de6589107af8/attacks.py#L1)
- [FGSM (for baseline)](https://github.com/yashk2000/SpotOn/blob/bb0ea8d5eaf8ff5aa211490b1931de6589107af8/attacks.py#L38)

## Citation

If you are using this code, please cite this paper.

```
@inproceedings {khare2023SpotOn,
title            = {{SpotOn: A Gradient-based Targeted Data Poisoning Attack on Deep Neural Networks}},
year             = "2023",
author           = "Yash Khare and Kumud Lakara and Sparsh Mittal and Arvind Kaushik and Rekha Singhal",
booktitle        = "24th International Symposium on Quality Electronic Design (ISQED)",
address          = "California, USA",
}
```
