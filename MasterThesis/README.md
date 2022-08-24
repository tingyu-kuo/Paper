# Master Thesis
## Enhancing Semi-Supervised Image Classification via Similar Pseudo Label and Feature Relationship<br>透過相似偽標籤和特徵關聯性增強半監督式影像分類
![image](https://user-images.githubusercontent.com/59983036/186312150-9707901f-aa44-4800-982e-5d39f05f684c.png)
## Method Comparison on two datasets
|                    |       CIFAR-10 (4000 labels)      |       CIFAR-100 (10000 labels)    |
|:------------------:|:-------------------:|:-------------------:|
|        **Method**      |     **Accuracy (%)**    |     **Accuracy (%)**    |
|       MixMatch     |         93.76       |         71.69       |
|      ReMixMatch    |         94.86       |         76.97       |
|       FixMatch     |         95.69       |         77.40       |
|        SimPLE      |         94.95       |         78.10       |
|      LaplaceNet    |         95.65       |         77.89       |
|      FlexMatch     |   **96.05**  |         78.10       |
|     DoubleMatch    |         95.35       |         78.78       |
|   **Ours**  |         95.34       |   **79.11**  |

## Visualization with CAM
![image width="200" height="200"](https://user-images.githubusercontent.com/59983036/186315923-67e651d0-ed1a-4887-8947-2294d0728d49.png)

## Implementation
A PyTorch implementation for the thesis is following [**SimPLE**](https://github.com/zijian-hu/SimPLE).

## Training
Default usage dataset: Mini-Imagenet, CIFAR-100, CIFAR-10 and SVHN.
- Train on CIFAR-100:
```
python main.py @runs/cifar100_args.txt
```
- Train on CIFAR-10:
```
python main.py @runs/cifar10_args.txt
```
