# ALA: Naturalness-aware Adversarial Lightness Attack
This is the official repository of [ALA: Naturalness-aware Adversarial Lightness Attack](https://arxiv.org/pdf/2201.06070.pdf).
The paper is accepted by ACM MM 2023.

[![arXiv](https://img.shields.io/badge/arXiv-2208.01618-b31b1b.svg)]([https://arxiv.org/abs/2201.06070](https://arxiv.org/pdf/2201.06070.pdf))

> **ALA Naturalness-aware Adversarial Lightness Attack**<br>
> Yihao Huang, Liangru Sun, Qing Guo, Felix Juefei-Xu, Jiayi Zhu, Jincao Feng, Yang Liu, Geguang Pu <br>

>**Abstract**: <br>
> Most researchers have tried to enhance the robustness of DNNs by revealing and repairing the vulnerability of DNNs with specialized adversarial examples. Parts of the attack examples have imperceptible perturbations restricted by Lp norm. However, due to their high-frequency property, the adversarial examples can be defended by denoising methods and are hard to realize in the physical world. To avoid the defects, some works have proposed unrestricted attacks to gain better robustness and practicality. It is disappointing that these examples usually look unnatural and can alert the guards. In this paper, we propose Adversarial Lightness Attack (ALA), a white-box unrestricted adversarial attack that focuses on modifying the lightness of the images. The shape and color of the samples, which are crucial to human perception, are barely influenced. To obtain adversarial examples with a high attack success rate, we propose unconstrained enhancement in terms of the light and shade relationship in images. To enhance the naturalness of images, we craft the naturalness-aware regularization according to the range and distribution of light. The effectiveness of ALA is verified on two popular datasets for different tasks (i.e., ImageNet for image classification and Places-365 for scene recognition).


# Requirements

```
python3
torch == 1.8.0
torchvision == 0.9.0
```

# References
```
@inproceedings{huang2023ala,
  title={ALA: Naturalness-aware Adversarial Lightness Attack},
  author={Huang, Yihao and Sun, Liangru and Guo, Qing and Juefei-Xu, Felix and Zhu, Jiayi and Feng, Jincao and Liu, Yang and Pu, Geguang},
  booktitle={Proceedings of the 31st ACM International Conference on Multimedia},
  pages={2418--2426},
  year={2023}
}
```
