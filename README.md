# PAMUS_unlearning
My undergraduate graduation project: A data poisoning attack method for unlearning. It mainly uses data poisoning attacks to make the model forget the data when executing the unlearning algorithm, so that the poisoned data will have a greater negative impact on the model parameters, thereby reducing the model performance.

The following README comes from a senior in the research group. This graduation project is also based on her previous semi-finished work: [Link](https://github.com/Weiww-Xu/unlearning-attack)

The original directory structure is as follows, this repo only contains the bold parts.

.
├── BSS-generation
│   ├── BSS_distillation
│   ├── Conditional-GANs-Pytorch
│   └── README.md
├── README.md
├── related-work
│   └── attack-unlearning
├── unlearning-algorithm
│   ├── AmnesiacML
│   ├── DeltaGrad
│   ├── LCODEC-deep-unlearning
│   ├── SelectiveForgetting
│   ├── certified_removal
│   ├── mcmc-unlearning
│   └── unrolling-sgd
└── **unlearning-attack**
    └── **LCODEC-deep-unlearning**

-----------------------------------------------------------------------------------

# Unlearning Attack(from xw)

## Unlearning Algorithm
代码都在源代码基础上做过一些改动，可以去link里找源代码。

| 算法        | 论文                                                        | 会议       | 代码                                                               | 实验         |
|-----------|-----------------------------------------------------------|----------|------------------------------------------------------------------|------------|
| Certified removal | Certified Data Removal from Machine Learning Models     | ICML 2020 | [Link](https://github.com/facebookresearch/certified-removal)    | 训练阶段投毒<br>遗忘阶段修改 |
| DeltaGrad    | DeltaGrad: Rapid retraining of machine learning models    | ICML 2020 | [Link](https://github.com/thuwuyinjun/DeltaGrad)                | 训练阶段投毒<br>遗忘阶段修改 |
| LCDECC     | Deep Unlearning via Randomized Conditionally Independent Hessians | CVPR 2022 | [Link](https://github.com/vsingh-group/LCODEC-deep-unlearning) | 训练阶段投毒<br>遗忘阶段修改 |
| Fisher(Selective Forgetting)     | Eternal Sunshine of the Spotless Net: Selective Forgetting in Deep Networks | CVPR 2020 | [Link](https://github.com/AdityaGolatkar/SelectiveForgetting)  | 训练阶段投毒 |
| Unrolling SGD | Unrolling SGD: Understanding Factors Influencing Machine Unlearning | EuroS&P'22 | [Link](https://github.com/cleverhans-lab/unrolling-sgd)         | 训练阶段投毒<br>遗忘阶段修改 |

## Related Work
可以参考的相关工作，也是做unlearning投毒的。

> Marchant, N. G., Rubinstein, B. I. P., & Alfeld, S. (2022). Hard to Forget: Poisoning Attacks on Certified Machine Unlearning. Proceedings of the AAAI Conference on Artificial Intelligence (to appear). [arXiv:2109.08266](https://arxiv.org/pdf/2109.08266.pdf)

## BBS Generation
之前用来生成boundary supporting samples（BSS）的代码，训练一个模型来生成可能落在攻击模型决策边界上的样本。用于训练前投毒，但没什么效果。

## Unlearning Attack 
只在LCDECC上实验，其他做过实验都没什么效果。详见`unlearning-attack`中的readme文件。


## Experiment
1. **（重点）** 只跑`unlearning-attack`文件夹中的实验即可，详见`unlearning-attack`中的readme文件。

2. （选做）`BSS-generation`中生成可能的决策边界样本，用于投毒，可以改进一下，说不定有效果。

3. （选做）有时间可以试试`unlearning-algorithm`中其他遗忘算法的攻击，说不定有新的攻击思路。

## Experiment Environment
可供参考，版本变动一点问题也不大：
- python 3.9
- pytorch 1.12.0
- cuda 11.3

-----------------------------------------------------------------------------

# origin method README

#  Deep Unlearning via Randomized Conditionally Independent Hessians (CVPR 2022)
[[Paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Mehta_Deep_Unlearning_via_Randomized_Conditionally_Independent_Hessians_CVPR_2022_paper.pdf) [[Slide]](https://github.com/vsingh-group/LCODEC-deep-unlearning/blob/main/assets/DeepUnlearningCVPR22.pdf)

#### Ronak Mehta*, Sourav Pal*, Vikas Singh, Sathya N. Ravi
(* Joint First authors)
![LCODEC Pipeline](/assets/lfoci_pipeline.png?raw=true)

## Abstract
Recent legislation has led to interest in machine unlearning, i.e., removing specific training samples from a predictive model as if they never existed in the training dataset. Unlearning may also be required due to corrupted/adversarial data or simply a user’s updated privacy requirement. For models which require no training (k-NN), simply deleting the closest original sample can be effective. But this idea is inapplicable to models which learn richer representations. Recent ideas leveraging optimization-based updates scale poorly with the model dimension d, due to inverting the Hessian of the loss function. We use a variant of a new conditional independence coefficient, L-CODEC, to identify a subset of the model parameters with the most semantic overlap on an individual sample level. Our approach completely avoids the need to invert a (possibly) huge matrix. By utilizing a Markov blanket selection common in the literature, we premise that L-CODEC is also suitable for deep unlearning, as well as other applications in vision. Compared to alternatives, L-CODEC makes approximate unlearning possible in settings that would otherwise be infeasible, including vision models used for face recognition, person re-identification and NLP models that may require unlearning data identified for exclusion.

[Full Paper Link at CVPR 2022 Proceedings.](https://openaccess.thecvf.com/content/CVPR2022/html/Mehta_Deep_Unlearning_via_Randomized_Conditionally_Independent_Hessians_CVPR_2022_paper.html)

[Supplementary Material](https://openaccess.thecvf.com/content/CVPR2022/supplemental/Mehta_Deep_Unlearning_via_CVPR_2022_supplemental.zip)

## Code
All experiments are run within the specified folders, and call out to 'codec'.
Navigate to each folder for example scripts and directions on how to run in __expname__/README.md.

#### Conditional Independence Core
For our core conditional independence testing engine, you can check out and use the functions in the `codec/` folder.
We provide MATLAB and Python implementations of the newly proposed measure of [Conditional Dependence (CODEC)](https://www.tandfonline.com/doi/full/10.1080/01621459.2020.1758115) and the associated feature selection scheme [FOCI](https://projecteuclid.org/journals/annals-of-statistics/volume-49/issue-6/A-simple-measure-of-conditional-dependence/10.1214/21-AOS2073.full). Additionally, it also has the implementation of our proposed randomized versions of LCODEC and LFOCI. There is significant gain in computation time when nearest neighbors are computed on GPU as done in our implementations.


#### Deep Learning Pipeline
For the deep learning unlearning pipeline, the `scrub/scrub_tools.py` file contains the main procedure. Our input perturbation revolves around the following at lines 145 and 188-192:
```
for m in range(params.n_perturbations):
	tmpdata = x + (0.1)*torch.randn(x.shape).to(device)
	acts, out = myActs.getActivations(tmpdata.to(device))
	loss = criterion(out, y_true)
	vec_acts = p2v(acts)
```
where the `getActivations` is computed using PyTorch activation hooks defined in `scrub/hypercolumn.py`.

## Reference
If you find our paper helpful and use this code, please cite our [publication](https://openaccess.thecvf.com/content/CVPR2022/html/Mehta_Deep_Unlearning_via_Randomized_Conditionally_Independent_Hessians_CVPR_2022_paper.html) at CVPR 2022. 


```
@InProceedings{Mehta_2022_CVPR,
    author    = {Mehta, Ronak and Pal, Sourav and Singh, Vikas and Ravi, Sathya N.},
    title     = {Deep Unlearning via Randomized Conditionally Independent Hessians},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {10422-10431}
}
```

## TODO
**攻击场景：**

遗忘算法是用于保护用户的“被遗忘权”，即用户有权要求模型方从模型中删除自己提供的数据的影响。

我们的攻击应用于在遗忘算法部署的场景中，模型方向用户收集数据用于训练模型，用户有权提出遗忘申请，从模型中遗忘自己提供的数据。攻击者可以生成虚假用户或者操纵普通用户账户，在训练阶段提供投毒数据用于训练，或者在遗忘阶段修改遗忘申请中的遗忘数据。

重要代码在`scrub`中：
- `train.py`：训练模型，记录各样本损失信息。
- `multi_scrub.py`：遗忘算法，用于从训练好的模型中遗忘数据。
- `plot.ipynb`: 简单的画图函数
- `scrub_scripts`：实验脚本，调参和跑实验

**建议：** 
1. 先看一下源代码遗忘算法的逻辑（[Github Link](https://github.com/vsingh-group/LCODEC-deep-unlearning)），我已经不记得自己改了哪些内容了。看看遗忘算法有哪些参数可以调，调起来有什么效果。

2. 先做mnist_logistic的实验，比较快，先看看有没有攻击效果；然后再做cnn，resnet的实验。之前resnet服务器跑不动，可以优化一下。

3. **遗忘时修改（先做）** ： `multi_scrub.py`中重要参数'unlearning_attack'和'attack_type'：
    - 白盒：知道梯度、参数等信息。挑选损失大的样本遗忘。
    - 假黑盒：可以查询梯度，修改原始样本为对抗样本，然后遗忘
    - 真黑盒：可以设计一个
4. 训练前投毒（有时间再做）：生成一些BSS样本作为投毒数据（或者设计一些新的投毒样本），在模型训练前投毒，然后部署阶段遗忘。可以看看投毒后遗忘随机样本对模型的影响，以及遗忘投毒样本对模型的影响。
    - 随机样本：对比投毒模型和未投毒模型，如果投毒模型遗忘任何样本模型性能下降大于未投毒模型，则攻击成功
    - 投毒样本：对比投毒样本和干净样本，如果遗忘投毒样本比遗忘普通样本对模型性能的影响更大，则攻击成功
