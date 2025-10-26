# A Unified Masked Jigsaw Puzzle Framework for Vision and Language Models, TPAMI2025

Weixin Ye<sup>1</sup>, Wei Wang<sup>1</sup>†, Yahui Liu<sup>2</sup>, Yue Song<sup>3</sup>, Bin Ren<sup>4</sup>, Wei Bi<sup>5</sup>, Rita Cucchiara<sup>6</sup>, Nicu Sebe<sup>4</sup>

**†: Corresponding Author** <br>
<sup>1</sup>Beijing Jiaotong University, Beijing, China  
<sup>2</sup>Kuaishou, Beijing, China  
<sup>3</sup>Caltech, USA  
<sup>4</sup>University of Trento, Italy  
<sup>5</sup>Hong Kong University of Science and Technology, Hong Kong, China  
<sup>6</sup>University of Modena and Reggio Emilia, Modena, Italy

<img width="2465" height="1106" alt="image" src="https://github.com/user-attachments/assets/cf3fcb59-5315-4200-bf31-c6d25f6fccac" />
The main idea of random jigsaw shuffle algorithm and the overview the proposed MJP for image and text data.
The repository offers the official implementation of our paper in PyTorch.


🔍 **News (Oct 5, 2025)!** Our paper is accepted by TPAMI2025!

## Abstract
In federated learning, Transformer, as a popular architecture, faces critical challenges in defending against gradient attacks and improving model performance in both Computer Vision (CV) and Natural Language Processing (NLP) tasks. It has been revealed that the gradient of Position Embeddings (PEs) in Transformer contains sufficient information, which can be used to reconstruct the input data. To mitigate this issue, we introduce a Masked Jigsaw Puzzle (MJP) framework. MJP starts with random token shuffling to break the token order, and then a learnable unknown (unk) position embedding is used to mask out the PEs of the shuffled tokens. In this manner, the local spatial information which is encoded in the position embeddings is disrupted, and the models are forced to learn feature representations that are less reliant on the local spatial information. Notably, with the careful use of MJP, we can not only improve models' robustness against gradient attacks, but also boost their performance in both vision and text application scenarios, such as classification for images (e.g., ImageNet-1K) and sentiment analysis for text (e.g., Yelp and Amazon).
Experimental results suggest that MJP is a unified framework for different Transformer-based models in both vision and language tasks.

## 中文简介
**一种基于掩码拼图的视觉-语言统一框架**
在联邦学习中，Transformer作为一种流行的架构，在防御梯度攻击和提高计算机视觉与自然语言处理任务的模型性能方面面临着重大挑战。研究表明，Transformer中的位置嵌入/编码（Position Embeddings，PEs）的梯度包含了足够的信息，这些信息可以被用来重建输入数据。为了解决这一问题，我们提出了一种掩码拼图MJP（Masked Jigsaw Puzzle）框架。MJP通过随机打乱输入Token序列来破坏Token的顺序，然后使用可学习的“未知（unknown，unk）”PEs来掩盖打乱Token的PEs。通过这种方式，PEs中编码的局部空间信息被打乱，模型被迫学习不依赖于局部空间信息的特征表示。值得注意的是，通过合理地应用 MJP，不仅可以提升模型对梯度攻击的鲁棒性，还能在图像分类（如 ImageNet-1K）和文本情感分析（如 Yelp 和 Amazon）等视觉与文本任务中提高模型性能。实验结果表明，MJP是一个统一的框架，可以在视觉和语言任务中应用于不同的基于Transformer的模型。

## Checkpoints
You can find our pretrained checkpoints.

## Datasets
Prepare Yelp and Amazon dataset (.csv files)

## Training
```
sh train_ngram.sh
```

## Gradient Inversion Attack
```
sh attack.sh
```

## Citation
```
@ARTICLE{11202736,
  author={Ye, Weixin and Wang, Wei and Liu, Yahui and Song, Yue and Ren, Bin and Bi, Wei and Cucchiara, Rita and Sebe, Nicu},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={A Unified Masked Jigsaw Puzzle Framework for Vision and Language Models}, 
  year={2025},
  volume={},
  number={},
  pages={1-17},
  keywords={Transformers;Privacy;Natural language processing;Principal component analysis;Computer vision;Computational modeling;Training;Data privacy;Three-dimensional displays;Image reconstruction;Masked Jigsaw Puzzle;Natural Language Processing;Computer Vision;Gradient Inversion;Position Embedding},
  doi={10.1109/TPAMI.2025.3621246}}
```
