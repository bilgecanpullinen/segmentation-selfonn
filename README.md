# Convolutional versus Operational Neural Networks for Semantic Segmentation of Maritime and Harbor Scenes
<p align="center">
<img width="541" alt="example" src="https://github.com/bilgecanpullinen/Convolutional-versus-Operational-Neural-Networks-for-Semantic-Segmentation-of-Maritime-and-Harbor-Sc/assets/36228505/30e8dc5d-f364-44be-af92-887d1b8e074f">
</p>
This repository is the official implementation of the [Convolutional versus Operational Neural Networks for Semantic Segmentation of Maritime and Harbor Scenes](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4476469) paper. This repository is based on [semantic-segmentation-pytorch](https://github.com/CSAILVision/semantic-segmentation-pytorch), in which you can use the instructions for training and evaluation steps.

## Abstract

Semantic segmentation aims to associate each pixel of an image with a corresponding label that describes what is being depicted. The key aspect of semantic segmentation emanates from incorporating the classification of objects and the recognition of their shape, which is paramount for the autonomous transportation industry, e.g., maritime autonomous ships and self-driving vehicles. Various semantic segmentation solutions are based on standard Convolutional Neural Networks (CNNs). Nevertheless, recent evidence suggests that self-organized operational neural networks (Self-ONNs) can yield better performance because of their increased heterogeneity and learning capacity. This paper presents a novel network approach by combining convolutional layers with operational layers for segmenting objects in a maritime/urban environment. Overall, the findings show that the operational layers are compatible with convolutional layers for semantic segmentation tasks. While the ResNet-18 model with convolutional layers achieved 19.8%, our model SelfONN-18 2 achieved 25.4% in Mean intersection-over-union (Mean IoU) over the former validation set of ADE20K Dataset. Even with a single layer of operational layer, we achieved better results. While PSPNet-18 with convolutional layers achieved 21.3%, our model SelfONNet-18 with a single operational layer achieved 32.5% in Mean IoU over the former validation set of ADE20K Dataset.

**Keywords: Semantic segmentation, Operational neural networks, Self-ONNs, Generative neurons, Machine learning, Maritime scenes**

The optimized PyTorch implementation of [Self-ONNs](http://selfonn.net/) is publically shared.

## Tables
Details of the models are shown in Table 1-2. Please refer to the paper for the explanation.
<p align="center">
<img width="278" alt="table1" src="https://github.com/bilgecanpullinen/Convolutional-versus-Operational-Neural-Networks-for-Semantic-Segmentation-of-Maritime-and-Harbor-Sc/assets/36228505/d8e9f4d8-258d-4788-95c8-cbb1b30e2dfe">

<img width="284" alt="table2" src="https://github.com/bilgecanpullinen/Convolutional-versus-Operational-Neural-Networks-for-Semantic-Segmentation-of-Maritime-and-Harbor-Sc/assets/36228505/f3ab6bc6-0411-4466-863e-734364b59fac">
</p>

Evaluation of models in [ADE20K Dataset](https://groups.csail.mit.edu/vision/datasets/ADE20K/).
<p align="center">
<img width="305" alt="table3" src="https://github.com/bilgecanpullinen/Convolutional-versus-Operational-Neural-Networks-for-Semantic-Segmentation-of-Maritime-and-Harbor-Sc/assets/36228505/68c6c394-6911-4cef-bf7d-7eb1cf3a23a2">
<img width="302" alt="table4" src="https://github.com/bilgecanpullinen/Convolutional-versus-Operational-Neural-Networks-for-Semantic-Segmentation-of-Maritime-and-Harbor-Sc/assets/36228505/18c6f593-8562-4c40-b93d-59d5515397c9">
</p>

