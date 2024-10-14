# Depth Attention ![Static Badge](https://img.shields.io/badge/Achieving_New_Sota_Without_Finetuning-ACCV2024_Oral-blue)

The official implementation for the ACCV 2024 paper \[[_Depth Attention for Robust RGB Tracking_](www.google.com)\]
![depth_attention (1)-1](https://github.com/user-attachments/assets/96159032-faf5-4305-ba91-ac0c5a3a6b75)

## News

:fire::fire::fire:

2024-10 :tada:Our new challenging dataset [NT-VOT211](https://github.com/LiuYuML/NV-VOT211) is available now! <sub>Click the link on the right :rewind: to access our full tutorial for benchmarking on this new dataset.</sub>

:fire::fire::fire:


## Qualitative Results
![DepthEstimationOnDiagram](https://github.com/user-attachments/assets/eec7afd1-6f11-4025-9f87-369988c6e3fa)

## Demo
![alt text](Demo.gif)

# Set up our algorithm
## Environment Setup
To ensure the replication of the precise results detailed in the paper, it is crucial to match the software environment closely. Please configure your system with the following specifications:

- **Python**: Version 3.8.10
- **PyTorch**: Version 1.11.0, built with CUDA 11.3 support
- **CUDA**: Version 11.3
- **NumPy**: Version 1.22.3
- **OpenCV**: Version 4.8.0

By adhering to these versions, you will be able to achieve consistency with the experimental setup described in the publication.

## Pretrained Model Download

To ensure the accuracy and consistency of the results as reported in our paper, it is essential to use the pretrained depth estimators that have been tested and validated with our algorithm. We have found the following models to be compatible and effective:

1. **Lite-Mono**: This is the primary pretrained model we have used in our research. You can download it from the [Lite-Mono](https://github.com/noahzn/Lite-Mono) GitHub repository.
2. **FastDepth**: In addition to Lite-Mono, we have also confirmed that the [FastDepth](https://github.com/dwofk/fast-depth) model can be used with our algorithm.
3. **Monodepth2**: Another option that has been tested is the [Monodepth2](https://github.com/nianticlabs/monodepth2) model.

We recommend starting with the Lite-Mono model, as it has been extensively used in our experiments.

## Setting Up the Trackers

In our quest to create a comprehensive tracking solution, we have meticulously chosen a diverse array of baseline trackers, each with its own unique strengths:

- **RTS**: Engineered for rapid tracking, this system excels in real-time scenarios. [Dive deeper](https://github.com/visionml/pytracking/blob/master/pytracking/README.md#RTS) 
- **AiATrack**: A cutting-edge tracker that harnesses the power of artificial intelligence. [Discover more](https://github.com/Little-Podi/AiATrack) 
- **ARTrack**: Optimized for augmented reality, this tracker is a leader in its field. [Find out more](https://github.com/MIV-XJTU/ARTrack) 
- **KeepTrack**: Renowned for its steadfast reliability and precision across a spectrum of conditions. [Get the details](https://github.com/visionml/pytracking/blob/master/pytracking/README.md#KeepTrack) 
- **MixFormer**: A versatile tracker that adapts to various tracking challenges. [Check it out](https://github.com/MCG-NJU/MixFormer) 
- **Neighbor**: This tracker focuses on proximity-based tracking for enhanced accuracy. [Explore here](https://github.com/franktpmvu/NeighborTrack) 
- **ODTrack**: Designed for object detection and tracking in complex environments. [Learn about it](https://github.com/GXNU-ZhongLab/ODTrack) 
- **STMTrack**: A tracker that offers a seamless tracking experience. [Read more](https://github.com/fzh0917/STMTrack) 

Together, these trackers form a powerful toolkit, adept at handling a wide range of tracking tasks across diverse settings and scenarios.

To set up these trackers, please refer to the comprehensive [tutorial](https://github.com/LiuYuML/Depth-Attention/tree/main/Trackers). 

# Citing Us
If you find our work valuable, we kindly ask you to consider citing our paper and starring ⭐ our repository. Our implementation includes mutiple trackers and we hope it make life easier for the VOT research community and Depth Estimation community.

# Maintenance

Please open a GitHub issue for any help. If you have any questions regarding the technical details, feel free to contact us. 

# License
[MIT License](https://mit-license.org/) 

