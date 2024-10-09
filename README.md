# Depth Attention
The official implementation for the ACCV 2024 paper \[[_Depth Attention for Robust RGB Tracking_](www.google.com)\]
![depth_attention (1)-1](https://github.com/user-attachments/assets/96159032-faf5-4305-ba91-ac0c5a3a6b75)

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

