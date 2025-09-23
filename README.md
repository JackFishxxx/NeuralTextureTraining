# NeuralTexture Training Code

This project is used to train Neural Texture (NTC) models. The code is implemented with [PyTorch](https://pytorch.org/get-started/locally/) and [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn), supporting optional quantization-aware training and model export. A pure PyTorch version is also provided, which does not require tiny-cuda-nn.

## What is Neural Texture?

Neural Texture (NTC) is a network-based method for texture representation and compression. Unlike traditional formats such as PNG, JPEG, or ASTC, Neural Texture combines a **feature grid** with a small **MLP (multi-layer perceptron)** to compactly store and reconstruct high-resolution texture information, while still supporting random access to texels. Its advantages include:

- **High compression ratio**: preserves more detail at equal or lower storage cost.  
- **Scalability**: supports multi-resolution, LODs, and joint representation with material parameters.  
- **Versatility**: applicable not only to albedo maps but also normal maps, roughness maps, and other material textures.  

### Differences from NVIDIA NTC SDK

In February 2025, NVIDIA open-sourced the [RTXNTC SDK](https://github.com/NVIDIA-RTX/Rtxntc), providing official support for Neural Texture. However, this project differs from RTXNTC in several ways:

- **NTC SDK**  
  - Requires specialized GPU primitives such as **CoopVec** and **CoopMat** to accelerate inference.  
  - Still faces compatibility issues in game engine integration.  

- **This project**  
  - Does **not** rely on CoopVec or CoopMat. Instead, it is implemented directly with PyTorch and tiny-cuda-nn, making it simpler and easier to understand.  
  - Provides **full UE5 plugin integration**: Neural Textures can be imported as custom resources (similar to PNG/JPG) and directly sampled in the material system.  
  - Supports the entire pipeline of training, quantization, and export, suitable for both research and production use.  

---

## Installation

This project is implemented in Python. Please install Python first (we recommend using [Anaconda](https://www.anaconda.com/download/success) for environment setup). Refer to existing tutorials online if necessary.

### Dependencies

- Python 3.9+
- PyTorch (matching your local CUDA version)
- tiny-cuda-nn (optional but strongly recommended)
- Others: `numpy`, `tensorboard`

### Hardware Requirements

Neural Texture training and inference itself does not strictly require special hardware, but tiny-cuda-nn is used for acceleration, which requires an NVIDIA GPU. Recommended requirements:

- An NVIDIA GPU (Tensor Cores improve performance if available).  
- A C++14-capable compiler. Recommended and tested:  
  - **Windows**: Visual Studio 2019 or 2022  
  - **Linux**: GCC/G++ 8 or higher  
- A recent CUDA version. Recommended and tested:  
  - **Windows**: CUDA 11.5 or higher  
  - **Linux**: CUDA 10.2 or higher  
- CMake v3.21 or higher.  

Additionally, this project can theoretically be run on Linux for large-scale offline texture compression. The resulting compressed Neural Texture files can then be imported into UE5 for use.

### Environment Setup

Clone the project:

```bash
git clone https://github.com/JackFishxxx/NeuralTextureTraining.git
cd NeuralTextureTraining