# NeuralTexture Training Code

This project is used to train Neural Texture (NTC) models. The code is implemented with [PyTorch](https://pytorch.org/get-started/locally/) and [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn), supporting optional quantization-aware training and model export. A pure PyTorch version is also provided, which does not require tiny-cuda-nn.

# What is Neural Texture?

Neural Texture (NTC) is a network-based method for texture representation and compression. Unlike traditional formats such as PNG, JPEG, or ASTC, Neural Texture combines a **feature grid** with a small **MLP (multi-layer perceptron)** to compactly store and reconstruct high-resolution texture information, while still supporting random access to texels. Its advantages include:

- **High compression ratio**: preserves more detail at equal or lower storage cost.  
- **Scalability**: supports multi-resolution, LODs, and joint representation with material parameters.  
- **Versatility**: applicable not only to albedo maps but also normal maps, roughness maps, and other material textures.  

## Differences from NVIDIA NTC SDK

In February 2025, NVIDIA open-sourced the [RTXNTC SDK](https://github.com/NVIDIA-RTX/Rtxntc), providing official support for Neural Texture. However, this project differs from RTXNTC in several ways:

- **NTC SDK**  
  - Requires specialized GPU primitives such as **CoopVec** and **CoopMat** to accelerate inference.  
  - Still faces compatibility issues in game engine integration.  

- **This project**  
  - Does **not** rely on CoopVec or CoopMat. Instead, it is implemented directly with PyTorch and tiny-cuda-nn, making it simpler and easier to understand.  
  - Provides **full UE5 plugin integration**: Neural Textures can be imported as custom resources (similar to PNG/JPG) and directly sampled in the material system.  
  - Supports the entire pipeline of training, quantization, and export, suitable for both research and production use.

# UE Plugin Integration

We provide a full Unreal Engine plugin integration for NeuralTexture, tested on Unreal Engine 5.5 (other engine versions may require minor adjustments). With the plugin, Neural Textures can be imported as custom assets (similar to PNG/JPG) and sampled directly in the Material system. The runtime inference is executed fully in shaders (no CUDA / PyTorch required).

## Full UE Project Source (Access Required)

For a complete Unreal Engine project with full plugin support, please refer to [JackFishxxx/UnrealEngine](https://github.com/JackFishxxx/UnrealEngine/tree/5.5.1-ntc). Note that this repository is forked from EpicGames' Unreal Engine source repository. Therefore, it inherits EpicGames access restrictions. You must have Unreal Engine GitHub access (i.e., your GitHub account has joined the EpicGames organization) before you can view or clone it.

If you do not have access yet, follow [EpicGames' official guide](https://www.unrealengine.com/en-US/ue-on-github) or [Doc of downloading source code](https://dev.epicgames.com/documentation/en-us/unreal-engine/downloading-source-code-in-unreal-engine) to link your Epic Games account with GitHub and accept the organization invitation.

# Pipeline

Here we visualize the whole training pipeline in the image below.

<div align="center">
  <img src="assets/training.png" alt="Neural Texture Training Pipeline" width="100%" />
</div>

During saving, we would first quantize the Feature Grid and packed them in uint32 format, then we add the texture info and the network data to save the whole texture, as illustrated below.

<div align="center">
  <img src="assets/saving.png" alt="Neural Texture Saving" width="75%" />
</div>

At runtime, the neural texture could be used just like common PNG or TGA images. The whole texture is able to be inferenced all by shader without any CUDA or Pytorch support. We also provide an [UnrealEngine 5 plugin](https://github.com/JackFishxxx/UnrealEngine/tree/5.5.1-ntc) for runtime usage. The runtime pipeline is shown as below.

<div align="center">
  <img src="assets/runtime.png" alt="UE5 Runtime Pipeline" width="100%" />
</div>


# Installation

This project is implemented in Python. Please install Python first (we recommend using [Anaconda](https://www.anaconda.com/download/success) for environment setup). Refer to existing tutorials online if necessary.

## Dependencies

- Python 3.9+
- PyTorch (matching your local CUDA version)
- tiny-cuda-nn (optional but strongly recommended)
- Others: `numpy`, `tensorboard`

## Hardware Requirements

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

## Environment Setup

1. First, Clone the project:

```bash
git clone https://github.com/JackFishxxx/NeuralTextureTraining.git
cd NeuralTextureTraining
```

2. Then install [PyTorch](https://pytorch.org/get-started/locally/).

3. Install [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) as well.

```bash
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

4. Finally, install the remaining dependencies:

```bash
pip install -r requirements.txt
```


# Usage

## Data

We provide an example dataset in the data folder. Alternatively, you can use your own data.

The expected format is:

```
data
└── CustomTexture
    ├── CustomTexture_Albedo.png
    ├── CustomTexture_Normal.png
    ├── ...
```

Place your texture maps in the corresponding folder, and append the proper suffix to each map according to the naming convention. The code will automatically detect which texture maps are present. For the exact naming rules, detection logic, and the number of channels, please refer to config.py.

## Training

For detailed configuration options, see the variables and comments in config.py.

To start training, run:

```bash
python train.py \
    --mode train \
    --data_dir ./data/test \
    --save_dir ./outputs \
    --quantize True \
    --quantize_bits 4 \
    --save_bits 32
```

To resume training from a previously saved checkpoint, use `--load_iter` and `--load_dir`:

```bash
python train.py \
    --mode train \
    --data_dir ./data/test \
    --save_dir ./outputs \
    --quantize True \
    --quantize_bits 4 \
    --save_bits 32 \
    --load_iter 50000 \
    --load_dir outputs/yyyy-mm-dd-hh-mm-ss
```

## Evaluation

During training, some evaluation metrics will be logged automatically. If you want to evaluate a trained model (or run inference for evaluation), use:

```bash
python train.py \
    --mode infer \
    --data_dir ./data/test \
    --save_dir ./outputs \
    --quantize True \
    --quantize_bits 4 \
    --save_bits 32 \
    --load_iter 50000 \
    --load_dir outputs/yyyy-mm-dd-hh-mm-ss
```

## Training via launch.json

This project provides a VS Code `launch.json` for convenience.


# Monitoring

The project includes runtime metric logging and monitoring utilities.

## TensorBoard

To monitor logs, convergence, and model quality:

```
tensorboard --logdir ./outputs/tensorboard
```
