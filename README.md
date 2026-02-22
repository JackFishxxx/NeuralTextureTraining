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
  - Provides **[full UE5 plugin integration - branch 5.5.1-FNTC](https://github.com/JackFishxxx/UnrealEngine/tree/5.5.1-FNTC) **: Neural Textures can be imported as custom resources (feature grids as R8G8B8A8 DDS textures and network parameters in `network_data.npz`) and directly sampled in the material system.  
  - Supports the entire pipeline of training, quantization, and export, suitable for both research and production use.  

---

## Installation

This project is implemented in Python. Please install Python first (we recommend using [Anaconda](https://www.anaconda.com/download/success) for environment setup). Refer to existing tutorials online if necessary.

### Dependencies

- Python 3.9+
- PyTorch (matching your local CUDA version)
- tiny-cuda-nn (optional but strongly recommended)
- Others: `numpy`, `Pillow`, `torchvision`, `torchmetrics`, `torchtyping`, `tensorboardX`

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
```

Install [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn):

```bash
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

Install remaining dependencies:

```bash
pip install numpy Pillow torchvision torchmetrics torchtyping tensorboardX
```

---

## Data

We provide sample data in the `data` folder. You may also use your own data.

The expected directory structure is:

```
data
└── CustomTexture
    ├── CustomTexture_Albedo.png
    ├── CustomTexture_Normal.png
    ├── CustomTexture_Roughness.png
    ├── CustomTexture_AO.png
    ├── CustomTexture_Metallic.png
    ├── CustomTexture_Specular.png
    ├── CustomTexture_Displacement.png
    └── ...
```

Place your textures in the corresponding folder. The code **automatically identifies** texture types from filename keywords. Supported texture types and their recognized keywords are:

| Texture Type | Channels | Keywords |
|-------------|:--------:|---------|
| diffuse (albedo/color) | 3 | `diffuse`, `albedo`, `color`, `diff` |
| normal | 3 | `normal`, `nor_gl` |
| roughness | 1 | `roughness`, `rough` |
| occlusion (AO) | 1 | `occlusion`, `ao`, `ambient` |
| metallic | 1 | `metallic`, `metalness` |
| specular | 1 | `specular` |
| displacement | 1 | `displacement`, `disp` |

> **Note**: The model output is fixed at **11 channels** (diffuse×3 + normal×3 + roughness×1 + occlusion×1 + metallic×1 + specular×1 + displacement×1). Channels for missing texture types are automatically filled with zeros — you do not need to provide all types. Textures with mismatched resolutions are automatically resized to match the first loaded texture.

For detailed naming conventions, keyword matching, and channel counts, refer to `get_texture_config()` in `dataset.py`.

---

## Running

All configurable variables and their descriptions can be found in `configs.py`.

### Training

Run training with the following command:

```bash
python train.py \
    --mode train \
    --data_dir ./data/test \
    --save_dir ./outputs \
    --quantize True \
    --quantize_bits 4 \
    --save_bits 32
```

To resume training from a checkpoint, use `--load_iter` and `--load_dir`:

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

### Batch Training (GroupsBatching Mode)

To train multiple texture groups in batch, use the `--groups_batching` mode. It automatically discovers all subdirectories under `--groups_batch_dir`, trains each group sequentially, and saves results to corresponding subdirectories under `--groups_save_dir`.

Example directory structure:

```
data_batch
├── TextureGroupA
│   ├── TextureGroupA_Albedo.png
│   └── TextureGroupA_Normal.png
├── TextureGroupB
│   ├── TextureGroupB_Albedo.png
│   └── ...
└── ...
```

Run command:

```bash
python train.py \
    --groups_batching \
    --groups_batch_dir ./data_batch \
    --groups_save_dir ./outputs_batch
```

Available GroupsBatching parameters:

| Argument | Default | Description |
|----------|:-------:|-------------|
| `--groups_batching` | `False` | Enable GroupsBatching mode |
| `--groups_batch_dir` | `data_batch` | Root directory containing texture group subdirectories |
| `--groups_save_dir` | `save_batch` | Root directory for saving per-group results |
| `--groups_max_workers` | `1` | Number of concurrent training workers (1 = sequential) |
| `--groups_verbose` | `False` | Print per-group training details |

### Evaluation

During training, evaluation metrics are automatically logged. To run inference on a previously trained model:

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

### VS Code Launch Configuration

A `launch.json` file for VS Code is provided for convenient debugging and running.

---

## Monitoring

### TensorBoard

Launch TensorBoard to monitor convergence and model quality:

```bash
tensorboard --logdir ./outputs/tensorboard
```

The following metrics are logged automatically during training:

- **Loss/train**: Training loss
- **PSNR / SSIM / LPIPS**: Image quality metrics per LOD level

---

## Key Features

### Multi-Texture Joint Representation (11-Channel Canonical Layout)

The model uses a fixed **11-channel canonical layout** that supports joint representation of multiple material textures. The channel order is:

```
diffuse(3) | normal(3) | roughness(1) | occlusion(1) | metallic(1) | specular(1) | displacement(1)
```

If only a subset of texture types is provided (e.g., only albedo + normal), the missing channels are automatically zero-filled without affecting training or inference.

### PSNR-Based Early Stopping

Training supports **automatic PSNR-based early stopping** to avoid wasting time on diminishing returns. When the average PSNR improvement over a segment of `--early_stop_interval` iterations falls below the threshold, the model is saved and training stops.

| Argument | Default | Description |
|----------|:-------:|-------------|
| `--early_stop` | `True` | Enable early stopping |
| `--early_stop_interval` | `5000` | Iterations per PSNR evaluation segment |
| `--early_stop_psnr_threshold` | `0.01` | Minimum PSNR improvement threshold (dB) |

### Quantization-Aware Training (QAT)

Training simulates quantization error in the forward pass so the model adapts to inference-time precision loss:

- **Noise Annealing**: Uniform noise is added to feature values at the start of training and linearly decayed to zero, improving robustness to quantization error.
  - `--noise_std`: Noise strength (default `1.0`)
  - `--noise_anneal_fraction`: Fraction of training over which noise decays (default `0.8`, i.e., noise is fully annealed by 80% of training)

- **Straight-Through Estimator (STE) Quantization**: The forward pass uses quantized values while gradients pass through unchanged, accurately simulating inference-time rounding and reducing color bias compared to additive noise.

### Seamless Tiling (Wrap Boundary Constraint)

The feature grid supports a **wrap boundary constraint**: during training, left/right and top/bottom border features (as well as corners) are softly tied together, preventing visible seams when the feature texture is sampled in Wrap/Repeat mode at runtime.

### Feature Grid DDS Export

When saving a model, feature grids are exported as **DDS files (R8G8B8A8, non-SRGB)** for direct loading in UE5 and other game engines. Network weights are saved as `.npz` files.

### Heterogeneous Multi-Hash Grid

The model supports **heterogeneous multi-hash grids**, where each grid can be independently configured in the `hash_grid_configs` field of `configs.py`:

| Field | Description |
|-------|-------------|
| `max_resolution` | Maximum resolution of the grid |
| `n_levels` | Number of LOD levels |
| `quantize_bits` | Quantization precision (2 / 4 / 8 / 16 bits) |
| `save_bits` | Save precision (8 / 16 / 32 / 64 bits) |
| `learning_rate` | Per-grid learning rate |

### Configurable Per-Texture Loss Weights

Independent loss weights can be assigned to each texture type via the `texture_loss_weights` field in `configs.py`, balancing the contribution of different channels during training (e.g., diffuse typically weighted higher than displacement).