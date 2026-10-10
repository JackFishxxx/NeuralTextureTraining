# NeuralTexture Training Code

This project is used to train Neural Texture (NTC) models. The code is implemented with [PyTorch](https://pytorch.org/get-started/locally/) and [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn), supporting optional quantization-aware training and model export. A pure PyTorch version is also provided, which does not require tiny-cuda-nn.

## What is Neural Texture?

Neural Texture (NTC) is a network-based method for texture representation and compression. Unlike traditional formats such as PNG, JPEG, or ASTC, Neural Texture combines a **feature grid** with a small **MLP (multi-layer perceptron)** to compactly store and reconstruct high-resolution texture information, while still supporting random access to texels. Its advantages include:

- **High compression ratio**: preserves more detail at equal or lower storage cost.
- **Scalability**: supports multi-resolution, Mips, and joint representation with material parameters.
- **Versatility**: applicable not only to albedo maps but also normal maps, roughness maps, and other material textures.

### Motivation and Tiered Deployment

As real-time content increasingly adopts 2K, 4K, and higher-resolution textures, material maps become a major contributor to game package size, patch bandwidth, and runtime memory usage.
Simply increasing the compression ratio of conventional texture formats can introduce detail loss, block artifacts, or platform-specific compatibility constraints. This motivates a tiered representation strategy that balances visual quality, hardware coverage, and deployment cost.

This project keeps 1K-and-below textures in established block-compressed formats such as ASTC for compatibility with existing assets, mobile platforms, and performance-constrained devices.
On higher-end PCs, 2K/4K textures can instead be reconstructed at runtime through neural texture inference rather than stored as complete high-resolution bitmaps. FNTC combines shared texture group features in a feature grid with a low-resolution source mip that is already required by the asset pipeline. 
The mip provides the low-frequency base after bilinear upsampling, while a small MLP predicts the high-frequency residual that bilinear interpolation cannot recover. Because this low-resolution mip is reused from the existing asset chain, the additional FNTC representation cost is concentrated in the feature texture and network parameters, while the traditional ASTC path remains available as a compatibility fallback.

### Differences from NVIDIA NTC SDK

In February 2025, NVIDIA open-sourced the [RTXNTC SDK](https://github.com/NVIDIA-RTX/Rtxntc), providing official support for Neural Texture. However, this project differs from RTXNTC in several ways:

- **NTC SDK**
  - Requires specialized GPU primitives such as **CoopVec** and **CoopMat** to accelerate inference.
  - Requires engine-side integration work for deployment.

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
    --quantize_bits 4 \
    --save_bits 16
```

To resume training from a checkpoint, use `--load_iter` and `--load_dir`:

```bash
python train.py \
    --mode train \
    --data_dir ./data/test \
    --save_dir ./outputs \
    --quantize_bits 4 \
    --save_bits 16 \
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
    --quantize_bits 4 \
    --save_bits 16 \
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
- **PSNR / SSIM / LPIPS**: Image quality metrics per Mip level

Set `mip0_only: true` in `config.yaml` (or pass `--mip0_only`) to restrict training samples and
loss, inference metrics and visualizations, and comparison inputs to the full-resolution Mip0.
The default is `false`, which preserves multi-Mip training and inference.

---

## Key Features

### Multi-Texture Joint Representation (Configurable Canonical Layout)

The model constructs a canonical layout according to the selected normal encoding. The channel order is:

```
diffuse(3) | normal(3 or 2) | roughness(1) | occlusion(1) | metallic(1) | specular(1) | displacement(1)
```

With the default `normal_encoding: xy`, the canonical layout has 10 channels; datasets that omit
specular contain 9 active channels. Missing texture types are zero-filled automatically.

### PSNR-Based Early Stopping

Training supports **automatic PSNR-based early stopping** to avoid wasting time on diminishing returns. When the average PSNR improvement over a segment of `--early_stop_interval` iterations falls below the threshold, the model is saved and training stops.

| Argument | Default | Description |
|----------|:-------:|-------------|
| `--early_stop` | `True` | Enable early stopping; disable with `--no-early_stop` |
| `--early_stop_interval` | `5000` | Iterations per PSNR evaluation segment |
| `--early_stop_psnr_threshold` | `0.01` | Minimum PSNR improvement threshold (dB) |

Two-stage training is enabled by default. Stage one uses an affine decoder without PE or feature differences. Stage two freezes the ASTC bitstream and compressed base, then fits a randomly initialized decoder with the configured architecture.

Warmup cannot stop early; codec activation resets the validation window. A decoder plateau multiplies LR by 0.25 down to 0.0001, then two stagnant windows stop training. Disabling early stopping still adjusts LR and runs the complete budget. Each phase restores the best compressed-validation PSNR state. Final checkpoint iterations denote completed training; `baseline_validation` and `adaptation.validation` in the existing `two_stage_finetune.json` identify the selected weights.

Run `python -m unittest discover -s tests -v` for the behavior checks (native codec tests require CUDA).

Each new validation best is saved atomically to `models/best_<phase>.pth`; standard final exports under `models/train_result_<completed_iteration>/` contain the restored best weights.

### Quantization-Aware Training (QAT)

Training simulates quantization error in the forward pass so the model adapts to inference-time precision loss:

- **Noise Annealing**: Uniform noise is added to feature-grid values and follows a cosine schedule.
  - `--qat_noise_schedule`: `cosine` or `none` (default `cosine`)
  - `--qat_noise_mult_start` / `--qat_noise_mult_end`: `1.0` / `0.25`
  - `--qat_noise_warmup_frac`: `0.1` (the first 10% keeps the starting multiplier)

- **Straight-Through Estimator (STE) Quantization**: The forward pass uses quantized values while gradients pass through unchanged, accurately simulating inference-time rounding and reducing color bias compared to additive noise.

### ASTC-Aware Latent Training

Training implementation is in `Core/ASTC_Aware`.

Backends are `astcenc` and `astc_differentiable_proxy`. The latter uses legal ASTC decoding in forward and surrogate gradients in backward, implemented with CUDA. Only current backend names and the `astc_proxy` checkpoint format are supported.

`astcenc` periodically encodes the currently sampled feature blocks on CPU.
`astc_differentiable_proxy` directly trains legal endpoint/weight parameters on CUDA every step
after warmup, with validated symbol searches and low-frequency material projection. Both use
the existing reconstruction/PBR losses; LPIPS remains an evaluation metric. Backend dependencies
resolve automatically. The default `config.yaml` uses:

```yaml
astc_aware_enable: true
astc_codec_backend: astc_differentiable_proxy
astc_codec_start_frac: 0.1
astc_learning_rate: 0.001
astc_codec_in_loop_interval: 100
astc_codec_update_latent: false
astc_decoder_projection_interval: 500
```

The codec interval and optional latent identity STE apply to the CPU backend. Setting the
projection interval to 0 disables material projection. ASTC comparison uses the existing
`astcenc_path`, `astcenc_quality` and `astc_block` settings.

The proxy requires CUDA-enabled PyTorch, a matching CUDA toolkit, a C++ compiler and ninja.
It reuses `tools/astc_encoder_source/Source`, downloading the pinned encoder source there if absent.
It currently supports quantized Mip0 with four-channel feature grids and direct diffuse disabled.
Material projection additionally requires one grid at the GT resolution and a linear SR decoder
without PE; other configurations should set `astc_decoder_projection_interval: 0`.

Stage two freezes the learned bitstream. Checkpoints are loaded from
`models/train_result_<iteration>/model.pth`; `--load_iter -1` selects the latest completed checkpoint.
The previous two-seed quality figures included a perceptual loss that has since been removed;
they do not establish long-budget quality or speed for the current recipe.

### Feature Gradient Inputs

Feature gradient options are configured in the `feature gradient configs` section:

```yaml
feature_gradient_count: 4  # 0 off | 2 right/down | 4 axial | 8 3x3 | 24 5x5
```

`Core/Feature_Gradient/sampling.py` samples the center and positions one feature
texel away with bilinear repeat addressing, then appends each neighbor-minus-center
difference to the decoder input. The differences do not add feature bitstream bytes,
but they increase decoder input size and sampling cost. Checkpoints and
`network_data.npz` record the direction count; inference must use the same value.

The two-stage entry disables gradient inputs during representation learning and
enables the configured count during decoder adaptation. In single-stage training,
gradients participate in representation learning directly; set
`astc_decoder_projection_interval: 0` because pointwise material projection does
not support spatial difference inputs.

The compact default uses four axial differences and a 1x32 decoder. Larger stencils
require relaxing the compact input limit and are intended for offline experiments.

ASTC training uses the same differentiably decoded texture for every feature-gradient
sample; stage two uses frozen decoded features.

`ASTCAwareTrainer.refine_gradient_codec` is an explicit fixed-decoder refinement
pass that validates legal codec proposals over complete block neighborhoods. It
is not automatically added to the default training flow.

### Seamless Tiling (Wrap Boundary Constraint)

The feature grid supports a **wrap boundary constraint**: during training, left/right and top/bottom border features (as well as corners) are softly tied together, preventing visible seams when the feature texture is sampled in Wrap/Repeat mode at runtime.

### Feature Grid DDS Export

When saving a model, feature grids are exported as **DDS files (R8G8B8A8, non-SRGB)** for direct loading in UE5 and other game engines. Network weights are saved as `.npz` files.

### Heterogeneous Multi-Feature Grid

The model supports **heterogeneous feature grids** through the `feature_grid_configs` config field:

| Field | Description |
|-------|-------------|
| `max_resolution` | Maximum resolution of the grid |
| `n_levels` | Number of Mip levels |
| `quantize_bits` | Quantization precision (2 / 4 / 8 bits) |
| `save_bits` | Feature bits per texel (8 / 16 / 32 bits; at most 4 channels) |
| `learning_rate` | Per-grid learning rate |

### Configurable Per-Texture Loss Weights

Independent loss weights can be assigned to each texture type via the `texture_loss_weights` field in `configs.py`, balancing the contribution of different channels during training (e.g., diffuse typically weighted higher than displacement).

### Configurable Network Architecture

```yaml
n_neurons: 32
n_hidden_layers: 0
output_activation: hard_swish
```

`n_hidden_layers: 0` selects a direct mapping without hidden layers. These values can also be
overridden with `--n_neurons`, `--n_hidden_layers`, and `--output_activation`.

The bundled tiny-cuda-nn CutlassMLP ignores `use_bias`. FNTC appends a constant
input for first-layer bias and stores an explicit `decoder_output_bias` parameter
for the output layer.

### Neural Texture Super-Resolution

Base upsampling and continuous queries in both stages use texel-centered bilinear repeat addressing, including opposite-edge texels when interpolating across a seam. ASTC comparisons and material projection follow the same convention. The existing upsampled base cache, decoder input size and texture query counts are retained.

Set `super_resolution_enable: true` and configure `super_resolution_base_resolution` to train
residual super-resolution. The source mip at that resolution is bilinearly upsampled and provided
alongside the feature-grid samples; the network predicts `GT - bilinear_base`. Training, inference,
and ASTC comparison reconstruct `clamp(bilinear_base + predicted_residual, 0, 1)`. The same mip-level
offset is used for lower Mips. Enabling this mode automatically disables direct-diffuse inference,
whose direct-output semantics conflict with residual prediction.
