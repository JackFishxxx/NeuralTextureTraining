# NeuralTexture Training Code

This project trains FNTC neural texture models with [PyTorch](https://pytorch.org/get-started/locally/) and [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn), including quantization-aware training and model export. The current FNTC entry requires tiny-cuda-nn and an NVIDIA GPU; this checkout does not include a standalone CPU training implementation.

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

## Core Modules

`model.py` assembles the feature and encoding modules with the decoder; `train.py` coordinates training with the ASTC-aware and multi-stage modules:

| Directory | Responsibility |
|---|---|
| `Core/Feature_Grid/` | Grid layout, quantization/QAT noise, boundary constraints, shared sampling and feature gradients |
| `Core/Positional_Encoding/` | Periodic UV sin/cos encoding |
| `Core/Material_Encoding/` | Normal encoding, decoding and angular errors |
| `Core/ASTC_Aware/` | Compression-aware training, latent ASTC round-trips and differentiable ASTC |
| `Core/Multi_Stage_Training/` | Representation training, decoder fitting and layout refitting |

## Installation

This project is implemented in Python. Please install Python first (we recommend using [Anaconda](https://www.anaconda.com/download/success) for environment setup). Refer to existing tutorials online if necessary.

### Dependencies

- Python 3.9+
- PyTorch (matching your local CUDA version)
- tiny-cuda-nn (required by the current FNTC model)
- Others: `numpy`, `Pillow`, `torchvision`, `torchmetrics`, `torchtyping`, `tensorboardX`, `PyYAML`, `ninja`

### Hardware Requirements

The current FNTC training/inference entry uses an NVIDIA GPU, CUDA-enabled PyTorch and tiny-cuda-nn. The default differentiable ASTC backend also needs a CUDA toolkit compatible with the installed PyTorch, a supported C++ compiler and ninja. On Windows, install the Visual Studio C++ build tools; on Linux, use a compiler supported by your PyTorch/CUDA versions. Building tiny-cuda-nn from source also requires CMake.

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
pip install numpy Pillow torchvision torchmetrics torchtyping tensorboardX PyYAML ninja
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
| normal | 2 (`xy`/`hemi_oct`) or 3 (`xyz`) | `normal`, `nor_gl` |
| roughness | 1 | `roughness`, `rough` |
| occlusion (AO) | 1 | `occlusion`, `ao`, `ambient` |
| metallic | 1 | `metallic`, `metalness` |
| specular | 1 | `specular` |
| displacement | 1 | `displacement`, `disp`, `height` |

> **Note**: The model has 10 canonical output channels with `xy`/`hemi_oct` normals, or 11 with `xyz`. Missing material slots are zero-filled in targets and excluded from losses/metrics. Mismatched texture resolutions follow the first available type in canonical order (diffuse, normal, R/O/M/S/D).

For detailed naming conventions, keyword matching, and channel counts, refer to `get_texture_config()` in `dataset.py`.

---

## Running

Run commands from the project root. `configs.py` automatically loads `config.yaml`; `--config path/to/recipe.yaml` selects another YAML, and CLI arguments override its values. The default recipe enables Mip0, XY normals, residual SR, four feature gradients and multi-stage training. Per-grid entries in `feature_grid_configs` specify their own quantization/export precision.

### Training

Run training with the following command:

```bash
python train.py \
    --config config.yaml \
    --mode train \
    --data_dir ./data/test \
    --save_dir ./outputs
```

To start single-stage training from a completed run's final checkpoint, reuse the YAML/CLI architecture that produced it and disable material projection. This loads model weights and starts new optimizer/scheduler state; it does not resume the multi-stage workflow or restore optimizer/RNG state:

```bash
python train.py \
    --config config.yaml \
    --mode train \
    --data_dir ./data/test \
    --save_dir ./outputs \
    --no-multi_stage_training_enable \
    --astc_decoder_projection_interval 0 \
    --load_iter -1 \
    --load_dir outputs/yy_mm_dd_HH_MM_SS
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
| `--groups_max_workers` | `1` | Reserved; currently only sequential execution (`1`) is supported |
| `--groups_verbose` | `False` | Reserved option; the current entry prints per-group logs directly |

### Evaluation

ROMSD LPIPS is computed independently for each active scalar channel, then averaged.

During training, evaluation metrics are automatically logged. For inference, reuse the architecture and material settings of the selected checkpoint; `--load_dir` is the run directory containing `models/`, and `--load_iter -1` selects its latest completed checkpoint:

```bash
python train.py \
    --config config.yaml \
    --mode infer \
    --data_dir ./data/test \
    --save_dir ./outputs \
    --load_iter -1 \
    --load_dir outputs/yy_mm_dd_HH_MM_SS
```

---

## Monitoring

### TensorBoard

Each run writes events under `<save_dir>/<timestamp>/tensorboard/`. Monitor all runs under the output root:

```bash
tensorboard --logdir ./outputs
```

The following metrics are logged automatically during training:

- **Loss/train**: Training loss
- **PSNR / SSIM / LPIPS**: Image quality metrics per Mip level

Set `mip0_only: true` in `config.yaml` (or pass `--mip0_only`) to restrict training samples and
loss, inference metrics and visualizations, and comparison inputs to the full-resolution Mip0.
The bundled `config.yaml` sets this to `true`. Multi-Mip training requires `--no-mip0_only --no-multi_stage_training_enable --no-astc_aware_enable`; native reconstruction below GT resolution additionally requires Mip0.

---

## Key Features

### Multi-Texture Joint Representation (Configurable Canonical Layout)

The model constructs a canonical layout according to the selected normal encoding. The channel order is:

```
diffuse(3) | normal(3 or 2) | roughness(1) | occlusion(1) | metallic(1) | specular(1) | displacement(1)
```

With the default `normal_encoding: xy`, the canonical layout has 10 channels; datasets that omit
specular contain 9 active channels. Missing texture types are zero-filled automatically.

The dataset defines `texture_packing` once for diffuse, normal and one RGBA scalar texture (R/O/M/S/D order, unused components set to one); when all five scalars exist, displacement is excluded from data, losses and metrics. ASTC references and the compressed SR base execute this same mapping, exported with canonical channel slices in the run's `texture_packing.json`; network slots stay fixed and visualizations remain separate per active channel. Packed `ROMS` inputs use alpha for specular; `ROMD` inputs use alpha for displacement.

### PSNR-Based Early Stopping

Training compares validation PSNR at `--early_stop_interval` checkpoints, using compressed FNTC PSNR when ASTC comparison is enabled. Two consecutive improvements below the threshold stop an eligible phase; the decoder phase first reduces its learning rate as described below.

| Argument | Default | Description |
|----------|:-------:|-------------|
| `--early_stop` | `True` | Enable early stopping; disable with `--no-early_stop` |
| `--early_stop_interval` | `5000` | Iterations per PSNR evaluation segment |
| `--early_stop_psnr_threshold` | `0.01` | Minimum PSNR improvement threshold (dB) |

Warmup cannot stop early; codec activation resets the validation window. A decoder plateau multiplies LR by 0.25 down to 0.0001, then two stagnant windows stop training. Disabling early stopping still adjusts LR and runs the complete budget. Each phase restores the best compressed-validation PSNR state. Final checkpoint iterations denote completed work; each named stage in `multi_stage_training.json` records its validation or refinement details and selected checkpoint.

Each new validation best is saved atomically to `models/best_<phase>.pth`; standard final exports under `models/train_result_<completed_iteration>/` contain the restored best weights.

### Multi-Stage Training

Multi-stage training is enabled by default (`multi_stage_training_enable`; entry: `python -m Core.Multi_Stage_Training.multi_stage_training`). The default recipe runs:

1. **Learn the representation (`representation`)**: train the feature grid with an affine decoder, without PE or feature gradients. After warmup, optimize legal ASTC endpoint/weight parameters to learn a compressed representation for material reconstruction.
2. **Fit the decoder (`decoder`)**: freeze the learned ASTC representation and initialize a new decoder with one hidden layer of width 32 and four axial feature differences. Train the decoder on the decoded features and SR base, with a budget of 50,000 steps (`multi_stage_training_decoder_steps`; LR multiplier: `multi_stage_training_decoder_lr_multiplier`).
3. **Refit latents and select block layouts (`layout_refit`)**: freeze the final decoder and optimize a temporary latent field, then re-encode it and select legal layouts per ASTC block. Validate at step 0, every 100 steps and the final step, retaining the best bitstream whose material and enabled-PBR PSNR/SSIM/LPIPS do not regress. The default budget is 1,000 steps (`multi_stage_training_layout_refit_steps`; 0 skips refinement), with no added inference cost.

Layout refinement requires one RGBA proxy grid, SR and feature gradients, with integer GT/material resolution ratios; unsupported configurations are skipped. `multi_stage_training.json` records each stage's metrics/checkpoint and `final_metrics`.

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

ASTC exports under `Compare_ASTC<block>/iter_<iteration>/` (for example, `Compare_ASTC6x6/iter_00001000/`) save annotated `comparison_strip_<texture>.png` containing FNTC-ASTC, ASTC and GT, plus four independent images named `<Type>_Uncompressed_FNTC.png`, `<Type>_FNTC_<block>.png`, `<Type>_ASTC_<block>.png` and `<Type>_GT.png` (for example, `Diffuse_`, `Normal_` and `Roughness_`). All images are directly inside the iteration directory; uncompressed FNTC remains in logs and independent images only. Unannotated strips are no longer generated. PBR exports place all images and `metrics.json` directly in `Compare_PBRScene/iter_<iteration>/`: `comparison_strip.png`, `comparison_strip_detail.png`, full-scene `Scene_FNTC_<block>.png`/`Scene_ASTC_<block>.png`/`Scene_GT.png`, and close-ups `detail_FNTC_<block>.png`/`detail_ASTC_<block>.png`/`detail_GT.png`. Without an iteration, files go directly in `Compare_PBRScene/`. Detail views are rendered again with `pbr_detail_center` (normalized x/y in the original square raster) and `pbr_detail_zoom`; their metrics are computed separately from the full-scene training logs.

The proxy requires CUDA-enabled PyTorch, a matching CUDA toolkit, a C++ compiler and ninja.
It reuses `tools/astc_encoder_source/Source`, downloading the pinned encoder source there if absent.
It currently supports quantized Mip0 with 6x6 ASTC, four-channel feature grids and direct diffuse disabled.
Material projection additionally requires one grid and an affine SR decoder without PE or feature gradients; it also supports GT larger than the native material raster. Other single-stage configurations should set `astc_decoder_projection_interval: 0`.

The `decoder` stage freezes the learned bitstream. Checkpoints are loaded from
`models/train_result_<iteration>/model.pth`; `--load_iter -1` selects the latest completed checkpoint.

### Feature Gradient Inputs

Feature gradient options are configured in the `feature gradient configs` section:

```yaml
feature_gradient_count: 4  # 0 off | 2 right/down | 4 axial | 8 3x3 | 24 5x5
```

`Core/Feature_Grid/feature_gradients.py` uses bilinear repeat addressing to sample the center and selected offsets: one feature texel for 2/4/8 directions, up to two texels for the 24-direction 5x5 stencil. It appends each neighbor-minus-center
difference to the decoder input. The differences do not add feature bitstream bytes,
but they increase decoder input size and sampling cost. Checkpoints and
`network_data.npz` record the direction count; inference must use the same value.

The multi-stage training entry disables gradient inputs during representation learning and
enables the configured count during decoder adaptation. In single-stage training,
gradients participate in representation learning directly; set
`astc_decoder_projection_interval: 0` because pointwise material projection does
not support spatial difference inputs.

The default decoder has one hidden layer of width 32 and four axial differences. Larger stencils are supported directly, with larger decoder inputs and greater sampling cost.

ASTC training uses the same differentiably decoded texture for every feature-gradient
sample; stage two uses frozen decoded features.

`ASTCAwareTrainer.refine_gradient_codec` is an explicit fixed-decoder refinement
pass that validates legal codec proposals over complete block neighborhoods. It
is not automatically added to the default training flow.

### Seamless Tiling (Wrap Boundary Constraint)

The feature grid supports a **wrap boundary constraint**: during training, selected levels' left/right and top/bottom border features (including corners) are tied together to reduce boundary discontinuities. Sampling uses repeat addressing; boundary constraints alone do not guarantee that the reconstructed material is seamless.

### Feature Grid DDS Export

Each `models/train_result_<iteration>/` contains `model.pth`, feature-grid **DDS files (R8G8B8A8, non-SRGB)** and `network_data.npz`. Proxy models also export `feature_grid_<index>.astc`, preserving the learned bitstream. The run directory contains `resolved_config.json`, `texture_packing.json` and, for multi-stage runs, `representation_config.json` and `multi_stage_training.json`.

### Heterogeneous Multi-Feature Grid

The model supports **heterogeneous feature grids** through the `feature_grid_configs` config field:

| Field | Description |
|-------|-------------|
| `max_resolution` | Maximum resolution of the grid |
| `n_levels` | Number of grid levels; fixed to 1 when `mip0_only` is enabled |
| `quantize_bits` | Quantization precision (2 / 4 / 8 bits) |
| `save_bits` | Feature bits per texel (8 / 16 / 32 bits; at most 4 channels) |
| `learning_rate` | Per-grid learning rate |

### Configurable Per-Texture Loss Weights

Independent loss weights can be assigned to each texture type via the `texture_loss_weights` field in `configs.py`, balancing the contribution of different channels during training (e.g., diffuse typically weighted higher than displacement).

### Configurable Network Architecture

```yaml
n_neurons: 32
n_hidden_layers: 1
output_activation: hard_swish
```

These are the default decoder settings; representation learning uses `n_hidden_layers: 0` (an affine mapping). Values can also be
overridden with `--n_neurons`, `--n_hidden_layers`, and `--output_activation`.

The bundled tiny-cuda-nn CutlassMLP ignores `use_bias`. FNTC appends a constant
input for first-layer bias and stores an explicit `decoder_output_bias` parameter
for the output layer. In residual SR mode, the decoder output is signed and skips the configured output activation.

### Neural Texture Super-Resolution

Material reconstruction uses the largest latent resolution (capped by GT and preserving aspect ratio). For 1K latent + 512 SR base against 2K GT, FNTC first decodes/clamps a 1K material raster; both FNTC and the default 1K ASTC baseline then use the same texel-centered bilinear-repeat upsampler to 2K. Training differentiates through the four neighboring **material predictions**, rather than querying the decoder directly at 2K coordinates. The base is sampled directly from its source mip onto the 1K raster, avoiding a 512-to-2K-to-1K cascade.

The same reconstruction is used by warmup, codec objectives/candidate checks, frozen decoder fitting, layout refitting, inference and ASTC/PBR evaluation. Layout selection includes all affected GT pixels at integer resolution scales such as 1K-to-2K. Cross-resolution training currently requires `mip0_only: true`. `ref_astc_resolution` defaults to the FNTC material edge; setting it explicitly overrides the baseline size. Logs and `reconstruction.json` distinguish material/GT sizes and native decoder query counts. Downsampling for the ASTC input and projection proposals shares the material-aware mip prefilter (including normal-vector handling).

Set `super_resolution_enable: true` and configure `super_resolution_base_resolution` to train
residual super-resolution. The source mip at that resolution is bilinearly upsampled and provided
alongside the feature-grid samples. At equal material/GT resolution the residual target is `GT - bilinear_base`; across resolutions the loss supervises the upsampled reconstructed material against GT. Training, inference,
and ASTC comparison reconstruct `clamp(bilinear_base + predicted_residual, 0, 1)` at native material centers before the shared final upsampling. The same mip-level
offset is used for lower Mips. Enabling this mode automatically disables direct-diffuse inference,
whose direct-output semantics conflict with residual prediction.
