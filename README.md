# NeuralTexture Training Code

This project trains FNTC neural texture models with [PyTorch](https://pytorch.org/get-started/locally/) and [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn), including quantization-aware training and compressed model export.

## What is Neural Texture?

Neural Texture (NTC) is a network-based method for texture representation and compression. Unlike traditional formats such as PNG, JPEG, or ASTC, Neural Texture combines a **feature grid** with a small **MLP (multi-layer perceptron)** to compactly store and reconstruct high-resolution texture information, while still supporting random access to texels. Its advantages include:

- **High compression ratio**: preserves more detail at equal or lower storage cost.
- **Scalability**: supports multiple latent resolutions and joint representation of material parameters.
- **Versatility**: applicable not only to albedo maps but also normal maps, roughness maps, and other material textures.

### Motivation and Tiered Deployment

As real-time content increasingly adopts 2K, 4K, and higher-resolution textures, material maps become a major contributor to game package size, patch bandwidth, and runtime memory usage.
Simply increasing the compression ratio of conventional texture formats can introduce detail loss, block artifacts, or platform-specific compatibility constraints. This motivates a tiered representation strategy that balances visual quality, hardware coverage, and deployment cost.

FNTC stores a material group as compressed latent textures and a small decoder. Optional residual super-resolution adds a low-resolution source mip for the low-frequency base, with the decoder predicting additional detail. The PC and Mobile presets configure latent formats and training settings for their target platforms; conventional material baselines provide a direct quality and storage comparison.

### Differences from NVIDIA NTC SDK

This project differs from NVIDIA's [RTXNTC SDK](https://github.com/NVIDIA-RTX/Rtxntc) in several ways:

- **NTC SDK**
  - Requires specialized GPU primitives such as **CoopVec** and **CoopMat** to accelerate inference.
  - Requires engine-side integration work for deployment.

- **This project**
  - Does **not** rely on CoopVec or CoopMat. Instead, it is implemented directly with PyTorch and tiny-cuda-nn, making it simpler and easier to understand.
  - Exports compressed latent textures and `network_data.npz` for engine integration. See the [UE5 FNTC branch](https://github.com/JackFishxxx/UnrealEngine/tree/5.5.1-FNTC) for engine-side work.
  - Supports the entire pipeline of training, quantization, and export, suitable for both research and production use.

---

## Core Modules

`model.py` assembles the feature and encoding modules with the decoder; `train.py` coordinates training:

| Directory | Responsibility |
|---|---|
| `Core/Feature_Grid/` | Grid layout, quantization/QAT noise, boundary constraints, shared sampling and feature gradients |
| `Core/Positional_Encoding/` | Periodic UV sin/cos encoding |
| `Core/ASTC_Aware/` | Compression-aware training, latent ASTC round-trips and differentiable ASTC |
| `Core/BCn_Aware/` | BC1 endpoint, selector and binary-alpha optimization |
| `Core/Tile_CNN/` | Material CNN inference and fitting |
| `Core/Multi_Stage_Training/` | Representation training, decoder fitting and layout refitting |
| `Common/Material/` | Normal encoding, material compression, reconstruction and losses |
| `Common/Block_Compression/` | Conventional ASTC and BC codecs |
| `Common/Comparison/` | Material/PBR evaluation, method labels, metrics and screenshots |

## Installation

This project is implemented in Python. Please install Python first (we recommend using [Anaconda](https://www.anaconda.com/download/success) for environment setup). Refer to existing tutorials online if necessary.

### Dependencies

- Python 3.10+
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
pip install numpy Pillow torchvision torchmetrics torchtyping tensorboardX PyYAML ninja ispc_texcomp==1.0.1 texture2ddecoder==1.0.6
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

Run commands from the project root. `configs.py` loads `config_pc.yaml` by default; select another recipe with `--config`, for example `--config config_mobile.yaml`. CLI arguments override YAML values. The PC preset uses BC1 latent grids, XY normals, residual SR, multi-stage training and a material CNN. Each `feature_grid_configs` entry controls its format, quantization and feature gradients.

### Training

Run training with the following command:

```bash
python train.py \
    --config config_pc.yaml \
    --mode train \
    --data_dir ./data/test \
    --save_dir ./outputs
```

Use `--load_dir <run> --load_iter -1` with the original training recipe to resume model, optimizer, scheduler, codec, RNG and stage state from `training_state.pth`. Representation and decoder stages rebuild their saved architecture before loading. An interrupted layout pass restarts from the previous completed stage; CNN fitting has its own progress checkpoint. A completed stage is not trained again. To start a new single-stage experiment from weights, use `--warm_start` and the architecture that produced them:

```bash
python train.py \
    --config config_pc.yaml \
    --mode train \
    --data_dir ./data/test \
    --save_dir ./outputs \
    --no-multi_stage_training_enable \
    --astc_decoder_projection_interval 0 \
    --load_iter -1 \
    --warm_start \
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

### Evaluation

During training, evaluation metrics are automatically logged. For inference, reuse the architecture and material settings of the selected checkpoint; `--load_dir` is the run directory containing `models/`, and `--load_iter -1` selects its latest completed checkpoint:

```bash
python train.py \
    --config config_pc.yaml \
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
- **PSNR / SSIM / LPIPS**: Mip0 quality metrics per material group; ROMSD LPIPS averages the independently evaluated scalar channels.

Training, latent block-compressed export and material image metrics use Mip0 (`mip0_only: true`).

---

## Key Features

### Multi-Texture Joint Representation (Configurable Canonical Layout)

The model constructs a canonical layout according to the selected normal encoding. The channel order is:

```
diffuse(3) | normal(3 or 2) | roughness(1) | occlusion(1) | metallic(1) | specular(1) | displacement(1)
```

With the default `normal_encoding: xy`, the canonical layout has 10 fixed channel slots.
Missing or excluded texture types are zero-filled automatically.

The dataset defines `texture_packing` once for diffuse, normal and one scalar texture, with R/O/M/S/D priority and unused components set to one. BC1 retains at most three actual scalar maps; when capacity is exceeded, displacement is excluded first, then specular. Other formats retain up to four scalars. Excluded maps do not participate in training, metrics or screenshots. Material baselines and the compressed SR base use this mapping, exported with canonical channel slices in `texture_packing.json`. Network slots stay fixed. Packed `ROMS` inputs use alpha for specular; `ROMD` inputs use alpha for displacement.

### PSNR-Based Early Stopping

Training selects models using the primary deployment method's weighted PSNR at `--early_stop_interval` checkpoints. `enable_material_compare` controls reports only. Two consecutive improvements below the threshold stop an eligible phase; the decoder phase first reduces its learning rate as described below.

| Argument | Default | Description |
|----------|:-------:|-------------|
| `--early_stop` | `True` | Enable early stopping; disable with `--no-early_stop` |
| `--early_stop_interval` | `5000` | Iterations per PSNR evaluation segment |
| `--early_stop_psnr_threshold` | `0.01` | Minimum PSNR improvement threshold (dB) |

Warmup cannot stop early; codec activation resets the validation window. A decoder plateau multiplies LR by 0.25 down to 0.0001, then two stagnant windows stop training. Disabling early stopping still adjusts LR and runs the complete budget. Each phase restores the best compressed-validation PSNR state. Final checkpoint iterations denote completed work; each named stage in `multi_stage_training.json` records its validation or refinement details and selected checkpoint.

Each new validation best is saved atomically to `models/best_<phase>.pth`; standard final exports under `models/train_result_<completed_iteration>/` contain the restored best weights.

### Multi-Stage Training

The PC preset enables `multi_stage_training_enable`. Run through `train.py` or `python -m Core.Multi_Stage_Training.multi_stage_training`:

1. **Learn the representation (`representation`)**: train latent grids with an affine decoder, without PE or feature gradients. Each grid follows its BC1 or ASTC compression-aware training policy.
2. **Fit the decoder (`decoder`)**: freeze the compressed latents and initialize the configured decoder with each grid's feature gradients and optional PE/SR inputs. `multi_stage_training_decoder_steps` controls the budget; `multi_stage_training_decoder_lr_multiplier` sets its LR multiplier.
3. **Refit latents and select block layouts (`layout_refit`)**: freeze the decoder, optimize temporary latent fields and re-encode them. Native ASTC uses local block selection at integer GT/grid scales; other formats use whole-grid proposals. Validate at step 0, every 100 steps and the final step, retaining bitstreams whose material and enabled-PBR PSNR/SSIM/LPIPS do not regress. Set `multi_stage_training_layout_refit_steps: 0` to skip this stage.
4. **Material CNN (`material_cnn`)**: refine reconstructed materials and retain the CNN only when it passes full-image quality validation.

CNN inference must use the native material resolution used for training. Resume an interrupted CNN stage with the same data/configuration and `--mode train --load_dir <run-directory> --load_iter -1`. If the final model excludes the CNN, add `--no-material_cnn_enable` for inference.

Layout refinement jointly accepts or rolls back all grids, including mixed ASTC/BC1 formats and different resolutions. SR and feature gradients are optional. `multi_stage_training.json` records each stage's metrics, checkpoint and `final_metrics`. Material CNN fitting is independent of feature gradients; `material_cnn_steps` controls its budget.

### Quantization-Aware Training (QAT)

Training simulates quantization error in the forward pass so the model adapts to inference-time precision loss:

- **Noise Annealing**: Uniform noise is added to feature-grid values and follows a cosine schedule.
  - `--qat_noise_schedule`: `cosine` or `none` (default `cosine`)
  - `--qat_noise_mult_start` / `--qat_noise_mult_end`: `1.0` / `0.25`
  - `--qat_noise_warmup_frac`: `0.1` (the first 10% keeps the starting multiplier)

- **Straight-Through Estimator (STE) Quantization**: The forward pass uses quantized values while gradients pass through unchanged, accurately simulating inference-time rounding and reducing color bias compared to additive noise.

### Material baselines

```yaml
enable_material_compare: true
material_compare_interval: 5000
material_compare_resolution: 1024
material_compression_formats:
  diffuse: BC7
  normal: BC5
  romsd: BC1
```

Each material group accepts BC1/3/4/5/7 or a legal ASTC footprint such as `ASTC6x6`. Continuous channel capacity is validated; BC5 RGB normals reconstruct positive Z from XY. `diffuse` means BaseColor/Albedo. Choose BC3, BC7 or ASTC to retain four continuous scalar maps. This example produces the conventional baseline `Baseline_DiffuseBC7_NormalBC5_RomsdBC1`.

### FNTC tile size

```yaml
fntc_tile_size: 128
```

Tile size is measured in inference material texels. CNN training/inference, FNTC recompression and conventional baselines share tile boundaries anchored at the top left. CNN tiles read a real neighbor halo to avoid seams. Right/bottom tiles keep their actual dimensions. Tile compression blocks are assembled into a full material image; if internal tile boundaries cut codec blocks, the stitched decoded image is re-encoded so the exported texture and evaluated pixels agree. `eval_inference_tile` only limits MLP query batches and does not change material tile boundaries.

Reports record `fntc_tile_size` and `tile_layout`. Full material textures are written directly under `Compare_Material/iter_<iteration>/`, for example `DDS_Diffuse_Baseline_BC7.dds` and `DDS_Diffuse_FNTC_Recompressed_BC7.dds`, with corresponding Normal and Romsd files in their configured formats. CNN checkpoints and NPZ exports preserve tile size and validate it during restoration.

DDS material images include a full mip chain down to 1×1. Mips are filtered successively from the assembled, decoded Mip0, with vector filtering for normals. ASTC material mips use separate files such as `Diffuse_Baseline_ASTC6x6.astc` and `Diffuse_Baseline_ASTC6x6_mip1.astc`. Quality metrics and storage comparisons count Mip0; exported files additionally contain lower mips and container headers. Model latent surfaces retain their Mip0 export.

### FNTC tile recompression

```yaml
material_recompression_enable: true
```

Recompression uses `material_compression_formats` from the baseline section. Standalone titles include their actual format, e.g. `FNTC_Recompressed_BC7[DIFFUSE]`, with filenames such as `Diffuse_FNTC_Recompressed.png`.

`FNTC_Uncompressed` is inference from compressed latent textures, before material recompression. `FNTC_Recompressed_DiffuseBC7_NormalBC5_RomsdBC1` applies the same per-group formats as `Baseline_DiffuseBC7_NormalBC5_RomsdBC1`. Recompression is enabled by default, making FNTC_Recompressed the primary deployment method; disabling it selects FNTC_Uncompressed. The reconstruction objective trains the decoder; conventional recompression is evaluated without differentiating through the codec.

`[Material Test]` runs every `material_compare_interval` iterations independently of saving, early-stop windows or stage starts, with an additional terminal report. `Compare_Material/iter_<iteration>/` contains `metrics.json`, `metrics.csv`, annotated standalone images, strips and baseline/recompressed DDS or ASTC files. PBR compares all enabled methods under `Compare_PBRScene/iter_<iteration>/`: Scene uses `pbr_render_resolution`, and the separately rendered Scene Detail uses `pbr_detail_resolution`.

For 2048 GT and a 1024 inference/baseline raster, every screenshot is 2048×2048: `Diffuse_GT[1024x1024].png`, `Diffuse_GT[2048x2048].png`, `Diffuse_BC7.png`, `Diffuse_FNTC_Uncompressed.png`, `Diffuse_FNTC_Recompressed.png`. Compression occurs at 1024 before bilinear upsampling and evaluation against full-resolution GT. The low-resolution GT image shows the prefiltered source before conventional compression. Every standalone image has metrics in its upper-left corner.

PBR samples baseline and FNTC_Recompressed from the actual decoded blocks of every full-material compressed mip, sharing those bitstreams with file export. Bilinear/trilinear sampling uses repeat addressing across the complete material image. FNTC_Uncompressed and GT use floating mip pyramids at their respective native resolutions. GT-sized screenshot images are used only for image metrics and output. The report records each method's `mip_sources` and texture addressing.

PBR reports use headless **DirectX 12** hardware rasterization on Windows. The first render builds a small native DLL with the installed x64 MSVC tools and Windows SDK; compiled files are cached under `%LOCALAPPDATA%/NeuralTextureTraining/dx12`. No window or additional graphics package is needed. Reports record the GPU adapter and renderer. On other platforms, set `enable_pbr_compare: false`; training and material evaluation retain their existing CUDA paths.

Scene and Detail share the device, geometry, pipelines and shadows. Each material is prepared and uploaded once for both views; floating mip texels use one packed readback per GPU. Each channel retains at most three GPU texture layouts for floating inference, compressed inference and GT (BC5 normals may use fewer channels than the floating texture). Scene renders directly at the requested height into a rectangle fitted to the room, without CPU cropping/resizing. Invalid mip chains are rejected before GPU state changes; a native rendering failure discards the renderer so the next call can rebuild it.

The room contains two boxes topped by smooth spheres. The elevated camera and default Detail viewport (`center: [0.76, 0.59]`, `zoom: 3.5`) show a close-up of part of the right sphere and its supporting box, together with the wall, floor and cast shadows. Dataset loading already linearizes albedo, so the renderer consumes linear material values and compressed mip texels directly. OpenGL source normals use the appropriate green-channel convention. Directional and point lights use cached 1024-pixel shadow maps with PCF. Interior point lights are enabled by default; optional directional sunlight is also occluded by the room ceiling. Shading uses Cook-Torrance GGX, a small ambient fill, Reinhard tone mapping and sRGB output; it does not simulate global illumination. These rendering changes invalidate comparisons with earlier PBR metric reports.

FNTC storage counts compressed latents, decoder, CNN and the enabled low-resolution SR base; material tile cache bytes are reported separately. Baseline storage counts compressed material payload. Containers are excluded. Exported `SR_Base/metadata.json` records payload and file sizes separately.

### ASTC-Aware Latent Training

`astcenc` uses periodic CPU encoding and straight-through gradients. `astc_differentiable_proxy` trains legal ASTC endpoint/weight parameters on CUDA after warmup, using exact decoding in forward and surrogate gradients in backward. Both use the material reconstruction/PBR losses; LPIPS is an evaluation metric. Configure ASTC grids with:

```yaml
astc_aware_enable: true
astc_codec_backend: astc_differentiable_proxy
astc_codec_start_frac: 0.1
astc_learning_rate: 0.001
astc_codec_in_loop_interval: 100
astc_decoder_projection_interval: 500
```

`astc_codec_in_loop_interval` controls CPU encoding frequency; CUDA updates run every step. Set `astc_decoder_projection_interval: 0` to disable material projection. `astcenc_path` and `astcenc_quality` configure ASTC encoding where used. SR base formats follow `material_compression_formats`; latent footprints follow each grid's `format`.

The native proxy supports 6x6 ASTC with four channels per grid and direct diffuse disabled on that grid. Other footprints and eight-channel ASTC grids use conventional ASTC encoding with a round-trip STE; eight channels occupy two surfaces. Native builds download pinned encoder source to `tools/astc_encoder_source/Source` if absent.

Material projection requires an affine SR decoder without PE or feature gradients. It updates each native ASTC grid while sampling all companion grids, including different resolutions and mixed formats. Other decoder configurations skip projection and use ordinary codec training.

### Feature Gradient Inputs

Configure feature gradients independently in each `feature_grid_configs` entry; omitted counts default to `0`:

```yaml
feature_grid_configs:
  - max_resolution: 1024
    format: BC1
    feature_gradient_count: 4  # 0 off | 2 right/down | 4 axial | 8 3x3 | 24 5x5
  - max_resolution: 1024
    format: BC1
    feature_gradient_count: 0
```

`Core/Feature_Grid/feature_gradients.py` uses bilinear repeat addressing to sample the center and selected offsets: one feature texel for 2/4/8 directions, up to two texels for the 24-direction 5x5 stencil. It appends each neighbor-minus-center
difference to the decoder input. The differences do not add feature bitstream bytes,
but they increase decoder input size and sampling cost. Checkpoints and
`network_data.npz` record each grid's direction count; inference must preserve the same counts and grid order.

The multi-stage training entry disables gradient inputs during representation learning and
enables each grid's configured count during decoder adaptation. In single-stage training,
gradients participate in representation learning directly; set
`astc_decoder_projection_interval: 0` because pointwise material projection does
not support spatial difference inputs.

ASTC training uses the same differentiably decoded texture for every feature-gradient
sample; stage two uses frozen decoded features.

### Seamless Tiling (Wrap Boundary Constraint)

The feature grid supports a **wrap boundary constraint**: during training, selected levels' left/right and top/bottom border features (including corners) are tied together to reduce boundary discontinuities. Sampling uses repeat addressing; boundary constraints alone do not guarantee that the reconstructed material is seamless.

### Model Export

Each `models/train_result_<iteration>/` contains `model.pth`, `training_state.pth`, `network_data.npz` and the compressed latent surfaces declared in its manifest (BC1 DDS or ASTC). The run directory contains `resolved_config.json`, `texture_packing.json` and, for multi-stage runs, `representation_config.json` and `multi_stage_training.json`.

### Heterogeneous Multi-Feature Grid

The model supports **heterogeneous feature grids** through the `feature_grid_configs` config field:

| Field | Description |
|-------|-------------|
| `format` | Required format: `BC1` or an ASTC footprint such as `astc6x6` / `astc4x4` |
| `max_resolution` | Maximum resolution of the grid |
| `quantize_bits` | Quantization precision (2 / 4 / 8 bits) |
| `n_features_per_level` | Features per grid: ASTC supports 1/2/4/8, BC1 supports 4/8; default: 4 |
| `feature_gradient_count` | Neighbor differences: 0/2/4/8/24; default: 0 |
| `learning_rate` | Per-grid learning rate |

### Configurable Per-Texture Loss Weights

Independent loss weights can be assigned to each texture type via the `texture_loss_weights` field in `configs.py`, balancing the contribution of different channels during training (e.g., diffuse typically weighted higher than displacement).

Normal inputs use the OpenGL convention. Convert DirectX normals by flipping green before input. XY and hemi-oct encode only the positive-Z hemisphere and automatically warn about negative-Z inputs; use `normal_encoding: xyz` to preserve negative Z.

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

Set `super_resolution_enable: true` and `super_resolution_base_resolution` to provide a source mip compressed with the Baseline's exact codecs, formats, packing and tile rules alongside the latent features. The PC preset uses BC7 diffuse / BC5 normal / BC1 ROMSD. Representation learning, codec optimization, decoder fitting, CNN, evaluation and inference share this compressed base. The decoder predicts a signed residual, combined as `clamp(bilinear_base + predicted_residual, 0, 1)` at native material texel centers. This mode disables direct-diffuse inference.

`super_resolution_base_resolution` must match the actual maximum edge of a source GT mip. Material reports and inference export this low-resolution base as single-level textures with metadata under `Compare_Material/SR_Base/`. Metadata records the source GT mip; the model inference contract records the formats and normal encoding.

The material raster follows the largest latent resolution, capped by GT and preserving aspect ratio. `material_compare_resolution` overrides FNTC inference and baseline dimensions independently of latent resolution. Material CNN filtering runs on the inference raster before material recompression.

For a 1K latent and 512 SR base against 2K GT, the output material mip0 is 1K, and the base corresponds to its mip1 and the GT's mip2. Training reconstructs the 1K material and differentiates through its bilinear-repeat interpolation to GT. The base is sampled directly from its source mip onto the native raster. Baseline preparation and ASTC projection use material-aware prefiltering, including normal-vector handling. PBR samples native rasters as described in the comparison section.

### BC1 latent textures

Each `format: BC1` grid defaults independently to four features on one surface: RGB plus binary alpha. The PC preset uses two such grids for eight features, including two binary components. Set `n_features_per_level: 8` explicitly to pack two surfaces into one logical grid. BC1-aware training jointly learns RGB565 endpoints, shared categorical selectors and block modes. Transparent texels have RGB=0; alpha is binary before bilinear filtering.

The `BCn aware configs` section exposes `bcn_aware_enable` and `bcn_learning_rate`. Training uses a weak binary-usage prior controlled by `bcn_usage_weight`. BC1 grids support decoder adaptation and material CNN fitting alongside ASTC grids in mixed models.
