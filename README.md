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

> **Note**: Network channels include only retained textures, with up to three diffuse, two BC5 normal and four scalar channels. Mismatched texture resolutions follow the first available type in canonical order (diffuse, normal, R/O/M/S/D).

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

Configure learned mip levels inside each `feature_grid_configs` entry:
`n_levels: 1` (default) stores only Mip0, `n_levels: full` learns the complete
chain down to 1x1, and an integer selects a shorter chain. PC uses single-level
grids and `inference_mode: on_load`; Mobile uses full chains and `on_sample`.
There is no global `mip0_only` setting: the runtime flag is derived from the grids.
On-load reconstruction requires `n_levels: 1` for every grid; learned mip chains
require `inference_mode: on_sample`.
Each learned level has independent parameters and codec state. Shorter or
single-level companion grids clamp to their smallest available level.

Mip level selects the latent level and blends adjacent levels for fractional LOD;
it is not an input channel of the decoder. The two four-channel mobile grids
therefore use eight feature inputs plus a constant bias input. Export writes one
file per latent surface, with all configured mips inside: ASTC uses KTX2
(`feature_grid_0.ktx2`), BC1 uses DDS (`feature_grid_0_bc1_0.dds`).
`network_data.npz` records their filenames and mip dimensions. Mobile PBR
filters latent mip levels and then decodes in the pixel shader. Inference LOD uses
the native material size, just like Baseline: two 1K latents and a 1K Baseline each
have 11 levels, from 1K mip0 to 1x1 mip10. GT resolution adds no inference LOD bias.
Training and validation directly sample the original GT at the same mip index:
1K latent mip0 fits 2K GT mip0; 512 latent mip1 fits 1K GT mip1, and so on.
There is no 1K-prefiltered training target. Integer training queries cover the
original GT texel centers; subpixel and fractional queries also use its mip chain.
If the native output chain is longer than GT's, supervision clamps to GT's last
available mip while the decoder query retains the native output LOD.
Baseline still compresses a native-size material image independently of FNTC supervision.
Each grid samples
`clamp(material_lod + log2(grid_resolution / native_material_max_edge), 0, grid_mip_count - 1)`;
1K/512 grids with 1K output use offsets 0/-1. `inference_spec.on_sample` exports
the native LOD reference dimensions, per-grid offsets and their rule.
Training uses a fixed mixture of 80% geometric fine-mip priority (decay 0.5)
and 20% uniform full-chain coverage. Fractional LODs occur in 25% of blendable
batches by default; `mip_fractional_sampling_ratio` is an advanced training override.
Multi-LOD validation uses `mip_validation_samples` fixed scrambled Sobol UVs per
LOD (default 16,384), covering subtexel/block phases without consuming the training RNG.
Validation keeps the fixed mip mixture and 25% half-integer LOD weighting regardless
of the training override. Every validation can save a new best checkpoint;
early stopping retains its configured window, and restoring weights invalidates
the cached score. Validation drives scheduling and early stopping. On-sample plateau
patience is fixed at 4: the fifth consecutive non-improving validation reduces the LR.
Material image metrics still measure Mip0.
On-sample requires `quantize_bits: 8` for every grid (the default), so decoded
ASTC/BC feature mapping agrees with native texture filtering. Per-texel 2/4-bit
re-quantization remains available for PC on-load.
Codec training reads proxy representations directly; updates mark source mirrors
stale and synchronize only for source forwards or checkpoints, avoiding a second
full-chain ASTC decode at each step. BC1 usage diagnostics include each mip.
Native ASTC6x6 accumulates Adam updates per block with gradients; BC1/ASTC
roundtrip retain ordinary Adam's global parameter-step semantics, so zero-gradient
levels may still move under historical momentum. Their optimization trajectories
are not assumed to be identical.
Mobile inference uses schema version 3 and DX12 protocol 2. The previous original-GT
LOD contract is rejected rather than silently reinterpreting old mip weights.
The training recipe records original-GT same-mip supervision and rejects exact
resumption of the previous prefiltered-target recipe. ASTC and multi-mip BC1 exports use
compression schema 4: `surfaces[]` lists container files and `mips[]` lists their
level dimensions. All-single-level BC1 exports retain compression schema 2.
Checkpoint weights and inference contracts are unchanged by container packaging.

Storage comparisons count actual deployment file sizes: every exported latent
level, decoder/CNN `network_data.npz` (including metadata), and texture/container
headers. Baseline and raw GT always count their full material mip chains down to
1x1, matching material export and PBR sampling, independently of latent level counts.
Enabled single-level SR base files are counted separately within the FNTC total;
training checkpoints and runtime material output caches are excluded. Small mip
costs use exact codec block rounding rather than a blanket 4/3 multiplier.

---

## Key Features

### Multi-Texture Joint Representation (Configurable Canonical Layout)

The model constructs a canonical layout according to the selected normal encoding. The channel order is:

```
diffuse(3) | normal(3 or 2) | roughness(1) | occlusion(1) | metallic(1) | specular(1) | displacement(1)
```

The network and SR base use only retained channels in this order, without empty material slots. Multi-mip model queries include normalized Mip for latent selection; decoder inputs omit it.

Diffuse source images are decoded with the standard IEC 61966-2-1 sRGB transfer
function before resizing, mip generation and training. Material values, decoder
outputs and PBR inputs remain linear; screenshots use the inverse standard sRGB
function, matching the PBR shader. Normal/scalar maps and latent features receive
no sRGB conversion. Training checkpoints record the source transfer function;
older Gamma 2.2 runs require a new run or `--warm_start` because the targets changed.

The dataset retains up to four scalar maps in R/O/M/S/D priority. When ROMSD uses BC1 and diffuse uses BC7, AO moves to diffuse alpha, leaving at most three scalars in BC1. Without a BC7 diffuse texture, BC1 retains only three scalars. BC5 normals always use positive-Z XY encoding for GT, baseline and SR base. Excluded maps do not participate in training, metrics or screenshots. Baseline and SR base share the mapping exported in `texture_packing.json`. Packed `ROMS` inputs use alpha for specular; `ROMD` inputs use alpha for displacement.

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

Material textures contain a full mip chain down to 1×1 in one file per group: BC uses DDS, ASTC uses KTX2, e.g. `Diffuse_Baseline_ASTC6x6.ktx2`. Mips are filtered successively from the assembled, decoded Mip0, with vector filtering for normals. KTX2 stores the existing linear ASTC UNORM blocks without transcoding or supercompression. Material image metrics measure Mip0; storage includes the complete chain and container headers. Latent export packs every configured independently learned level into the same surface file; it never generates substitute mips by resizing latent Mip0. SR base exports remain single-level DDS/KTX2 textures.

### FNTC tile recompression

```yaml
material_recompression_enable: true
```

Recompression uses `material_compression_formats` from the baseline section. Standalone titles include their actual format, e.g. `FNTC_Recompressed_BC7[DIFFUSE]`, with filenames such as `Diffuse_FNTC_Recompressed.png`.

`FNTC_Uncompressed` is inference from compressed latent textures, before material recompression. `FNTC_Recompressed_DiffuseBC7_NormalBC5_RomsdBC1` applies the same per-group formats as `Baseline_DiffuseBC7_NormalBC5_RomsdBC1`. Recompression is enabled by default, making FNTC_Recompressed the primary deployment method; disabling it selects FNTC_Uncompressed. The reconstruction objective trains the decoder; conventional recompression is evaluated without differentiating through the codec.

`[Material Test]` runs every `material_compare_interval` iterations independently of saving, early-stop windows or stage starts, with an additional terminal report. `Compare_Material/iter_<iteration>/` contains `metrics.json`, `metrics.csv`, annotated standalone images, strips and baseline/recompressed DDS or KTX2 files. PBR compares all enabled methods under `Compare_PBRScene/iter_<iteration>/`: Scene uses `pbr_render_resolution`, and the separately rendered Scene Detail uses `pbr_detail_resolution`.

For 2048 GT and a 1024 inference/baseline raster, every screenshot is 2048×2048: `Diffuse_GT[1024x1024].png`, `Diffuse_GT[2048x2048].png`, `Diffuse_BC7.png`, `Diffuse_FNTC_Uncompressed.png`, `Diffuse_FNTC_Recompressed.png`. Compression occurs at 1024 before bilinear upsampling and evaluation against full-resolution GT. The low-resolution GT image shows the prefiltered source before conventional compression. Every standalone image has metrics in its upper-left corner.

PBR samples baseline and FNTC_Recompressed from the actual decoded blocks of every full-material compressed mip, sharing those bitstreams with file export. Bilinear/trilinear sampling uses repeat addressing across the complete material image. FNTC_Uncompressed and GT use floating mip pyramids at their respective native resolutions. GT-sized screenshot images are used only for image metrics and output. The report records each method's `mip_sources` and texture addressing.

PBR reports use headless **DirectX 12** hardware rasterization on Windows. The first render builds a small native DLL with the installed x64 MSVC tools and Windows SDK; compiled files are cached under `%LOCALAPPDATA%/NeuralTextureTraining/dx12`. No window or additional graphics package is needed. Reports record the GPU adapter and renderer. On other platforms, set `enable_pbr_compare: false`; training and material evaluation retain their existing CUDA paths.

All retained ROMSD properties participate in rendering: roughness controls GGX highlight width; occlusion attenuates ambient fill without darkening direct light; metallic blends dielectric/metal Fresnel and removes metallic diffuse reflection; specular controls dielectric F0. Displacement moves refined mesh vertices along their surface normals by `(height - 0.5) * pbr_displacement_scale`, updating surface gradients, silhouettes and cast shadows. The scale defaults to `0.03` world units; `0` disables displacement, and absent height maps leave geometry unchanged. Height comes from each method's native Mip0, including actual decoded compressed texels for baseline/recompressed output. Room/box faces use 32 subdivisions per edge; spheres use their existing dense mesh, and duplicated edge positions are welded to prevent cracks. Scene and Detail share each material's displaced geometry and refreshed shadows. The differentiable per-texel training proxy remains separate from this geometry renderer. Packing still retains at most four scalar maps; when all five are supplied, R/O/M/S take priority and D is excluded.

Scene and Detail share the device, geometry, pipelines and shadows. Each material is prepared and uploaded once for both views; floating mip texels use one packed readback per GPU. Each channel retains at most three GPU texture layouts for floating inference, compressed inference and GT (BC5 normals may use fewer channels than the floating texture). Scene renders directly at the requested height into a rectangle fitted to the room, without CPU cropping/resizing. Invalid mip chains are rejected before GPU state changes; a native rendering failure discards the renderer so the next call can rebuild it.

The room contains two boxes topped by smooth spheres. The elevated camera and default Detail viewport (`center: [0.76, 0.59]`, `zoom: 3.5`) show a close-up of part of the right sphere and its supporting box, together with the wall, floor and cast shadows. Dataset loading already linearizes albedo, so the renderer consumes linear material values and compressed mip texels directly. OpenGL source normals use the appropriate green-channel convention. Directional and point lights use cached 1024-pixel shadow maps with PCF. Interior point lights are enabled by default; optional directional sunlight is also occluded by the room ceiling. Shading uses Cook-Torrance GGX, a small ambient fill, Reinhard tone mapping and sRGB output; it does not simulate global illumination. These rendering changes invalidate comparisons with earlier PBR metric reports.

FNTC storage counts all compressed latent files, decoder/CNN NPZ metadata and the enabled single-level SR base; material tile cache bytes are reported separately. Baseline storage includes full material mip-chain files and their container headers. Exported `SR_Base/metadata.json` records payload and file sizes separately.

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

Each `models/train_result_<iteration>/` contains `model.pth`, `training_state.pth`, `network_data.npz` and the compressed latent surfaces declared in its manifest (BC1 DDS or ASTC KTX2). The run directory contains `resolved_config.json`, `texture_packing.json` and, for multi-stage runs, `representation_config.json` and `multi_stage_training.json`.

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

Normal inputs use the OpenGL convention. Convert DirectX normals by flipping green before input. XY and hemi-oct force positive Z for both GT and reconstructed normals; `normal_encoding: xyz` preserves both Z signs unless the material normal format is BC5, which requires XY.

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

The `BCn aware configs` section exposes `bcn_aware_enable` and `bcn_learning_rate`. `bcn_alpha_input_enable: false` (default) omits each BC1 latent texture's one-bit alpha from decoder inputs and feature differences; set it to `true` to include alpha. Storage stays RGBA. Training uses a weak binary-usage prior controlled by `bcn_usage_weight`. BC1 grids support decoder adaptation and material CNN fitting alongside ASTC grids in mixed models.
