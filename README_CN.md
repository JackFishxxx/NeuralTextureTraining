# NeuralTexture 训练代码

该项目用于训练神经纹理（Neural Texture）模型，代码基于 [PyTorch](https://pytorch.org/get-started/locally/) 与 [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) 实现，支持量化感知训练与压缩模型导出。

## 什么是神经纹理?

神经纹理（Neural Texture, NTC）是一种基于神经网络的纹理表示与压缩方法。与传统的 PNG、JPEG、ASTC 等压缩方式不同，神经纹理使用**特征网格（Feature Grid）** 与**小型神经网络（MLP）** 相结合，将高分辨率的纹理信息以更紧凑的形式存储和还原，同时支持纹理对于图像随机读取的要求。其优势包括：

- **高压缩率**：在相同或更低的存储开销下保留更多细节。
- **可扩展性**：支持多个 latent 分辨率及材质参数的联合表示。
- **强适用性**：不仅适用于颜色贴图，还可以用于法线、粗糙度等材质纹理的表示。

### 研究背景与分层部署动机

随着实时渲染内容向 2K、4K 及更高分辨率发展，材质纹理已成为游戏安装包体积、补丁下载规模以及运行时显存占用的重要来源。
单纯提高传统纹理压缩比通常会带来细节损失、块状伪影或平台兼容性问题，因此需要在资产兼容性、设备覆盖范围和视觉质量之间进行分层设计。

FNTC 使用压缩 latent 纹理和小型 decoder 表示一组材质。可选的残差超分以低分辨率源 mip 提供低频基础图像，由 decoder 预测额外细节。PC 和 Mobile 预设分别配置目标平台的 latent 格式与训练设置，传统材质基线用于直接比较质量和存储开销。

### 与 NVIDIA NTC SDK 的区别

本项目与 NVIDIA 的 [RTXNTC SDK](https://github.com/NVIDIA-RTX/Rtxntc) 有以下区别：

- NTC SDK
  - 需要使用**协同向量 (CoopVec)** 、 **协同矩阵 (CoopMat)** 等新硬件特性来实现推理加速。
  - 目前在引擎侧仍存在适配问题。

- 本项目
  - **不**依赖协同向量或协同矩阵，直接基于PyTorch与tiny-cuda-nn实现，简洁且易于理解。
  - 导出压缩 latent 纹理及 `network_data.npz` 供引擎集成；引擎侧实现参见 [UE5 FNTC 分支](https://github.com/JackFishxxx/UnrealEngine/tree/5.5.1-FNTC)。
  - 支持训练、量化、导出全流程，研究与工程实践均可落地。


## Core 算法模块

`model.py` 将特征和编码模块与 decoder 组合起来；`train.py` 组织训练流程：

| 目录 | 职责 |
|---|---|
| `Core/Feature_Grid/` | 网格布局、量化/QAT 噪声、边界约束、共享采样和 feature gradients |
| `Core/Positional_Encoding/` | UV 周期性 sin/cos 编码 |
| `Core/Material_Encoding/` | 法线编码及材质通道打包/压缩 |
| `Core/ASTC_Aware/` | 压缩感知训练、latent ASTC 编解码与可微 ASTC |
| `Core/BCn_Aware/` | BC1 端点、选择器和二值 alpha 优化 |
| `Core/Tile_CNN/` | 材质 CNN 推理与拟合 |
| `Core/Multi_Stage_Training/` | 表示训练、decoder 拟合和布局重拟合 |
| `Common/Block_Compression/` | 传统 ASTC 和 BC 编解码 |
| `Common/Comparison/` | 材质/PBR 评估、方法命名、指标与截图 |

## 安装

这是一个基于Python实现的项目，因此请先安装Python，本项目建议使用 [Anaconda](https://www.anaconda.com/download/success) 进行环境配置。具体安装可参考网上已有教程。

### 依赖

- Python 3.10+
- PyTorch （与本机的CUDA版本匹配）
- tiny-cuda-nn（当前 FNTC 模型必需）
- 其他依赖：`numpy`、`Pillow`、`torchvision`、`torchmetrics`、`torchtyping`、`tensorboardX`、`PyYAML`、`ninja`

### 设备要求

当前 FNTC 训练与推理入口需要 NVIDIA GPU、CUDA 版 PyTorch 和 tiny-cuda-nn。默认的可微 ASTC 后端还需要与 PyTorch 匹配的 CUDA toolkit、受支持的 C++ 编译器和 ninja。Windows 使用 Visual Studio C++ 构建工具；Linux 使用对应 PyTorch/CUDA 版本支持的编译器。从源码安装 tiny-cuda-nn 还需要 CMake。

### 部署环境

先clone项目

```bash
git clone https://github.com/JackFishxxx/NeuralTextureTraining.git
cd NeuralTextureTraining
```

然后，安装 [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn)

```bash
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

最后，安装其余依赖

```bash
pip install numpy Pillow torchvision torchmetrics torchtyping tensorboardX PyYAML ninja ispc_texcomp==1.0.1 texture2ddecoder==1.0.6
```

## 数据

我们在 `data` 文件夹中提供了一份示例数据。亦或者，你也可以使用自己的数据。

格式如下：

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

你需要将纹理存放在对应的文件夹内。代码会根据文件名中的关键词**自动识别**纹理类型，支持的纹理类型及其识别关键词如下：

| 纹理类型 | 通道数 | 识别关键词 |
|---------|:------:|-----------|
| diffuse（漫反射/颜色） | 3 | `diffuse`、`albedo`、`color`、`diff` |
| normal（法线） | 2（`xy`/`hemi_oct`）或 3（`xyz`） | `normal`、`nor_gl` |
| roughness（粗糙度） | 1 | `roughness`、`rough` |
| occlusion（环境光遮蔽） | 1 | `occlusion`、`ao`、`ambient` |
| metallic（金属度） | 1 | `metallic`、`metalness` |
| specular（高光） | 1 | `specular` |
| displacement（位移） | 1 | `displacement`、`disp`、`height` |

> **提示**：规范布局总通道数取决于法线编码：`normal_encoding: xyz` 时为 11 通道，`xy` 或 `hemi_oct` 时为 10 通道。缺失材质槽位在目标中填零，不参与 loss 和指标。分辨率不一致时，按 canonical 顺序（diffuse、normal、R/O/M/S/D）中首个有效类型的分辨率对齐。

具体命名、识别和通道数量可参考 `dataset.py` 中的 `get_texture_config()` 函数。


## 运行

从项目根目录执行命令。`configs.py` 默认加载 `config_pc.yaml`，可通过 `--config` 选择其他配置，例如 `--config config_mobile.yaml`；CLI 参数覆盖 YAML 值。PC 预设使用 BC1 latent grid、XY 法线、残差超分、多阶段训练和材质 CNN。每个 `feature_grid_configs` 条目独立配置格式、量化及 feature gradients。

### 训练

对于运行时，可以使用如下的指令

```bash
python train.py \
    --config config_pc.yaml \
    --mode train \
    --data_dir ./data/test \
    --save_dir ./outputs
```

使用原训练配置及 `--load_dir <运行目录> --load_iter -1` 可从 `training_state.pth` 恢复优化器、调度器、codec、随机数和阶段状态，表示/decoder 阶段会先重建保存时的结构。布局阶段中断后从上一已完成阶段重新执行；CNN 使用独立进度文件。已完成阶段不会重复训练。若要从权重开始新的单阶段实验，显式使用 `--warm_start` 并沿用保存时的网络结构：

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

### 批量训练（GroupsBatching 模式）

当需要对多组纹理进行批量训练时，可以使用 `--groups_batching` 模式。该模式会自动遍历 `--groups_batch_dir` 下的所有子文件夹，依次对每组纹理进行训练，并将结果分别保存至 `--groups_save_dir` 下对应名称的子文件夹中。

数据目录结构示例：

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

运行命令：

```bash
python train.py \
    --groups_batching \
    --groups_batch_dir ./data_batch \
    --groups_save_dir ./outputs_batch
```

相关参数：

| 参数 | 默认值 | 说明 |
|------|:------:|------|
| `--groups_batching` | `False` | 启用批量训练模式 |
| `--groups_batch_dir` | `data_batch` | 包含多个纹理组子目录的根目录 |
| `--groups_save_dir` | `save_batch` | 批量训练结果的保存根目录 |

### 评估

训练时自动记录评估数据。推理时须沿用所选 checkpoint 的网络结构和材质设置；`--load_dir` 指向包含 `models/` 的运行目录，`--load_iter -1` 选择最新已完成 checkpoint：

```bash
python train.py \
    --config config_pc.yaml \
    --mode infer \
    --data_dir ./data/test \
    --save_dir ./outputs \
    --load_iter -1 \
    --load_dir outputs/yy_mm_dd_HH_MM_SS
```

## 监控

该项目提供了运行时指标监控等功能。

### Tensorboard

每次运行的事件日志位于 `<save_dir>/<timestamp>/tensorboard/`，可从输出根目录监控所有运行：

```bash
tensorboard --logdir ./outputs
```

训练过程中会自动记录以下指标：

- **Loss/train**：训练损失
- **PSNR / SSIM / LPIPS**：各材质组的 Mip0 图像质量指标；ROMSD LPIPS 分别评估有效标量通道后取平均。

训练、latent 块压缩导出和材质指标对比使用 Mip0（`mip0_only: true`）。

## 主要特性

### 多纹理联合表示（可配置规范布局）

模型按法线编码方式构造规范布局，通道顺序如下：

```
diffuse(3) | normal(3 或 2) | roughness(1) | occlusion(1) | metallic(1) | specular(1) | displacement(1)
```

默认 `normal_encoding: xy`，因此法线占 2 个网络通道，总规范布局为 10 个固定通道槽位；
缺失或被排除的纹理类型对应槽位填零。

法线输入统一使用 OpenGL 约定；DirectX 法线需在输入前翻转绿色通道。`xy` 和 `hemi_oct` 只能表示正 Z 半球，发现负 Z 输入时自动警告；需要保留负 Z 时使用 `normal_encoding: xyz`。

数据集统一生成 `texture_packing`，描述 diffuse、normal 和一张标量纹理，按 R/O/M/S/D 优先级紧凑打包，空余分量填 1。BC1 最多保留三个实际加载的标量通道，超出时先移除 displacement，再移除 specular；其他格式最多保留四个标量。被移除的贴图不参与训练、指标和截图输出。材质基线和压缩 SR base 按同一映射执行，映射及 canonical 通道范围导出到 `texture_packing.json`；网络槽位保持固定。输入 `ROMS` 的 alpha 表示 specular，`ROMD` 的 alpha 表示 displacement。
如果只提供部分纹理类型，对应的缺失通道将自动填零，不影响训练和推理。

### 基于 PSNR 的自动早停

训练在 `--early_stop_interval` 检查点使用核心部署方法的加权 PSNR 选择模型；`enable_material_compare` 仅控制报告输出。连续两个窗口的提升低于阈值时，可早停的阶段才停止；decoder 阶段会先按下文规则降低学习率。

| 参数 | 默认值 | 说明 |
|------|:------:|------|
| `--early_stop` | `True` | 是否启用早停，可用 `--no-early_stop` 关闭 |
| `--early_stop_interval` | `5000` | 计算 PSNR 提升的迭代间隔 |
| `--early_stop_psnr_threshold` | `0.01` | 最小 PSNR 提升阈值（dB） |

warmup 不早停，codec 启动后重置验证窗口。decoder 阶段平台期先将学习率乘 0.25，最低降至 0.0001；达到下限后，连续两个停滞窗口才停止。关闭早停仍执行 LR 调整并跑满步数。阶段结束恢复压缩验证 PSNR 最佳的模型；最终导出编号表示实际完成工作量，`multi_stage_training.json` 的具名阶段记录验证、精修详情和所选 checkpoint。

每次验证刷新最佳分数时原子写入 `models/best_<phase>.pth`；常规最终导出仍在 `models/train_result_<完成迭代>/`，包含恢复后的最佳权重。

### 多阶段训练（Multi-Stage Training）

PC 预设开启 `multi_stage_training_enable`，可通过 `train.py` 或 `python -m Core.Multi_Stage_Training.multi_stage_training` 运行：

1. **学习表示（`representation`）**：使用仿射 decoder 训练 latent grid，关闭 PE 和 feature gradients。每张 grid 按其 BC1 或 ASTC 感知训练策略优化。
2. **拟合解码器（`decoder`）**：固定压缩 latent，按配置初始化 decoder，启用各 grid 的 feature gradients 及可选 PE/SR 输入。`multi_stage_training_decoder_steps` 控制步数，`multi_stage_training_decoder_lr_multiplier` 控制学习率倍率。
3. **重拟合 latent 与选择块布局（`layout_refit`）**：固定 decoder，优化临时 latent 并重新编码。原生 ASTC 在 GT/grid 为整数比例时逐块选择，其他格式使用整张 grid 候选。在第 0 步、每 100 步和末步验证，保留材质及已启用 PBR 的 PSNR/SSIM/LPIPS 均不退化的码流。`multi_stage_training_layout_refit_steps: 0` 跳过此阶段。
4. **材质 CNN（`material_cnn`）**：进一步改善重建材质，通过全图质量验收后才启用，否则保留不含 CNN 的模型。

CNN 推理须沿用训练时的原生材质分辨率。CNN 阶段中断后，使用相同数据和配置，传入 `--mode train --load_dir <运行目录> --load_iter -1` 可继续训练。若最终未启用 CNN，推理时添加 `--no-material_cnn_enable`。

布局精修对所有 grid 整体接受或回滚，支持不同分辨率及 ASTC/BC1 混合格式；SR 和 feature gradients 均为可选。`multi_stage_training.json` 记录各阶段指标、checkpoint 及 `final_metrics`。材质 CNN 拟合独立于 feature gradients，步数由 `material_cnn_steps` 控制。

### 量化感知训练（QAT）

训练时使用**量化感知训练**，在前向传播中模拟量化误差，使模型适应推理时的精度损失：

- **噪声退火（Noise Annealing）**：训练初期向 feature grid 特征加入均匀噪声，按 cosine 计划衰减。
  - `--qat_noise_schedule`：`cosine` 或 `none`，默认 `cosine`
  - `--qat_noise_mult_start` / `--qat_noise_mult_end`：默认 `1.0` / `0.25`
  - `--qat_noise_warmup_frac`：默认 `0.1`，前 10% 迭代保持起始强度

- **直通估计器（STE）量化**：前向传播使用量化值，反向传播梯度直接通过，准确模拟推理时的舍入行为，降低颜色偏差。

### 材质基线

```yaml
enable_material_compare: true
material_compare_interval: 5000
material_compare_resolution: 1024
material_compression_formats:
  diffuse: BC7
  normal: BC5
  romsd: BC1
```

每个材质组可选择 BC1、BC3、BC4、BC5、BC7 或合法 ASTC footprint（如 `ASTC6x6`）；连续通道容量必须匹配，RGB 法线使用 BC5 时由 XY 重建正 Z。`diffuse` 对应 BaseColor/Albedo。需要保留四个连续标量时选择 BC3、BC7 或 ASTC。上例的传统材质基线命名为 `Baseline_DiffuseBC7_NormalBC5_RomsdBC1`。

### FNTC tile 大小

```yaml
fntc_tile_size: 128
```

tile 大小以推理材质栅格上的 texel 计。CNN 训练采样和推理中心区域、FNTC 二次压缩、传统基线压缩使用同一套从左上角开始的 tile 边界。CNN 额外读取真实邻域 halo，避免引入 tile 接缝；右侧和底部不足一个 tile 时使用实际尺寸。各 tile 的压缩块组装为整张材质纹理，字节数按组装后的有效压缩载荷统计。`eval_inference_tile` 仅控制 MLP 查询批次的显存用量，不改变上述 tile 边界。

报告记录 `fntc_tile_size` 和 `tile_layout`；完整材质纹理直接输出到 `Compare_Material/iter_<iteration>/`，例如 `DDS_Diffuse_Baseline_BC7.dds`、`DDS_Diffuse_FNTC_Recompressed_BC7.dds`，Normal 和 Romsd 文件同样标明方法及配置格式。CNN checkpoint 和 NPZ 保存 tile 大小，并校验恢复时的设置。

### FNTC 二次 tile 压缩

```yaml
material_recompression_enable: true
```

二次压缩使用材质基线一节的 `material_compression_formats`。独立截图标题带实际格式，例如 `FNTC_Recompressed_BC7[DIFFUSE]`，文件名如 `Diffuse_FNTC_Recompressed.png`。

`FNTC_Uncompressed` 指压缩 latent 解码后推理得到、尚未再次压缩的材质；`FNTC_Recompressed_DiffuseBC7_NormalBC5_RomsdBC1` 指同一材质按基线配方再次压缩；`Baseline_DiffuseBC7_NormalBC5_RomsdBC1` 指传统材质压缩基线。默认开启二次压缩，以 FNTC_Recompressed 为核心部署方法；关闭后使用 FNTC_Uncompressed。重建损失用于训练 decoder，真实二次压缩用于验证，不通过传统编码器反传。

`[Material Test]` 按 `material_compare_interval` 触发，周期独立于保存、早停及阶段起点，阶段结束另输出最终结果。材质报告位于 `Compare_Material/iter_<iteration>/`，同目录保存 `metrics.json`、`metrics.csv`、带左上角指标的独立图、拼图以及二次压缩和基线的 DDS/ASTC 文件。PBR 在 `Compare_PBRScene/iter_<iteration>/` 比较各启用方法，Scene 使用 `pbr_render_resolution`，独立渲染的 Scene Detail 使用 `pbr_detail_resolution`。

整张材质 DDS 包含逐级缩小至 1×1 的完整 mip 链，从拼接、解码后的 Mip0 连续过滤生成各级，法线采用向量过滤。ASTC 材质单独保存为 `Diffuse_Baseline_ASTC6x6.astc`、`Diffuse_Baseline_ASTC6x6_mip1.astc` 等文件。压缩块对齐时直接拼接 tile 码流；内部 tile 边界切开压缩块时，对拼接后的解码图重新编码，使导出纹理与评估像素一致。质量指标与存储对比仍按 Mip0 统计；导出文件另外包含低级 mip 和容器头。模型 latent surface 仍只导出 Mip0。

例如 GT=2048、推理/基线=1024，以下文件都保存为2048×2048；低分辨率内容在压缩之后双线性拉伸再对高分辨率 GT 计算指标：`Diffuse_GT[1024x1024].png`、`Diffuse_GT[2048x2048].png`、`Diffuse_BC7.png`、`Diffuse_FNTC_Uncompressed.png`、`Diffuse_FNTC_Recompressed.png`。低分辨率 GT 图表示预滤波后的未压缩源，`Diffuse_BC7.png` 表示其传统压缩结果。

PBR 对 baseline 和 FNTC_Recompressed 采样整张材质各级实际压缩、解码后的 mip，与文件导出共用相同码流。双线性/三线性采样在完整材质上采用 repeat 寻址。FNTC_Uncompressed 和 GT 分别在各自原生分辨率生成浮点 mip 链。对齐 GT 尺寸的截图仅用于图像指标与输出；报告记录各方法的 `mip_sources` 和纹理寻址规则。

PBR 报告在 Windows 上使用无窗口的 **DirectX 12** 硬件光栅化。首次渲染通过已安装的 x64 MSVC 和 Windows SDK 编译小型原生 DLL，缓存到 `%LOCALAPPDATA%/NeuralTextureTraining/dx12`，不需要额外图形包。报告记录渲染器与 GPU 型号。其他平台设置 `enable_pbr_compare: false`；训练和材质评估继续使用原有 CUDA 路径。

Scene 与 Detail 共用设备、几何、管线和阴影，每种材质只准备并上传一次；浮点 mip 纹素在每个 GPU 上打包后统一回读。每个通道最多缓存三种 GPU 纹理布局，对应浮点推理、压缩推理和 GT（BC5 法线的通道数可能少于浮点纹理）。Scene 按目标高度直接渲染贴合房间范围的矩形，移除 CPU 裁剪缩放。非法 mip 链在修改 GPU 状态前被拒绝；原生渲染失败后会丢弃失效渲染器，下次调用时重新创建。

场景包含两个顶置平滑球体的方体。相机略微俯视，默认 Detail 视口（`center: [0.76, 0.59]`、`zoom: 3.5`）拉近到右侧球体与方体的局部，同时保留墙面、地面和投影。数据集加载时已将颜色转为线性，因此渲染器直接使用线性材质值及压缩 mip 纹素，避免重复 gamma 转换。OpenGL 源法线按对应的绿色通道约定解释。平行光、点光源均采用缓存的 1024 像素阴影贴图与 PCF 过滤。默认使用室内点光源；可选的室外平行光也会被天花板遮挡。着色采用 Cook-Torrance GGX、少量环境补光、Reinhard 色调映射及 sRGB 输出，不模拟全局光照。本次渲染修改后的 PBR 指标不能直接与旧报告比较。

FNTC 存储统计计压缩 latent、decoder 和 CNN；材质 tile cache 大小单列。基线统计压缩材质 payload，不含容器头或共享 SR base。

### ASTC 感知 Latent 训练

`astcenc` 定期在 CPU 编码并使用直通梯度。`astc_differentiable_proxy` 在 warmup 后通过 CUDA 训练合法 ASTC 端点/权重，前向精确解码，反向使用近似梯度。两者使用材质重建/PBR loss，LPIPS 用于评测。ASTC grid 的配置示例：

```yaml
astc_aware_enable: true
astc_codec_backend: astc_differentiable_proxy
astc_codec_start_frac: 0.1
astc_learning_rate: 0.001
astc_codec_in_loop_interval: 100
astc_decoder_projection_interval: 500
```

`astc_codec_in_loop_interval` 控制 CPU 编码频率，CUDA 每步更新。`astc_decoder_projection_interval: 0` 关闭材质投影。`astcenc_path`、`astcenc_quality` 配置 ASTC 编码；`astc_block` 控制共享 SR base 的块尺寸，与各 grid 的 `format` 及材质基线格式独立。

原生可微代理支持 6x6 ASTC、每 grid 四通道，且该 grid 不承担 direct diffuse。其他块尺寸及八通道 ASTC grid 使用传统编码配合 round-trip STE；八通道占用两张 surface。本地构建所需的固定版本编码器源码会在缺失时下载到 `tools/astc_encoder_source/Source`。

材质投影要求无 PE 和 feature gradients 的仿射 SR decoder，逐张更新原生 ASTC grid，同时采样其他不同分辨率或格式的 grid。其他 decoder 配置跳过投影，使用普通 codec 训练。

### Feature Gradient 输入

Feature gradients 在每个 `feature_grid_configs` 条目内独立配置，省略时默认为 `0`：

```yaml
feature_grid_configs:
  - max_resolution: 1024
    format: BC1
    feature_gradient_count: 4  # 0 关闭 | 2 右/下 | 4 四轴 | 8 3x3 | 24 5x5
  - max_resolution: 1024
    format: BC1
    feature_gradient_count: 0
```

`Core/Feature_Grid/feature_gradients.py` 使用双线性 repeat 采样中心与所选偏移位置：2/4/8 方向偏移一个 feature 纹素，24 方向的 5x5 邻域最远偏移两个纹素。邻居减去中心的差分拼接到 decoder 输入。差分不增加
feature 码流，但会增加 decoder 输入和采样开销。checkpoint 和 `network_data.npz`
会记录每张 grid 的梯度方向数，推理必须沿用相同配置及 grid 顺序。

多阶段入口在 representation 阶段关闭所有 grid 的梯度输入，decoder 阶段按每张 grid 的配置启用；单阶段训练时梯度直接
参与表示学习。由于逐纹素材质投影不支持空间差分输入，单阶段使用梯度时应将
`astc_decoder_projection_interval` 设为 `0`。

ASTC 阶段的 feature-gradient 采样使用同一份可微解码 feature；第二阶段使用冻结的解码 feature。


### 无缝平铺（Wrap Boundary Constraint）

特征网格支持**边界平铺约束**：训练时约束所选层级的左右、上下及四角特征，降低边界不连续。采样使用 repeat 寻址，单靠边界约束不能保证最终重建材质完全无缝。

### 模型导出

每个 `models/train_result_<迭代>/` 包含 `model.pth`、`training_state.pth`、`network_data.npz` 和清单声明的压缩 latent surface（BC1 DDS 或 ASTC）。运行目录保存 `resolved_config.json`、`texture_packing.json`；多阶段运行还保存 `representation_config.json` 和 `multi_stage_training.json`。

### 异构多特征网格

通过 `feature_grid_configs` 可以配置多个分辨率或不同精度的 feature grid：

| 字段 | 说明 |
|------|------|
| `format` | 必填格式：`BC1` 或 `astc6x6`、`astc4x4` 等 ASTC 块尺寸 |
| `max_resolution` | 最大分辨率 |
| `quantize_bits` | 量化精度（2/4/8 位） |
| `n_features_per_level` | 每 grid 的特征数：ASTC 支持 1/2/4/8，BC1 支持 4/8；默认 4 |
| `feature_gradient_count` | 邻域差分数：0/2/4/8/24；默认 0 |
| `learning_rate` | 独立学习率 |

### 可配置的纹理损失权重

可在 `configs.py` 的 `texture_loss_weights` 字段中为每种纹理类型设置独立的损失权重，以平衡不同纹理通道对训练的贡献（如颜色贴图通常比粗糙度更重要）。

### 可配置网络结构

```yaml
n_neurons: 32
n_hidden_layers: 1
output_activation: hard_swish
```

以上为默认 decoder 设置；representation 阶段使用 `n_hidden_layers: 0`（仿射映射）。可通过 CLI 覆盖 `--n_neurons`、
`--n_hidden_layers` 和 `--output_activation`。

当前 tiny-cuda-nn CutlassMLP 构建会忽略 `use_bias`，因此模型显式追加常数输入作为第一层
bias，并增加 `decoder_output_bias` 作为输出层 bias。残差超分模式下 decoder 输出为有符号残差，不应用配置指定的输出激活。

### 神经纹理超分

设置 `super_resolution_enable: true` 和 `super_resolution_base_resolution`，可将 ASTC 压缩的源 mip 与 latent 特征一起输入 decoder。decoder 预测有符号残差，在原生材质纹素中心计算 `clamp(bilinear_base + predicted_residual, 0, 1)`。此模式关闭 direct diffuse 推理。

材质栅格默认采用最大 latent 分辨率，不超过 GT 并保持宽高比。`material_compare_resolution` 可独立于 latent 分辨率覆盖 FNTC 推理及基线尺寸。材质 CNN 在推理栅格上运行，之后执行材质二次压缩。

例如 1K latent＋512 SR base 对齐 2K GT，训练先重建 1K 材质，再通过其 bilinear-repeat 插值对 GT 计算损失并反传。base 从源 mip 直接采样到原生栅格。基线准备及 ASTC 投影使用材质专用预滤波，包含法线向量处理。PBR 按对比章节所述直接采样原生栅格。

### BC1 latent 纹理

每个 `format: BC1` grid 默认为四个特征，一张 surface 保存 RGB 和二值 alpha。PC 预设使用两个 grid，合计八个特征，其中两个为二值分量。显式设置 `n_features_per_level: 8` 可在一个逻辑 grid 中存放两张 surface。BC1 感知训练联合学习 RGB565 端点、共享选择器及块模式；透明 texel 的 RGB 为零，alpha 在双线性过滤前为二值。

`BCn aware configs` 配置 `bcn_aware_enable` 和 `bcn_learning_rate`，二值使用率的弱先验由 `bcn_usage_weight` 控制。BC1 grid 支持 decoder 拟合和材质 CNN，也可与 ASTC grid 混合使用。
