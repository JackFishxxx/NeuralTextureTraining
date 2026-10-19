# NeuralTexture 训练代码

该项目用于训练神经纹理（Neural Texture）模型，代码基于 [PyTorch](https://pytorch.org/get-started/locally/) 与 [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) 实现，支持量化感知训练与模型导出。当前 FNTC 入口依赖 tiny-cuda-nn 和 NVIDIA GPU；本仓库没有独立的 CPU 训练实现。

## 什么是神经纹理?

神经纹理（Neural Texture, NTC）是一种基于神经网络的纹理表示与压缩方法。与传统的 PNG、JPEG、ASTC 等压缩方式不同，神经纹理使用**特征网格（Feature Grid）** 与**小型神经网络（MLP）** 相结合，将高分辨率的纹理信息以更紧凑的形式存储和还原，同时支持纹理对于图像随机读取的要求。其优势包括：

- **高压缩率**：在相同或更低的存储开销下保留更多细节。
- **可扩展性**：支持多分辨率、Mip，以及与材质参数的联合表示。
- **强适用性**：不仅适用于颜色贴图，还可以用于法线、粗糙度等材质纹理的表示。

### 研究背景与分层部署动机

随着实时渲染内容向 2K、4K 及更高分辨率发展，材质纹理已成为游戏安装包体积、补丁下载规模以及运行时显存占用的重要来源。
单纯提高传统纹理压缩比通常会带来细节损失、块状伪影或平台兼容性问题，因此需要在资产兼容性、设备覆盖范围和视觉质量之间进行分层设计。

本项目采用面向部署的混合策略：1K 及以下纹理继续使用成熟的 ASTC 等传统块压缩格式，以兼容既有资产、移动平台和性能受限设备；
对于具备更强 GPU 推理能力的中高端 PC，2K/4K 纹理可不作为完整高分辨率位图存储，而是在运行时通过神经纹理推理重建。
FNTC不仅使用 feature grid 表示纹理组的共享特征，还复用资产中本来就需要保留的低分辨率原始纹理 mip 作为基础图像，再由小型 MLP 预测双线性上采样无法恢复的高频残差。该低分辨率 mip 属于已有资产链路中的复用数据（freelunch），因此 FNTC 的新增表示成本主要集中在 feature texture 与网络参数，同时保留了传统路径作为兼容性回退方案。

### 与 NVIDIA NTC SDK 的区别

NVIDIA在2025年2月开源了 [RTXNTC SDK](https://github.com/NVIDIA-RTX/Rtxntc)，提供了NVIDIA官方对于神经纹理的支持。但本项目与RTXNTC仍有一定差别：

- NTC SDK
  - 需要使用**协同向量 (CoopVec)** 、 **协同矩阵 (CoopMat)** 等新硬件特性来实现推理加速。
  - 目前在引擎侧仍存在适配问题。

- 本项目
  - **不**依赖协同向量或协同矩阵，直接基于PyTorch与tiny-cuda-nn实现，简洁且易于理解。
  - 提供了**[完整的 UE5 插件项目 - FNTC 分支](https://github.com/JackFishxxx/UnrealEngine/tree/5.5.1-FNTC) **：可将神经纹理作为自定义资源（特征网格 DDS 纹理文件和 `network_data.npz` 里的网络参数）导入 UE5，并在材质系统中直接采样使用。
  - 支持训练、量化、导出全流程，研究与工程实践均可落地。


## Core 算法模块

`model.py` 将特征和编码模块与 decoder 组合起来；`train.py` 配合 ASTC 感知和多阶段模块组织训练流程：

| 目录 | 职责 |
|---|---|
| `Core/Feature_Grid/` | 网格布局、量化/QAT 噪声、边界约束、共享采样和 feature gradients |
| `Core/Positional_Encoding/` | UV 周期性 sin/cos 编码 |
| `Core/Material_Encoding/` | 法线编码、解码和角度误差 |
| `Core/ASTC_Aware/` | 压缩感知训练、latent ASTC 编解码与可微 ASTC |
| `Core/Multi_Stage_Training/` | 表示训练、decoder 拟合和布局重拟合 |

## 安装

这是一个基于Python实现的项目，因此请先安装Python，本项目建议使用 [Anaconda](https://www.anaconda.com/download/success) 进行环境配置。具体安装可参考网上已有教程。

### 依赖

- Python 3.9+
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
pip install numpy Pillow torchvision torchmetrics torchtyping tensorboardX PyYAML ninja
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

从项目根目录执行命令。`configs.py` 自动加载 `config.yaml`；`--config path/to/recipe.yaml` 可指定其他 YAML，CLI 参数覆盖 YAML 值。默认配置启用 Mip0、XY 法线、残差超分、四个 feature gradients 和多阶段训练。`feature_grid_configs` 中每个 grid 单独指定量化与导出精度。

### 训练

对于运行时，可以使用如下的指令

```bash
python train.py \
    --config config.yaml \
    --mode train \
    --data_dir ./data/test \
    --save_dir ./outputs
```

以已完成运行的最终 checkpoint 开始新的单阶段训练时，须沿用生成它的 YAML/CLI 网络结构，并关闭材质投影。此操作只加载模型权重，重新创建优化器和调度器；不会恢复多阶段流程、优化器或 RNG 状态：

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
| `--groups_max_workers` | `1` | 预留参数，目前仅支持顺序执行（`1`） |
| `--groups_verbose` | `False` | 预留参数，当前入口直接打印每组训练日志 |

### 评估

ROMSD 的 LPIPS 按有效标量通道分别计算，再取平均值。

训练时自动记录评估数据。推理时须沿用所选 checkpoint 的网络结构和材质设置；`--load_dir` 指向包含 `models/` 的运行目录，`--load_iter -1` 选择最新已完成 checkpoint：

```bash
python train.py \
    --config config.yaml \
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
- **PSNR / SSIM / LPIPS**：各 Mip 级别的图像质量评估指标

在 `config.yaml` 中设置 `mip0_only: true`（或传入 `--mip0_only`），可将训练采样与 loss、
推理指标与可视化以及 comparison 输入统一限制为全分辨率 Mip0。仓库默认 YAML 为 `true`。多 Mip 训练需使用 `--no-mip0_only --no-multi_stage_training_enable --no-astc_aware_enable`；原生材质分辨率低于 GT 时仍须使用 Mip0。

## 主要特性

### 多纹理联合表示（可配置规范布局）

模型按法线编码方式构造规范布局，通道顺序如下：

```
diffuse(3) | normal(3 或 2) | roughness(1) | occlusion(1) | metallic(1) | specular(1) | displacement(1)
```

默认 `normal_encoding: xy`，因此法线占 2 个网络通道，总规范布局为 10 通道（包含 specular）；
实际仅加载 diffuse、normal、roughness、occlusion、metallic、displacement 时为 9 通道。

数据集统一生成 `texture_packing`，描述 diffuse、normal 和一张 RGBA 标量纹理（按 R/O/M/S/D 顺序紧凑打包，空余分量填 1）；五类标量齐全时，从数据、loss 和指标中移除 displacement。ASTC 基线和压缩 SR base 按同一映射执行，映射及 canonical 通道范围导出到运行目录的 `texture_packing.json`；网络槽位保持固定，可视化仍逐通道输出。输入 `ROMS` 的 alpha 表示 specular，`ROMD` 的 alpha 表示 displacement。
如果只提供部分纹理类型，对应的缺失通道将自动填零，不影响训练和推理。

### 基于 PSNR 的自动早停

训练在 `--early_stop_interval` 检查点比较验证 PSNR；启用 ASTC 对比时使用压缩 FNTC 的 PSNR。连续两个窗口的提升低于阈值时，可早停的阶段才停止；decoder 阶段会先按下文规则降低学习率。

| 参数 | 默认值 | 说明 |
|------|:------:|------|
| `--early_stop` | `True` | 是否启用早停，可用 `--no-early_stop` 关闭 |
| `--early_stop_interval` | `5000` | 计算 PSNR 提升的迭代间隔 |
| `--early_stop_psnr_threshold` | `0.01` | 最小 PSNR 提升阈值（dB） |

warmup 不早停，codec 启动后重置验证窗口。decoder 阶段平台期先将学习率乘 0.25，最低降至 0.0001；达到下限后，连续两个停滞窗口才停止。关闭早停仍执行 LR 调整并跑满步数。阶段结束恢复压缩验证 PSNR 最佳的模型；最终导出编号表示实际完成工作量，`multi_stage_training.json` 的具名阶段记录验证、精修详情和所选 checkpoint。

每次验证刷新最佳分数时原子写入 `models/best_<phase>.pth`；常规最终导出仍在 `models/train_result_<完成迭代>/`，包含恢复后的最佳权重。

### 多阶段训练（Multi-Stage Training）

多阶段训练默认开启（`multi_stage_training_enable`，入口 `python -m Core.Multi_Stage_Training.multi_stage_training`）。当前默认配置依次执行：

1. **学习表示（`representation`）**：使用无隐藏层的仿射 decoder 训练 feature grid，关闭 PE 和 feature gradients。warmup 后优化合法 ASTC 端点与权重参数，学习用于材质重建的压缩表示。
2. **拟合解码器（`decoder`）**：固定已学习的 ASTC 表示，重新初始化一层宽度为 32 的 decoder，启用四个轴向 feature 差分。使用解码后的 feature 和 SR base 训练 decoder，默认最多 50,000 步（`multi_stage_training_decoder_steps`，学习率倍率为 `multi_stage_training_decoder_lr_multiplier`）。
3. **重拟合 latent 与选择块布局（`layout_refit`）**：固定最终 decoder，优化临时 latent，再重新编码并逐 ASTC 块选择合法布局。在第 0 步、每 100 步和末步验证，保留材质及已启用 PBR 的 PSNR/SSIM/LPIPS 均不退化的最佳码流。默认 1,000 步（`multi_stage_training_layout_refit_steps`，0 跳过），不增加推理开销。

布局精修要求单张 RGBA proxy grid、SR、feature gradients，且 GT/材质分辨率比例为整数；不支持的配置会跳过。`multi_stage_training.json` 记录各阶段指标、checkpoint 及 `final_metrics`。

### 量化感知训练（QAT）

训练时使用**量化感知训练**，在前向传播中模拟量化误差，使模型适应推理时的精度损失：

- **噪声退火（Noise Annealing）**：训练初期向 feature grid 特征加入均匀噪声，按 cosine 计划衰减。
  - `--qat_noise_schedule`：`cosine` 或 `none`，默认 `cosine`
  - `--qat_noise_mult_start` / `--qat_noise_mult_end`：默认 `1.0` / `0.25`
  - `--qat_noise_warmup_frac`：默认 `0.1`，前 10% 迭代保持起始强度

- **直通估计器（STE）量化**：前向传播使用量化值，反向传播梯度直接通过，准确模拟推理时的舍入行为，降低颜色偏差。

### ASTC 感知 Latent 训练

训练实现位于 `Core/ASTC_Aware`。

后端为 `astcenc` 和 `astc_differentiable_proxy`。后者使用合法 ASTC 前向解码及近似反向梯度，由 CUDA 执行；仅支持当前名称和 `astc_proxy` checkpoint 格式。

`astcenc` 定期在 CPU 编解码当前采样的 feature block。`astc_differentiable_proxy` 在 warmup 后
每步通过 CUDA 直接训练合法端点/权重，配合经过真实 loss 验证的量化搜索和低频材质投影。
两者复用项目的材质重建/PBR loss，LPIPS 仅用于评测，后端依赖自动处理。当前默认配置为：

```yaml
astc_aware_enable: true
astc_codec_backend: astc_differentiable_proxy
astc_codec_start_frac: 0.1
astc_learning_rate: 0.001
astc_codec_in_loop_interval: 100
astc_codec_update_latent: false
astc_decoder_projection_interval: 500
```

codec 间隔和可选 latent 恒等 STE 仅用于 CPU 后端；投影间隔设为0可关闭材质投影。
ASTC 对比继续复用项目已有的 `astcenc_path`、`astcenc_quality` 和 `astc_block`。

ASTC 图片位于 `Compare_ASTC<block>/iter_<迭代>/`（如 `Compare_ASTC6x6/iter_00001000/`），`comparison_strip_<纹理>.png` 仅包含 FNTC-ASTC、ASTC 和 GT；四张独立图命名为 `<Type>_Uncompressed_FNTC.png`、`<Type>_FNTC_<block>.png`、`<Type>_ASTC_<block>.png`、`<Type>_GT.png`。所有图片直接放在迭代目录，未压缩 FNTC 仍保留在日志和独立图中，不再生成无指标拼图。PBR 的所有图片及 `metrics.json` 统一放在 `Compare_PBRScene/iter_<迭代>/`：`comparison_strip.png`、`comparison_strip_detail.png`、全景独立图 `Scene_FNTC_<block>.png`/`Scene_ASTC_<block>.png`/`Scene_GT.png`，以及带 `detail_` 前缀的特写独立图；未提供迭代号时直接放在 `Compare_PBRScene/`。特写按 `pbr_detail_center`（原始正方形画面中的归一化 x/y）及 `pbr_detail_zoom` 重新渲染，单独计算指标，全景日志口径不变。

代理需要 CUDA 版 PyTorch、匹配的 CUDA toolkit、C++ 编译器和 ninja。它复用
`tools/astc_encoder_source/Source`，源码缺失时下载固定版本到该目录。目前支持量化 Mip0、
6x6 ASTC、四通道 feature grid，且 direct diffuse 必须关闭。材质投影还要求单 grid、无 PE 和 feature gradients 的仿射超分 decoder，也支持 GT 大于原生材质分辨率；其他单阶段配置应设置 `astc_decoder_projection_interval: 0`。

`decoder` 阶段冻结学习后的码流。checkpoint 从 `models/train_result_<迭代>/model.pth` 加载，
`--load_iter -1` 选择最新已完成 checkpoint。

### Feature Gradient 输入

Feature gradient 参数位于独立的 `feature gradient configs` 栏目：

```yaml
feature_gradient_count: 4  # 0 关闭 | 2 右/下 | 4 四轴 | 8 3x3 | 24 5x5
```

`Core/Feature_Grid/feature_gradients.py` 使用双线性 repeat 采样中心与所选偏移位置：2/4/8 方向偏移一个 feature 纹素，24 方向的 5x5 邻域最远偏移两个纹素。邻居减去中心的差分拼接到 decoder 输入。差分不增加
feature 码流，但会增加 decoder 输入和采样开销。checkpoint 和 `network_data.npz`
会记录梯度方向数，推理必须使用相同配置。

多阶段入口在 representation 阶段关闭梯度输入，decoder 阶段按配置启用；单阶段训练时梯度直接
参与表示学习。由于逐纹素材质投影不支持空间差分输入，单阶段使用梯度时应将
`astc_decoder_projection_interval` 设为 `0`。

默认 decoder 使用四个轴向差分、一层宽度为 32 的隐藏层。更大邻域可直接配置，会增加 decoder 输入维度与采样开销。

ASTC 阶段的 feature-gradient 采样使用同一份可微解码 feature；第二阶段使用冻结的解码 feature。
`ASTCAwareTrainer.refine_gradient_codec` 提供显式的完整块邻域码流微调，固定 decoder 并对合法端点/权重候选做材质/PBR 验收；该方法尚未自动插入默认训练流程。


### 无缝平铺（Wrap Boundary Constraint）

特征网格支持**边界平铺约束**：训练时约束所选层级的左右、上下及四角特征，降低边界不连续。采样使用 repeat 寻址，单靠边界约束不能保证最终重建材质完全无缝。

### 特征网格 DDS 导出

每个 `models/train_result_<迭代>/` 包含 `model.pth`、特征网格 **DDS（R8G8B8A8，非 SRGB）**及 `network_data.npz`。代理模型另导出保留学习后码流的 `feature_grid_<index>.astc`。运行目录保存 `resolved_config.json`、`texture_packing.json`；多阶段运行还保存 `representation_config.json` 和 `multi_stage_training.json`。

### 异构多特征网格

通过 `feature_grid_configs` 可以配置多个分辨率或不同精度的 feature grid：

| 字段 | 说明 |
|------|------|
| `max_resolution` | 最大分辨率 |
| `n_levels` | 网格层级数；启用 `mip0_only` 时固定为 1 |
| `quantize_bits` | 量化精度（2/4/8 位） |
| `save_bits` | 每像素特征位数（8/16/32 位，最多 4 通道） |
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

材质重建采用最大 latent 分辨率（不超过 GT，保持宽高比）。1K latent＋512 SR base 对齐 2K GT 时，FNTC 先解码并 clamp 得到 1K 材质；FNTC 和默认 1K ASTC 基线再使用相同的 texel-center bilinear-repeat 放大到 2K。训练通过邻近四个**材质预测**可微插值，直接监督最终 2K 结果，不再在 2K 坐标直接调用 decoder。base 从原始低分辨率 mip 直接采样到 1K，避免 512→2K→1K 的重复插值。

warmup、codec 目标及候选验收、冻结 decoder 拟合、布局重拟合、普通推理和 ASTC/PBR 评测复用同一重建路径。布局选择覆盖整倍数比例（如 1K→2K）下所有受影响的 GT 像素。跨分辨率训练当前要求 `mip0_only: true`；`ref_astc_resolution` 默认跟随 FNTC 材质边长，显式设置可覆盖基线尺寸。日志及 `reconstruction.json` 区分材质/GT 分辨率和原生 decoder 查询数。ASTC 输入及材质投影候选共用材质专用 mip 预滤波，包含法线向量处理。

在 `config.yaml` 中设置 `super_resolution_enable: true` 并指定
`super_resolution_base_resolution` 可启用超分残差训练。该分辨率的原始纹理 mip 作为 base，
经双线性上采样后与 feature grid 特征拼接。同分辨率时残差目标为 `GT - bilinear_base`；跨分辨率时监督放大后的重建材质与 GT。评测、推理和 ASTC
对比均先在原生材质中心计算 `clamp(bilinear_base + predicted_residual, 0, 1)`，再统一放大后与 GT 计算指标。更低 Mip 使用
相同的纹理层级偏移。启用后会自动关闭与残差输出语义冲突的 direct diffuse 模式。
