# NeuralTexture 训练代码

该项目用于训练神经纹理（Neural Texture）模型，代码基于 [PyTorch](https://pytorch.org/get-started/locally/) 与 [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) 实现，支持可选的量化感知训练与模型导出等功能。同时，该项目也提供了基于纯 Pytorch 实现的版本，无需 tiny-cuda-nn 等库。

## 什么是神经纹理?

神经纹理（Neural Texture, NTC）是一种基于神经网络的纹理表示与压缩方法。与传统的 PNG、JPEG、ASTC 等压缩方式不同，神经纹理使用**特征网格（Feature Grid）** 与**小型神经网络（MLP）** 相结合，将高分辨率的纹理信息以更紧凑的形式存储和还原，同时支持纹理对于图像随机读取的要求。其优势包括：  

- **高压缩率**：在相同或更低的存储开销下保留更多细节。  
- **可扩展性**：支持多分辨率、LOD，以及与材质参数的联合表示。  
- **强适用性**：不仅适用于颜色贴图，还可以用于法线、粗糙度等材质纹理的表示。  

### 与 NVIDIA NTC SDK 的区别

NVIDIA在2025年2月开源了 [RTXNTC SDK](https://github.com/NVIDIA-RTX/Rtxntc)，提供了NVIDIA官方对于神经纹理的支持。但本项目与RTXNTC仍有一定差别：

- NTC SDK  
  - 需要使用**协同向量 (CoopVec)** 、 **协同矩阵 (CoopMat)** 等新硬件特性来实现推理加速。 
  - 目前在引擎侧仍存在适配问题。  

- 本项目  
  - **不**依赖协同向量或协同矩阵，直接基于PyTorch与tiny-cuda-nn实现，简洁且易于理解。  
  - 提供了**[完整的 UE5 插件项目 - FNTC 分支](https://github.com/JackFishxxx/UnrealEngine/tree/5.5.1-FNTC) **：可将神经纹理作为自定义资源（特征网格 DDS 纹理文件和 `network_data.npz` 里的网络参数）导入 UE5，并在材质系统中直接采样使用。
  - 支持训练、量化、导出全流程，研究与工程实践均可落地。  


## 安装

这是一个基于Python实现的项目，因此请先安装Python，本项目建议使用 [Anaconda](https://www.anaconda.com/download/success) 进行环境配置。具体安装可参考网上已有教程。

### 依赖

- Python 3.9+
- PyTorch （与本机的CUDA版本匹配）
- tiny-cuda-nn（可选，但强烈建议）
- 其他依赖：`numpy`、`Pillow`、`torchvision`、`torchmetrics`、`torchtyping`、`tensorboardX`

### 设备要求

该项目训练的神经纹理在训练和推理时本身并不需要特定的设备，但为了加速到可接受的程度，使用了 tiny-cuda-nn 进行加速，而该项目要求使用 NVIDIA 显卡进行训练。因此，设备要求参考如下：

- 需要一块 NVIDIA GPU，如有 Tensor Cores 可以提升性能。

- 需要一款支持 C++14 的编译器，推荐并已测试的选择如下：

  - Windows：Visual Studio 2019 或 2022

  - Linux：GCC/G++ 8 或更高版本

- 需要一个较新的 CUDA 版本，推荐并已测试的选择如下：

  - Windows：CUDA 11.5 或更高版本

  - Linux：CUDA 10.2 或更高版本

- 需要 CMake v3.21 或更高版本。

另外，该项目理论上可以在Linux系统上进行大批量的纹理压缩，只需要将对应压缩后的神经纹理文件放到UE5内使用即可。

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
pip install numpy Pillow torchvision torchmetrics torchtyping tensorboardX
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
| normal（法线） | 3 | `normal`、`nor_gl` |
| roughness（粗糙度） | 1 | `roughness`、`rough` |
| occlusion（环境光遮蔽） | 1 | `occlusion`、`ao`、`ambient` |
| metallic（金属度） | 1 | `metallic`、`metalness` |
| specular（高光） | 1 | `specular` |
| displacement（位移） | 1 | `displacement`、`disp` |

> **提示**：模型输出固定为 **11 通道**（diffuse×3 + normal×3 + roughness×1 + occlusion×1 + metallic×1 + specular×1 + displacement×1），缺失的纹理通道将自动填零，无需提供全部类型。不同分辨率的纹理会被自动缩放至与第一张纹理相同的分辨率。

具体命名、识别和通道数量可参考 `dataset.py` 中的 `get_texture_config()` 函数。


## 运行

项目的具体控制变量可参考 `configs.py` 内的变量与注释。 

### 训练

对于运行时，可以使用如下的指令

```bash
python train.py \
    --mode train \
    --data_dir ./data/test \
    --save_dir ./outputs \
    --quantize True \
    --quantize_bits 4 \
    --save_bits 32 
```

如果想要重新训练已经训练了一段时间的模型，可以使用 `--load_iter` 和  `--load_dir` 指令：

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
| `--groups_max_workers` | `1` | 并发训练的 worker 数量（默认顺序执行） |
| `--groups_verbose` | `False` | 是否打印每组的详细训练日志 |

### 评估

训练时，会自动输出一些评估数据。而当想对已训练完毕的模型重启训练对于评估时，可以使用如下的指令

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

### 运行Json文件

本项目提供了基于vscode的 `launch.json` 文件，便于使用。

## 监控

该项目提供了运行时指标监控等功能。

### Tensorboard

开启日志监控，评估收敛情况与模型效果：

```bash
tensorboard --logdir ./outputs/tensorboard
```

训练过程中会自动记录以下指标：

- **Loss/train**：训练损失
- **PSNR / SSIM / LPIPS**：各 LOD 级别的图像质量评估指标

## 主要特性

### 多纹理联合表示（11 通道规范布局）

模型采用固定的 **11 通道规范布局**，支持同时表示多种材质纹理，通道顺序如下：

```
diffuse(3) | normal(3) | roughness(1) | occlusion(1) | metallic(1) | specular(1) | displacement(1)
```

如果只提供部分纹理类型（例如仅提供 albedo + normal），对应的缺失通道将自动填零，不影响训练和推理。

### 基于 PSNR 的自动早停

训练过程支持**基于 PSNR 的自动早停**，避免过度训练浪费时间。当连续一段时间内（`--early_stop_interval` 次迭代）的平均 PSNR 提升低于设定阈值时，训练将自动保存模型并停止。

| 参数 | 默认值 | 说明 |
|------|:------:|------|
| `--early_stop` | `True` | 是否启用早停 |
| `--early_stop_interval` | `5000` | 计算 PSNR 提升的迭代间隔 |
| `--early_stop_psnr_threshold` | `0.01` | 最小 PSNR 提升阈值（dB） |

### 量化感知训练（QAT）

训练时使用**量化感知训练**，在前向传播中模拟量化误差，使模型适应推理时的精度损失：

- **噪声退火（Noise Annealing）**：训练初期向特征值加入均匀噪声，随训练进度线性衰减至零，增强模型对量化误差的鲁棒性。
  - `--noise_std`：噪声强度（默认 `1.0`）
  - `--noise_anneal_fraction`：噪声衰减的训练比例（默认 `0.8`，即在前 80% 的训练迭代中完成退火）

- **直通估计器（STE）量化**：前向传播使用量化值，反向传播梯度直接通过，准确模拟推理时的舍入行为，降低颜色偏差。

### 无缝平铺（Wrap Boundary Constraint）

特征网格支持**无缝平铺约束**：训练过程中自动同步网格左右、上下及四角的边界特征，使得特征纹理在 Wrap/Repeat 采样模式下不产生接缝，适合需要平铺的材质。

### 特征网格 DDS 导出

模型保存时，特征网格以 **DDS（R8G8B8A8，非 SRGB）格式**导出，方便在 UE5 等游戏引擎中直接加载；网络权重保存为 `.npz` 文件。

### 异构多哈希网格

模型支持**异构多哈希网格**，可在 `configs.py` 的 `hash_grid_configs` 字段中为每个网格独立配置：

| 字段 | 说明 |
|------|------|
| `max_resolution` | 最大分辨率 |
| `n_levels` | LOD 层级数 |
| `quantize_bits` | 量化精度（2/4/8/16 位） |
| `save_bits` | 保存精度（8/16/32/64 位） |
| `learning_rate` | 独立学习率 |

### 可配置的纹理损失权重

可在 `configs.py` 的 `texture_loss_weights` 字段中为每种纹理类型设置独立的损失权重，以平衡不同纹理通道对训练的贡献（如颜色贴图通常比粗糙度更重要）。