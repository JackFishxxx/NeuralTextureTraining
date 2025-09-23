# NeuralTexture 训练代码

该项目用于训练神经纹理（Neural Texture）模型，代码基于 [PyTorch](https://pytorch.org/get-started/locally/) 与 [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) 实现，支持可选的量化训练与模型导出等功能。同时，该项目也提供了基于纯 Pytorch 实现的版本，无需 tiny-cuda-nn 等库。

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
  - 提供了**完整的 UE5 插件适配实践**：可将神经纹理作为自定义资源（类似 PNG/JPG）导入 UE，并在材质系统中直接采样使用。  
  - 支持训练、量化、导出全流程，研究与工程实践均可落地。  


## 安装

这是一个基于Python实现的项目，因此请先安装Python，本项目建议使用 [Anaconda](https://www.anaconda.com/download/success) 进行环境配置。具体安装可参考网上已有教程。

### 依赖

- Python 3.9+
- PyTorch （与本机的CUDA版本匹配）
- tiny-cuda-nn（可选，但强烈建议）
- 其他依赖：`numpy`、`tensorboard`

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
pip install -r requirements.txt
```

## 数据

我们在 `data` 文件夹中提供了一份示例数据。亦或者，你也可以使用自己的数据。

格式如下：

```
data
└── CustomTexture
    ├── CustomTexture_Albedo.png
    ├── CustomTexture_Normal.png
    ├── ...
```

你需要将纹理存放在对应的文件夹内，然后按照命名格式给纹理的对应部分添加后缀。代码会自动识别存在哪些纹理，具体命名、识别和通道数量可参考`config.py`文件。


## 运行

项目的具体控制变量可参考 `config.py` 内的变量与注释。 

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