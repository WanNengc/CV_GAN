# GAN Learning Project

这是一个用于学习生成对抗网络（GAN）的项目，包含了传统的 DCGAN 以及参考了 StyleGAN 思路的升级版 GAN。项目还为这两种 GAN 模型分别提供了 SEFA（Semantic Factorization）脚本，用于潜在空间的因子分析和图像生成。

## 目录

- [项目简介](#项目简介)
- [特性](#特性)
- [目录结构](#目录结构)
- [安装](#安装)
- [使用方法](#使用方法)
  - [传统 DCGAN](#传统-dcgan)
    - [训练 DCGAN](#训练-dcgan)
    - [运行 SEFA 脚本](#运行-sefa-脚本)
  - [升级版 GAN（参考 StyleGAN）](#升级版-gan参考-stylegan)
    - [训练升级版 GAN](#训练升级版-gan)
    - [运行 SEFA 脚本](#运行-sefa-脚本-1)
  - [评估生成器](#评估生成器)
- [SEFA 方法详解](#sefa-方法详解)
- [Adv_SeFa 可视化工具使用指南](#adv_sefa-可视化工具使用指南)
  - [Streamlit 可视化](#streamlit-可视化)
- [贡献](#贡献)


## 项目简介

本项目旨在深入学习和实践生成对抗网络（GAN）的各种架构和技术。项目包含以下主要组件：

1. **传统 DCGAN**：实现了一个简略的深度卷积生成对抗网络，用于生成图像。
2. **升级版 GAN（参考 StyleGAN）**：在传统 DCGAN 的基础上，参考了 StyleGAN 的思路，加入了映射网络和自适应实例归一化（AdaIN）层，以提升生成图像的质量和多样性。
3. **SEFA 脚本**：为 DCGAN 提供了 SEFA（Semantic Factorization）脚本，用于潜在空间的因子分析和图像生成。

## 特性

- **传统 DCGAN**：
  - 基础的生成器和判别器设计
  - 使用 WGAN 损失和梯度惩罚进行训练
  - 支持生成图像的保存和 GIF 动图的制作

- **升级版 GAN**：
  - 引入映射网络（Mapping Network）和自适应实例归一化（AdaIN）层
  - 更高质量的图像生成
  - 支持 W 空间的潜在向量操作

- **SEFA 脚本**：
  - 对生成器的权重进行因子分析
  - 在潜在空间中发现并操控语义方向
  - 提供 Streamlit 可视化工具

## 目录结构

| 目录                   | 描述                                       |
|------------------------|--------------------------------------------|
| `.idea/`               | PyCharm 项目配置文件                        |
| `adv_gif_results/`     | 升级版 GAN 生成的 GIF 动图结果                |
| `adv_results/`         | 升级版 GAN 生成的图像结果                    |
| `adv_snapshots/`       | 升级版 GAN 的模型快照                        |
| `face/face/`           | 图像数据集                                  |
| `gif_results/`         | 传统 DCGAN 生成的 GIF 动图结果                |
| `results/`             | 传统 DCGAN 生成的图像结果                    |
| `sefa_result/`         | SEFA 分析结果                                |
| `snapshots/`           | 传统 DCGAN 的模型快照                        |
| `Adv_SeFa.py`          | SEFA 脚本的 Streamlit 可视化应用            |
| `GAN.py`               | 传统 DCGAN 的训练和生成脚本                   |
| `New_GAN.py`           | 升级版 GAN 的训练和生成脚本                   |
| `evaluate_gan.py`      | 评估生成器性能的脚本                          |
| `sefa.py`              | 传统 DCGAN 的 SEFA 分析脚本                   |
| `requirements.txt`     | 项目的依赖包列表                              |
| `README.md`            | 项目说明文档                                  |

## 安装

### 环境要求

- Python 3.7+
- PyTorch 2.5.1+
- 其他依赖见 `requirements.txt`

### 安装

1. **安装依赖**
    ```bash
    pip install -r requirements.txt
    ```


## 使用方法

### 传统 DCGAN

#### 训练 DCGAN

1. **准备数据集**

    将你的图像数据集放在 `face/` 目录下，确保数据集按照 PyTorch `ImageFolder` 的格式组织。

2. **运行训练脚本**

    ```bash
    python GAN.py
    ```

    训练过程中，生成的图像将保存在 `results/` 目录，训练模型的权重保存在 `snapshots/` 目录，生成的 GIF 动图保存在 `gif_results/` 目录。

#### 运行 SEFA 脚本

1. **确保已训练 DCGAN 并保存了生成器权重**

    默认情况下，生成器权重保存在 `snapshots/gnet.pth`。

2. **运行 SEFA 脚本**

    ```bash
    python sefa.py
    ```

    SEFA 脚本将对生成器的权重进行因子分析，并生成 HTML 可视化结果，保存在 `sefa_result/` 目录。

### 升级版 GAN（参考 StyleGAN）

#### 训练升级版 GAN

1. **准备数据集**

    将你的图像数据集放在 `face/` 目录下，确保数据集按照 PyTorch `ImageFolder` 的格式组织。

2. **运行训练脚本**

    ```bash
    python New_GAN.py
    ```

    训练过程中，生成的图像将保存在 `adv_results/` 目录，训练模型的权重保存在 `adv_snapshots/` 目录，生成的 GIF 动图保存在 `adv_gif_results/` 目录。

#### 运行 SEFA 脚本

##### Streamlit 可视化

```bash
streamlit run Adv_SeFa.py
```
打开浏览器访问提示的本地地址，即可通过交互式界面进行 SEFA 分析和图像生成。

### 评估生成器

本项目提供了脚本 `evaluate_gan.py` 用于评估多个训练完成的 GAN 模型，计算 Inception Score (IS), Frechet Inception Distance (FID) 以及 Kernel Inception Distance (KID)。

#### 评估指标解释

- **Inception Score (IS)**：
  - 衡量生成图像的质量和多样性。
  - 分数越高，表示生成的图像在类别上的多样性越高，质量越好。

- **Frechet Inception Distance (FID)**：
  - 衡量生成图像与真实图像之间的距离。
  - 分数越低越好，表示生成的图像分布越接近真实图像分布。

- **Kernel Inception Distance (KID)**：
  - 也是衡量生成图像质量的指标，类似于 FID。
  - 分数越低越好，表示生成图像与真实图像在特征空间中的距离越小。

#### 运行评估脚本

1. **确保已训练并保存生成器权重**

    生成器权重应保存在 `snapshots/` 或 `adv_snapshots/` 目录下。

2. **运行评估脚本**

    ```bash
    python evaluate_gan.py \
        --generators "GAN.Gnet:snapshots/gnet.pth:generated_images_gan" \
                     "New_GAN.AdvancedGnet:adv_snapshots/adv_gnet.pth:generated_images_new_gan" \
        --real_images face/face/ \
        --num_images 1000 \
        --batch_size 32 \
        --device cuda
    ```

    **参数说明**：
    
    - `--generators`: 生成器的规范列表，格式为 `"module.class:path_to_weights:generated_dir"`。
      - `module.class`: 生成器类的模块和类名，如 `GAN.Gnet` 或 `New_GAN.AdvancedGnet`。
      - `path_to_weights`: 生成器权重文件的路径。
      - `generated_dir`: 生成的图像保存的目录。
    - `--real_images`: 真实图像的文件夹路径。
    - `--num_images`: 要生成用于评估的图像数量。
    - `--batch_size`: 批量大小。
    - `--device`: 使用的设备（`cuda` 或 `cpu`）。

    **示例**：
    
    ```bash
    python evaluate_gan.py --generators "GAN.Gnet:snapshots/gnet.pth:generated_images_gan" "New_GAN.AdvancedGnet:adv_snapshots/adv_gnet.pth:generated_images_new_gan" --real_images face/face/ --num_images 1000 --batch_size 32 --device cuda
    ```

    脚本将生成指定数量的图像，并保存在 `generated_images_gan` 和 `generated_images_new_gan` 目录下，然后计算 Inception Score、FID 和 KID，并输出评估结果。

## SEFA 方法详解

### 什么是 SEFA？

**SEFA**（Semantic Factorization）是一种用于分析和操控生成对抗网络（GAN）潜在空间的方法。SEFA 通过因子分析技术，识别出潜在空间中的主要因子（因素），从而实现对生成图像的语义级别操控。该方法由 [相关论文](https://arxiv.org/abs/2007.06600) 在 2020 年提出，旨在增强 GAN 模型的可解释性和可控性。

### SEFA 的核心思想

1. **潜在空间分析**：
   SEFA 分析 GAN 的潜在空间，尤其是映射后的 `w` 空间，通过因子分析识别出潜在的语义因子。这些因子代表了生成图像中不同的语义属性，如面部表情、姿势、背景等。

2. **因子操控**：
   识别出的因子可以用于操控生成器的输出。例如，通过调整特定因子的权重，可以在生成图像中增加或减少某种特征，如笑容的幅度或头部的倾斜角度。

3. **可视化**：
   SEFA 提供了可视化工具，帮助用户直观地理解和操作这些因子。通过交互式界面，用户可以实时调整因子的权重并观察生成图像的变化，从而深入理解 GAN 的潜在空间结构。

### SEFA 的优点

- **可解释性**：
  SEFA 提供了一种系统的方法来理解 GAN 潜在空间中的不同维度及其对应的语义意义。

- **可控性**：
  通过因子分析，用户可以精确地操控生成图像的特定属性，实现更高水平的图像生成控制。

- **易用性**：
  结合 Streamlit 等可视化工具，SEFA 使得复杂的潜在空间操作变得直观和易于使用，即使对于非专业用户也能轻松上手。

### SEFA 的应用

在本项目中，SEFA 被应用于：

- **潜在向量分析**：
  收集大量的潜在向量，并通过因子分析提取主要因子，揭示潜在空间中的关键语义维度。

- **语义操控**：
  通过调整潜在向量的方向，实现对生成图像的语义级别控制，如调整面部表情、姿势或其他特征。

- **结果可视化**：
  提供交互式工具，帮助用户理解和应用这些因子，实时观察因子调整对生成图像的影响。

## Adv_SeFa 可视化工具使用指南

**Adv_SeFa** 提供了基于 **Streamlit** 的交互式可视化工具，用于展示和操控 SEFA 分析的结果。以下是如何使用该工具的详细说明。

### Streamlit 可视化   

**Streamlit** 是一个用于构建数据应用的 Python 库，允许您快速创建交互式网页应用。

#### 运行 Streamlit 应用

1. **确保已安装 Streamlit**

   如果尚未安装，可以通过 `requirements.txt` 安装，或单独安装：

   ```bash
   pip install streamlit
   ```
2. **运行 SEFA 脚本**

   在项目根目录下，运行以下命令：

   ```bash
   streamlit run Adv_SeFa.py
   ```
3. **访问应用**

    命令运行后，终端会显示一个本地地址（如 http://localhost:8501）。打开浏览器，访问该地址即可使用 SEFA 可视化工具
#### 使用 Streamlit 应用

应用界面主要包括以下部分：

1. **参数配置**（侧边栏）：
   - **Number of factors to analyze**：选择要分析的因子数量。
   - **Number of top factors to visualize**：选择要可视化的前几个因子。
   - **Adjustment magnitude**：调整因子方向的幅度。

2. **主界面**：
   - **设备信息**：显示使用的设备（如 CUDA 或 CPU）。
   - **进度指示器**：显示潜在向量收集和因子分析的进度。
   - **生成的图像展示**：展示操控后的图像及因子的影响。
   - **下载结果**：提供按钮，允许用户下载生成的 SEFA 分析结果。

3. **操作步骤**：

   - **调整因子**：通过滑块调整各个因子的方向和幅度。
   - **生成图像**：根据调整的因子生成相应的图像。
   - **查看和下载结果**：查看可视化的图像，并可选择下载分析结果。

#### 示例操作

1. **启动应用**：
   - 运行 `streamlit run Adv_SeFa.py`。
   - 打开浏览器访问提供的本地地址。

2. **配置参数**：
   - 在侧边栏调整 **Number of factors to analyze**、**Number of top factors to visualize** 和 **Adjustment magnitude**。

3. **执行分析**：
   - 应用将自动开始收集潜在向量、执行因子分析，并展示前 `top_k` 个因子的影响。

4. **查看结果**：
   - 在主界面查看操控后的图像，理解不同因子对生成图像的影响。

5. **下载结果**：
   - 点击 **Download Results** 按钮，将生成的 SEFA 分析结果下载为压缩文件。

### 注意事项

- **性能要求**：
  - SEFA 分析和图像生成过程可能需要较高的计算资源，建议在具备 GPU 支持的环境中运行，以加快处理速度。

- **数据准备**：
  - 确保在运行 SEFA 脚本前，已成功训练 GAN 模型，并生成了相应的权重文件。

- **文件路径**：
  - 确保配置类中的路径（如 `adv_g_net_path`）正确指向预训练模型和结果保存目录。

## 贡献


---

感谢您的关注与支持！希望这个项目能帮助您更好地理解和掌握生成对抗网络（GAN）的各种技术和应用。
