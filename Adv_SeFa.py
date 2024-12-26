# Adv_SeFa.py

import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
from sklearn.decomposition import FactorAnalysis
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from tqdm import tqdm
import streamlit as st
from New_GAN import AdvancedGnet  # 从 New_GAN.py 导入 AdvancedGnet 类

# -------------------------------1. 配置类--------------------------------- #
class AdvancedConfig:
    adv_result_save_path = 'adv_results/'          # 生成图像保存的路径（不影响 SEFA）
    adv_g_net_path = 'adv_snapshots/adv_gnet.pth' # 生成网络权重文件保存的路径
    adv_noise_dim = 100                            # 噪声向量维度
    w_dim = 128                                    # W空间维度
    feats_channel = 64                             # 特征图维度

# -------------------------------2. SEFA实现--------------------------------- #
class SEFA:
    def __init__(self, generator, device='cuda'):
        self.generator = generator
        self.device = device
        self.generator.to(self.device)
        self.generator.eval()

    def collect_latent_vectors(self, num_samples, noise_dim, batch_size=64):
        """收集潜在向量 z 和映射后的 w 向量"""
        z_vectors = []
        w_vectors = []
        num_batches = num_samples // batch_size

        with torch.no_grad():
            for _ in tqdm(range(num_batches), desc="Collecting latent vectors"):
                z = torch.randn(batch_size, noise_dim, device=self.device)
                w = self.generator.mapping_network(z)
                z_vectors.append(z.cpu().numpy())
                w_vectors.append(w.cpu().numpy())

        z_vectors = np.concatenate(z_vectors, axis=0)
        w_vectors = np.concatenate(w_vectors, axis=0)
        return z_vectors, w_vectors

    def perform_factor_analysis(self, w_vectors, n_factors=50):
        """对 w 向量进行因子分析"""
        fa = FactorAnalysis(n_components=n_factors, random_state=42)
        fa.fit(w_vectors)
        factors = fa.components_  # 因子矩阵
        return fa, factors

    def find_z_direction(self, fa_model, factor_idx, noise_dim=100, num_samples=1000):
        """
        寻找与指定因子最相关的 z 方向。
        通过线性回归拟合因子变化与 z 的关系。
        """
        # 随机生成一组 z 向量
        z = torch.randn(num_samples, noise_dim, device=self.device)
        with torch.no_grad():
            w = self.generator.mapping_network(z).cpu().numpy()

        # 获取指定因子的系数
        factor = fa_model.components_[factor_idx]

        # 回归：w 与 z 之间的关系
        # 由于w = f(z), 但f是非线性的，我们只能做线性近似
        # 将 factor 与 z 进行线性回归，找到 z 的方向使得 w 在 factor 方向上变化最大

        # 使用最小二乘法拟合
        reg = LinearRegression()
        reg.fit(z.cpu().numpy(), w[:, factor_idx])

        # 线性回归的系数作为 z 的方向
        z_direction = reg.coef_

        # 归一化方向向量
        z_direction = z_direction / np.linalg.norm(z_direction)

        return z_direction

    def manipulate_z(self, z_direction, direction_magnitude=3.0, num_images=8, noise_dim=100):
        """沿着指定的 z 方向调整 z 向量并生成图像"""
        self.generator.eval()
        manipulated_images = []
        with torch.no_grad():
            # 生成基准 z 向量
            z = torch.randn(num_images, noise_dim, device=self.device)
            # 调整 z 向量
            z = z.cpu().numpy()
            z += direction_magnitude * z_direction
            z = torch.from_numpy(z).to(self.device).float()
            # 生成图像
            manipulated_imgs = self.generator(z)
            manipulated_images = manipulated_imgs.cpu()
        return manipulated_images

    def visualize_factors(self, fa_model, n_factors=5, direction=3.0, noise_dim=100, num_images=8):
        """寻找因子方向并返回对应的图像"""
        images = []
        for factor_idx in range(n_factors):
            print(f"Processing Factor {factor_idx}")
            # 寻找与因子最相关的 z 方向
            z_direction = self.find_z_direction(fa_model, factor_idx, noise_dim=noise_dim)
            # 沿着 z 方向调整 z 并生成图像
            imgs = self.manipulate_z(z_direction, direction_magnitude=direction, num_images=num_images, noise_dim=noise_dim)
            images.append(imgs)
        return images

# -------------------------------3. Streamlit 应用--------------------------------- #
def main():
    """Main function for Streamlit app."""
    st.title('SEFA: StyleGAN Embedding Factor Analysis')
    st.sidebar.title('SEFA Options')

    # -------------------------------参数配置--------------------------------- #
    config = AdvancedConfig()
    model_path = config.adv_g_net_path            # 预训练生成器模型的路径
    noise_dim = config.adv_noise_dim              # 输入噪声向量的维度
    w_dim = config.w_dim                          # W 空间的维度
    feats_channel = config.feats_channel          # 生成器中的特征图维度
    num_samples = 10000                           # 用于因子分析的潜在向量数量
    n_factors = st.sidebar.slider('Number of factors to analyze', min_value=1, max_value=50, value=10, step=1)
    top_k = st.sidebar.slider('Number of top factors to visualize', min_value=1, max_value=10, value=5, step=1)
    direction = st.sidebar.slider('Adjustment magnitude', min_value=0.0, max_value=10.0, value=3.0, step=0.1)

    save_dir = 'sefa_results'
    os.makedirs(save_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    st.write(f"Using device: {device}")

    # 检查模型路径是否存在
    if not os.path.exists(model_path):
        st.error(f"Error: Model path '{model_path}' does not exist.")
        return

    # 初始化生成器并加载预训练权重
    generator = AdvancedGnet(opt=config).to(device)
    try:
        generator.load_state_dict(torch.load(model_path, map_location=device))
        st.write(f"Loaded generator model from '{model_path}'.")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return

    # 初始化 SEFA 对象
    sefa = SEFA(generator, device=device)

    # 收集潜在向量和 w 向量
    with st.spinner('Collecting latent vectors...'):
        z, w = sefa.collect_latent_vectors(num_samples=num_samples, noise_dim=noise_dim)
    st.write(f"Collected {w.shape[0]} w vectors.")

    # 执行因子分析
    with st.spinner('Performing factor analysis...'):
        fa_model, factors = sefa.perform_factor_analysis(w, n_factors=n_factors)
    st.write(f"Performed factor analysis with {n_factors} factors.")

    # 可视化前 top_k 个因子的效果
    st.sidebar.header('Adjust Semantic Factors')
    factor_steps = {}
    for factor_idx in range(top_k):
        eigen_value = np.var(w @ factors[factor_idx].T)
        step = st.sidebar.slider(
            f'Factor {factor_idx+1} (Eigenvalue: {eigen_value:.2f})',
            min_value=-10.0,
            max_value=10.0,
            value=0.0,
            step=0.1
        )
        factor_steps[factor_idx] = step

    # 生成并显示图像
    with st.spinner('Generating manipulated images...'):
        # 初始化 z 向量
        z_init = torch.randn(1, noise_dim, device=device).cpu().numpy()

        # 调整 z 向量
        z_adjusted = z_init.copy()
        for factor_idx, step in factor_steps.items():
            z_direction = sefa.find_z_direction(fa_model, factor_idx, noise_dim=noise_dim, num_samples=1000)
            z_adjusted += step * z_direction

        # 生成图像
        z_tensor = torch.from_numpy(z_adjusted).to(device).float()
        with torch.no_grad():
            generated_img = generator(z_tensor).cpu()
        img = torchvision.utils.make_grid(generated_img, normalize=True).permute(1, 2, 0).numpy()

    st.image(img, caption='Manipulated Image', use_container_width=True)

    # 生成图像并显示所有 top_k 因子的影响
    with st.spinner('Generating images for all factors...'):
        images = sefa.visualize_factors(fa_model, n_factors=top_k, direction=direction, noise_dim=noise_dim, num_images=8)
        for idx, imgs in enumerate(images):
            grid = torchvision.utils.make_grid(imgs, nrow=4, normalize=True)
            img_np = grid.permute(1, 2, 0).numpy()
            st.image(img_np, caption=f'Factor {idx+1} Adjustment', use_container_width=True)

    # 提供下载链接
    if st.button('Download Results'):
        import zipfile
        import io

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
            for root, dirs, files in os.walk(save_dir):
                for file in files:
                    zip_file.write(os.path.join(root, file), arcname=file)
        st.download_button(
            label="Download SEFA Results",
            data=zip_buffer.getvalue(),
            file_name='sefa_results.zip',
            mime='application/zip'
        )

if __name__ == "__main__":
    main()
