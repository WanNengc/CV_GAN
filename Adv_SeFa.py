# -------------------------------1. 参数配置--------------------------------- #
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
                z_vectors.append(z.cpu())
                w_vectors.append(w.cpu())

        z_vectors = torch.cat(z_vectors, dim=0)
        w_vectors = torch.cat(w_vectors, dim=0)
        return z_vectors, w_vectors

    def perform_factor_analysis(self, w_vectors, n_factors=50):
        """对 w 向量进行因子分析"""
        fa = FactorAnalysis(n_components=n_factors, random_state=42)
        fa.fit(w_vectors.numpy())
        factors = fa.components_  # 因子矩阵
        return fa, factors

    def find_z_directions(self, fa_model, factors, noise_dim=100, num_samples=1000):
        """
        寻找与因子最相关的 z 方向。
        通过线性回归拟合因子变化与 z 的关系。
        """
        z_directions = []
        for factor_idx in range(factors.shape[0]):
            # 随机生成一组 z 向量
            z = torch.randn(num_samples, noise_dim, device=self.device)
            with torch.no_grad():
                w = self.generator.mapping_network(z).cpu().numpy()

            # 获取指定因子的系数
            factor = fa_model.components_[factor_idx]

            # 回归：w 与 z 之间的关系
            reg = LinearRegression()
            reg.fit(z.cpu().numpy(), w[:, factor_idx])

            # 线性回归的系数作为 z 的方向
            z_direction = reg.coef_

            # 归一化方向向量
            z_direction = z_direction / np.linalg.norm(z_direction)
            z_direction = torch.from_numpy(z_direction).to(self.device).float()

            z_directions.append(z_direction)

        return z_directions

# -------------------------------3. Streamlit 应用--------------------------------- #
def main():
    """Main function for Streamlit app."""
    st.title('SEFA: StyleGAN Embedding Factor Analysis')
    st.sidebar.title('SEFA Options')

    # -------------------------------参数配置--------------------------------- #
    class AdvancedConfig:
        adv_result_save_path = 'adv_results/'          # 生成图像保存的路径（不影响 SEFA）
        adv_g_net_path = 'adv_snapshots/adv_gnet.pth' # 生成网络权重文件保存的路径
        adv_noise_dim = 100                            # 噪声向量维度
        w_dim = 128                                    # W空间维度
        feats_channel = 64                             # 特征图维度

    config = AdvancedConfig()
    model_path = config.adv_g_net_path            # 预训练生成器模型的路径
    noise_dim = config.adv_noise_dim              # 输入噪声向量的维度
    w_dim = config.w_dim                          # W 空间的维度
    feats_channel = config.feats_channel          # 生成器中的特征图维度
    num_samples = 10000                           # 用于因子分析的潜在向量数量

    # 滑条参数
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

    # 初始化 Session State
    if 'z_fixed' not in st.session_state:
        z_fixed = torch.randn(1, noise_dim, device=device)
        st.session_state.z_fixed = z_fixed
    else:
        z_fixed = st.session_state.z_fixed

    if 'factor_directions' not in st.session_state:
        with st.spinner('Collecting latent vectors and performing factor analysis...'):
            z, w = sefa.collect_latent_vectors(num_samples=num_samples, noise_dim=noise_dim)
            fa_model, factors = sefa.perform_factor_analysis(w, n_factors=n_factors)
            factor_directions = sefa.find_z_directions(fa_model, factors, noise_dim=noise_dim, num_samples=1000)[:top_k]
            st.session_state.factor_directions = factor_directions
            st.session_state.fa_model = fa_model
            st.session_state.factors = factors
        st.write(f"Collected {w.shape[0]} w vectors and performed factor analysis.")
    else:
        factor_directions = st.session_state.factor_directions
        fa_model = st.session_state.fa_model
        factors = st.session_state.factors

    # -------------------------------调整语义因子滑条--------------------------------- #
    st.sidebar.header('Adjust Semantic Factors')
    factor_steps = {}
    for factor_idx in range(top_k):
        eigen_value = fa_model.components_[factor_idx].var()
        step = st.sidebar.slider(
            f'Factor {factor_idx+1} (Eigenvalue: {eigen_value:.2f})',
            min_value=-10.0,
            max_value=10.0,
            value=0.0,
            step=0.1
        )
        factor_steps[factor_idx] = step

    # -------------------------------生成混合图像--------------------------------- #
    with st.spinner('Generating mixed image with all factor adjustments...'):
        z_mixed = z_fixed.clone()
        for factor_idx, step in factor_steps.items():
            z_mixed += step * factor_directions[factor_idx].view(1, -1)
        with torch.no_grad():
            generated_mixed = generator(z_mixed).cpu()
        img_mixed = torchvision.utils.make_grid(generated_mixed, normalize=True).permute(1, 2, 0).numpy()
    st.image(img_mixed, caption='Mixed Adjusted Image', use_container_width=True)

    # -------------------------------生成单独因子调整图像--------------------------------- #
    st.header('Individual Factor Adjustments')
    num_cols = min(top_k, 4)  # 根据因子数量动态生成列，最多4列
    cols = st.columns(num_cols)
    for idx, (factor_idx, step) in enumerate(factor_steps.items()):
        with cols[idx % num_cols]:
            if step != 0.0:
                z_adjusted = z_fixed.clone() + step * factor_directions[factor_idx].view(1, -1)
                with torch.no_grad():
                    generated_adjusted = generator(z_adjusted).cpu()
                img_adjusted = torchvision.utils.make_grid(generated_adjusted, normalize=True).permute(1, 2, 0).numpy()
                st.image(img_adjusted, caption=f'Factor {factor_idx+1} Adjustment (Step: {step})', use_container_width=True)
            else:
                # 如果步长为0，则显示固定图像
                with torch.no_grad():
                    generated_fixed = generator(z_fixed).cpu()
                img_fixed = torchvision.utils.make_grid(generated_fixed, normalize=True).permute(1, 2, 0).numpy()
                st.image(img_fixed, caption=f'Factor {factor_idx+1} Adjustment (No Change)', use_container_width=True)

if __name__ == "__main__":
    main()
