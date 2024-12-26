"""
evaluate_gan.py

用来评估多个训练完成的 GAN 模型，计算 Inception Score (IS), Frechet Inception Distance (FID)
以及 Kernel Inception Distance (KID)。请根据项目需求进行修改。
"""

import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
import numpy as np
from scipy.stats import entropy
from pytorch_fid import fid_score
from PIL import Image
from tqdm import tqdm
import argparse
import torchvision.utils as vutils

# ======== 1. 导入配置  ========
from GAN import opt  # 基础配置
from New_GAN import adv_opt  # 高级配置


# =========================== InceptionDataset ============================

class InceptionDataset(Dataset):
    """
    如果 self.images 是一个 4D Tensor (N, 3, H, W)，
    那么 transform 应该能处理 Tensor；
    如果 self.images 是一批 PIL Image，则 transform 应该能处理 PIL Image。
    """

    def __init__(self, images, transform=None):
        self.images = images  # 可能是 Tensor，也可能是 list (PIL)
        self.transform = transform

    def __len__(self):
        if isinstance(self.images, torch.Tensor):
            return self.images.size(0)
        else:
            return len(self.images)

    def __getitem__(self, idx):
        # 当 self.images 是一个 Tensor
        if isinstance(self.images, torch.Tensor):
            img = self.images[idx]
            # 如果需要对 Tensor 进行进一步变换（如 Normalize），transform 必须能处理 Tensor
            if self.transform:
                img = self.transform(img)
            return img

        # 当 self.images 是一个 list (PIL Image)
        else:
            img = self.images[idx]
            if self.transform:
                img = self.transform(img)
            return img


# =========================== Inception Score 实现 ============================

def calculate_inception_score(imgs, batch_size=32, resize=True, splits=10):
    """
    计算生成图像的 Inception Score
    imgs: torch.Tensor, 形状 (N, 3, H, W)，值范围 [0,1]
    """
    N = len(imgs)
    # 允许 N >= batch_size
    assert N >= batch_size, f"Number of images (N={N}) must be >= batch_size (batch_size={batch_size})."

    # 加载 Inception v3 模型
    inception_model = models.inception_v3(pretrained=True, transform_input=False).eval()
    if torch.cuda.is_available():
        inception_model = inception_model.cuda()

    # 预处理
    if resize:
        # 注意，这里对 Tensor 做 transforms.Resize 可能需要新的 torchvision 版本
        # 旧版只支持 PIL，这里仅作示例。如果不兼容，可改用 torch.nn.functional.interpolate 等。
        transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    dataset = InceptionDataset(imgs, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    preds = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Calculating Inception Score"):
            if torch.cuda.is_available():
                batch = batch.cuda()
            preds_batch = inception_model(batch)
            preds_batch = nn.Softmax(dim=1)(preds_batch).cpu().numpy()
            preds.append(preds_batch)

    preds = np.concatenate(preds, axis=0)

    # 计算 Inception Score
    split_scores = []
    for k in range(splits):
        part = preds[k * (N // splits): (k + 1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for p in part:
            scores.append(entropy(p, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


# =========================== 提取 Inception 特征，用于 FID / KID ============================

def get_inception_features(images, device, batch_size=32):
    """
    提取 Inception v3 模型的 pool3 特征，用于 FID 和 KID 计算。

    images: 形状 (N, 3, H, W)，值范围 [0,1]
    """
    # 加载 Inception v3 模型
    inception = models.inception_v3(pretrained=True, transform_input=False).to(device)
    inception.eval()

    # 注册 hook 以提取 pool3 特征
    features = []

    def hook(module, input, output):
        features.append(output.cpu().numpy())  # (N, 2048, 1, 1)

    handle = inception.avgpool.register_forward_hook(hook)

    # 定义预处理
    # 注意：若 images 已经是 Tensor，需要能处理 Tensor 的 transforms.Resize 或使用 F.interpolate
    transform = transforms.Compose([
        transforms.Resize((299, 299)),  # 若报错，需使用其他方法对 Tensor resize
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    dataset = InceptionDataset(images, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting Inception Features"):
            batch = batch.to(device)
            inception(batch)

    handle.remove()
    features = np.concatenate(features, axis=0)  # (N, 2048, 1, 1)
    features = features.reshape(features.shape[0], features.shape[1])  # (N, 2048)
    return features


def compute_kid(real_features, fake_features, degree=3, coef0=1.0):
    """
    计算 Kernel Inception Distance (KID) 使用多项式核。
    real_features, fake_features: (N, 2048) 或相同维度
    """
    Kxx = (np.dot(real_features, real_features.T) + coef0) ** degree
    Kyy = (np.dot(fake_features, fake_features.T) + coef0) ** degree
    Kxy = (np.dot(real_features, fake_features.T) + coef0) ** degree

    m = real_features.shape[0]
    sum_Kxx = np.sum(Kxx) - np.sum(np.diag(Kxx))
    sum_Kyy = np.sum(Kyy) - np.sum(np.diag(Kyy))
    sum_Kxy = np.sum(Kxy)

    kid = (sum_Kxx + sum_Kyy) / (m * (m - 1)) - 2 * sum_Kxy / (m * m)
    return kid


# =========================== 辅助函数 ============================

def load_images_from_folder(folder, transform=None, max_images=None):
    """
    从文件夹中加载图像并应用给定的转换，返回 (N, 3, H, W) 的 Tensor。
    """
    # 如果没有指定 transform，则默认把图片转为 Tensor
    if transform is None:
        transform = transforms.ToTensor()

    images = []
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            img_path = os.path.join(folder, filename)
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img)
                images.append(img_tensor)
                if max_images and len(images) >= max_images:
                    break
            except Exception as e:
                print(f"加载图像时出错 {img_path}: {e}")

    if not images:
        raise ValueError(f"文件夹 {folder} 中没有有效的图像文件。")

    return torch.stack(images)


def save_generated_images(images, folder, img_size=64):
    """
    将生成的图像保存到指定文件夹。
    images: Tensor (N, 3, H, W)，值范围 [-1,1] 或 [0,1]
    """
    os.makedirs(folder, exist_ok=True)
    # 假设生成器输出范围是 [-1,1]，转换为 [0,1]
    images = (images + 1) / 2
    for i in range(images.size(0)):
        img = images[i].cpu().clamp(0, 1)  # 防止出界
        pil_img = transforms.ToPILImage()(img)
        pil_img.save(os.path.join(folder, f'generated_{i + 1:05d}.png'))


# =========================== 核心评估函数 ============================

def evaluate_gan(generator, device, num_images, batch_size, generated_dir, real_images_path):
    """
    评估指定GAN模型，生成图像并计算 IS/FID/KID。
    """
    generator.eval()
    os.makedirs(generated_dir, exist_ok=True)

    print(f"Generating images for {generated_dir}...")
    with torch.no_grad():
        for i in tqdm(range(0, num_images, batch_size), desc="Generating Images"):
            current_batch_size = min(batch_size, num_images - i)

            noise = torch.randn(current_batch_size, 100, 1, 1, device=device)
            fake_images = generator(noise).cpu()
            save_generated_images(fake_images, generated_dir, img_size=64)

    # 1. Inception Score
    print(f"\nLoading generated images from {generated_dir} for Inception Score...")
    gen_imgs = load_images_from_folder(generated_dir, transform=transforms.ToTensor(), max_images=num_images)
    print(f"Number of generated images: {len(gen_imgs)}")
    mean_is, std_is = calculate_inception_score(gen_imgs, batch_size=batch_size, resize=True, splits=10)
    print(f'Inception Score: {mean_is:.4f} ± {std_is:.4f}')

    # 2. Frechet Inception Distance
    print(f"\nCalculating Frechet Inception Distance (FID) between real_images and {generated_dir}...")
    fid = fid_score.calculate_fid_given_paths(
        [real_images_path, generated_dir],
        batch_size=batch_size,
        device=device,
        dims=2048
    )
    print(f'FID: {fid:.4f}')

    # 3. Kernel Inception Distance
    print(f"\nCalculating Kernel Inception Distance (KID) between real_images and {generated_dir}...")
    real_imgs = load_images_from_folder(real_images_path, transform=None, max_images=num_images)
    print(f"Number of real images: {len(real_imgs)}")

    fake_imgs = gen_imgs
    real_features = get_inception_features(real_imgs, device, batch_size)
    fake_features = get_inception_features(fake_imgs, device, batch_size)
    kid_score = compute_kid(real_features, fake_features, degree=3, coef0=1.0)
    print(f'KID: {kid_score:.6f}')

    return {
        'Inception Score': (mean_is, std_is),
        'FID': fid,
        'KID': kid_score
    }

def evaluate_advanced_gan(generator, device, num_images, batch_size, generated_dir, real_images_path):
    """
    评估指定高级 GAN 模型，生成图像并计算 IS/FID/KID。
    """
    generator.eval()
    os.makedirs(generated_dir, exist_ok=True)

    print(f"Generating images for {generated_dir}...")
    with torch.no_grad():
        for i in tqdm(range(0, num_images, batch_size), desc="Generating Images"):
            current_batch_size = min(batch_size, num_images - i)
            # 使用 AdvancedGnet 的噪声维度生成噪声
            noise = torch.randn(current_batch_size, 100, 1, 1, device=device)
            noise = noise.view(noise.size(0), -1)  # 展平为 (batch_size, 100)
            fake_images = generator(noise).cpu()
            save_generated_images(fake_images, generated_dir, img_size=64)

    # 1. Inception Score
    print(f"\nLoading generated images from {generated_dir} for Inception Score...")
    gen_imgs = load_images_from_folder(generated_dir, transform=transforms.ToTensor(), max_images=num_images)
    print(f"Number of generated images: {len(gen_imgs)}")
    mean_is, std_is = calculate_inception_score(gen_imgs, batch_size=batch_size, resize=True, splits=10)
    print(f'Inception Score: {mean_is:.4f} ± {std_is:.4f}')

    # 2. Frechet Inception Distance
    print(f"\nCalculating Frechet Inception Distance (FID) between real_images and {generated_dir}...")
    fid = fid_score.calculate_fid_given_paths(
        [real_images_path, generated_dir],
        batch_size=batch_size,
        device=device,
        dims=2048
    )
    print(f'FID: {fid:.4f}')

    # 3. Kernel Inception Distance
    print(f"\nCalculating Kernel Inception Distance (KID) between real_images and {generated_dir}...")
    real_imgs = load_images_from_folder(real_images_path, transform=None, max_images=num_images)
    print(f"Number of real images: {len(real_imgs)}")

    fake_imgs = gen_imgs
    real_features = get_inception_features(real_imgs, device, batch_size)
    fake_features = get_inception_features(fake_imgs, device, batch_size)
    kid_score = compute_kid(real_features, fake_features, degree=3, coef0=1.0)
    print(f'KID: {kid_score:.6f}')

    return {
        'Inception Score': (mean_is, std_is),
        'FID': fid,
        'KID': kid_score
    }

# =========================== 脚本入口 ============================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate multiple trained GAN models (IS, FID, KID).')
    parser.add_argument('--generators', type=str, nargs='+', required=True,
                        help='Generator specs: "module.class:path_to_weights:generated_dir"')
    parser.add_argument('--real_images', type=str, required=True, help='Path to the folder containing real images.')
    parser.add_argument('--num_images', type=int, default=1000, help='Number of images to generate for evaluation.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--device', type=str, default='cuda', help="Use 'cuda' or 'cpu'.")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    results = {}
    for gen_spec in args.generators:
        try:
            module_class, model_path, generated_dir = gen_spec.split(':')
            module_name, class_name = module_class.rsplit('.', 1)
            # 动态导入生成器类
            module = __import__(module_name, fromlist=[class_name])
            GeneratorClass = getattr(module, class_name)

            # 根据类名传入 opt 或 adv_opt
            if class_name in ["Gnet", "Dnet"]:
                generator = GeneratorClass(opt).to(device)
                print(f"\nLoading generator '{class_name}' from '{module_name}' with weights '{model_path}'...")
                try:
                    generator.load_state_dict(torch.load(model_path, map_location=device))
                except RuntimeError as ex:
                    print(f"加载模型权重时出错: {ex}")
                    continue
                print(f"Generator '{class_name}' loaded successfully.")
                metrics = evaluate_gan(
                    generator=generator,
                    device=device,
                    num_images=args.num_images,
                    batch_size=args.batch_size,
                    generated_dir=generated_dir,
                    real_images_path=args.real_images
                )
            elif class_name in ["AdvancedGnet", "AdvancedDnet"]:
                generator = GeneratorClass(adv_opt).to(device)
                print(f"\nLoading generator '{class_name}' from '{module_name}' with weights '{model_path}'...")
                try:
                    generator.load_state_dict(torch.load(model_path, map_location=device))
                except RuntimeError as ex:
                    print(f"加载模型权重时出错: {ex}")
                    continue
                print(f"Generator '{class_name}' loaded successfully.")
                metrics = evaluate_advanced_gan(
                    generator=generator,
                    device=device,
                    num_images=args.num_images,
                    batch_size=args.batch_size,
                    generated_dir=generated_dir,
                    real_images_path=args.real_images
                )
            else:
                raise ValueError(f"未识别的生成器类: {class_name}")

        except Exception as e:
            print(f"解析生成器规范 '{gen_spec}' 时出错: {e}")
            continue

        # 评估
        results[class_name] = metrics

    # 输出结果
    print("\n=== Evaluation Metrics ===")
    for gen_name, metrics in results.items():
        is_mean, is_std = metrics['Inception Score']
        print(f"\nGenerator: {gen_name}")
        print(f"Inception Score: {is_mean:.4f} ± {is_std:.4f}")
        print(f"FID: {metrics['FID']:.4f}")
        print(f"KID: {metrics['KID']:.6f}")