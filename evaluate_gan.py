"""
evaluate_gan.py

Used to evaluate multiple trained GAN models, calculating Inception Score (IS), Frechet Inception Distance (FID),
and Kernel Inception Distance (KID). Please modify according to project requirements.
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

# ======== 1. Import Configurations ========
from GAN import opt  # Basic configuration
from New_GAN import adv_opt  # Advanced configuration


# =========================== InceptionDataset ============================

class InceptionDataset(Dataset):
    """
    If self.images is a 4D Tensor (N, 3, H, W),
    then transform should be able to handle Tensors;
    If self.images is a list of PIL Images, then transform should handle PIL Images.
    """

    def __init__(self, images, transform=None):
        self.images = images  # Could be Tensor or list (PIL)
        self.transform = transform

    def __len__(self):
        if isinstance(self.images, torch.Tensor):
            return self.images.size(0)
        else:
            return len(self.images)

    def __getitem__(self, idx):
        # When self.images is a Tensor
        if isinstance(self.images, torch.Tensor):
            img = self.images[idx]
            # If further transformation (like normalization) is needed, transform must handle Tensor
            if self.transform:
                img = self.transform(img)
            return img

        # When self.images is a list (PIL Image)
        else:
            img = self.images[idx]
            if self.transform:
                img = self.transform(img)
            return img


# =========================== Inception Score Implementation ============================

def calculate_inception_score(imgs, batch_size=32, resize=True, splits=10):
    """
    Calculate the Inception Score for generated images
    imgs: torch.Tensor, shape (N, 3, H, W), value range [0,1]
    """
    N = len(imgs)
    # Ensure N >= batch_size
    assert N >= batch_size, f"Number of images (N={N}) must be >= batch_size (batch_size={batch_size})."

    # Load Inception v3 model
    inception_model = models.inception_v3(pretrained=True, transform_input=False).eval()
    if torch.cuda.is_available():
        inception_model = inception_model.cuda()

    # Preprocessing
    if resize:
        # Note: transforms.Resize for Tensor may require a new version of torchvision
        # Older versions support only PIL; this is just an example. If incompatible, use torch.nn.functional.interpolate.
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

    # Calculate Inception Score
    split_scores = []
    for k in range(splits):
        part = preds[k * (N // splits): (k + 1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for p in part:
            scores.append(entropy(p, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


# =========================== Extract Inception Features for FID / KID ============================

def get_inception_features(images, device, batch_size=32):
    """
    Extract pool3 features from the Inception v3 model for FID and KID calculations.

    images: shape (N, 3, H, W), value range [0,1]
    """
    # Load Inception v3 model
    inception = models.inception_v3(pretrained=True, transform_input=False).to(device)
    inception.eval()

    # Register hook to extract pool3 features
    features = []

    def hook(module, input, output):
        features.append(output.cpu().numpy())  # (N, 2048, 1, 1)

    handle = inception.avgpool.register_forward_hook(hook)

    # Define preprocessing
    # Note: If images are already Tensor, transforms.Resize or F.interpolate should be used
    transform = transforms.Compose([
        transforms.Resize((299, 299)),  # If error occurs, use other methods to resize Tensor
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
    Compute Kernel Inception Distance (KID) using polynomial kernel.
    real_features, fake_features: (N, 2048) or same dimension
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


# =========================== Helper Functions ============================

def load_images_from_folder(folder, transform=None, max_images=None):
    """
    Load images from a folder and apply given transformations, returning a Tensor of shape (N, 3, H, W).
    """
    # If no transform is specified, default to converting images to Tensor
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
                print(f"Error loading image {img_path}: {e}")

    if not images:
        raise ValueError(f"No valid image files in folder {folder}.")

    return torch.stack(images)


def save_generated_images(images, folder, img_size=64):
    """
    Save generated images to a specified folder.
    images: Tensor (N, 3, H, W), value range [-1,1] or [0,1]
    """
    os.makedirs(folder, exist_ok=True)
    # Assuming generator output range is [-1,1], convert to [0,1]
    images = (images + 1) / 2
    for i in range(images.size(0)):
        img = images[i].cpu().clamp(0, 1)  # Clamp to prevent out-of-bounds values
        pil_img = transforms.ToPILImage()(img)
        pil_img.save(os.path.join(folder, f'generated_{i + 1:05d}.png'))


# =========================== Core Evaluation Function ============================

def evaluate_gan(generator, device, num_images, batch_size, generated_dir, real_images_path):
    """
    Evaluate a specified GAN model, generate images, and compute IS/FID/KID.
    """
    generator.eval()
    os.makedirs(generated_dir, exist_ok=True)

    print(f"Generating images for {generated_dir}...")
    with torch.no_grad():
        for i in tqdm(range(0, num_images, batch_size), desc="Generating Images"):
            current_batch_size = min(batch_size, num_images - i)

            noise = torch.randn(current_batch_size, 100, 1, 1, device=device)
            fake_images = generator(noise)

            save_generated_images(fake_images, generated_dir)

    print(f"Evaluating Inception Score (IS) and FID/KID for the generated images...")

    # Load real images
    real_images = load_images_from_folder(real_images_path)
    real_images = real_images.to(device)

    # Get Inception features for both real and fake images
    fake_images = load_images_from_folder(generated_dir)
    fake_images = fake_images.to(device)

    real_features = get_inception_features(real_images, device)
    fake_features = get_inception_features(fake_images, device)

    # Calculate Inception Score
    inception_score_mean, inception_score_std = calculate_inception_score(fake_images)
    print(f"Inception Score (IS): {inception_score_mean:.2f} ± {inception_score_std:.2f}")

    # Calculate FID
    fid_value = fid_score.calculate_fid_given_paths([real_images_path, generated_dir], batch_size=50, device=device)
    print(f"Frechet Inception Distance (FID): {fid_value:.2f}")

    # Calculate KID
    kid_value = compute_kid(real_features, fake_features)
    print(f"Kernel Inception Distance (KID): {kid_value:.2f}")
