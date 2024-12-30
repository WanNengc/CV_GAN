import torch
import torch.nn as nn
import torchvision
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import imageio
from torchvision import models, transforms
from scipy.linalg import sqrtm
import numpy as np
from PIL import Image
from scipy.stats import entropy

#--------------------------------1. Parameter Configuration---------------------------------#
class Config():
    result_save_path = 'results/'  # Path to save generated images
    gif_save_path = 'gif_results/'  # Path to save GIF images
    d_net_path = 'snapshots/dnet.pth'  # Path to save the discriminator network weights
    g_net_path = 'snapshots/gnet.pth'  # Path to save the generator network weights
    img_path = 'face/'  # Path to source image files

    img_size = 96  # Image crop size
    batch_size = 256  # Batch size
    max_epoch = 1000  # Number of epochs
    noise_dim = 100  # Dimension of the initial noise
    feats_channel = 64  # Dimension of the intermediate feature maps

opt = Config()  # Instantiate the class

# Ensure save paths exist
if not os.path.exists('results'):
    os.mkdir('results')
if not os.path.exists('snapshots'):
    os.mkdir('snapshots')
if not os.path.exists(opt.gif_save_path):
    os.mkdir(opt.gif_save_path)

#---------------------------------2. Generator Network Design----------------------------------#
class Gnet(nn.Module):
    def __init__(self, opt):
        super(Gnet, self).__init__()
        self.feats = opt.feats_channel
        self.generate = nn.Sequential(
            nn.ConvTranspose2d(in_channels=opt.noise_dim, out_channels=self.feats * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.feats * 8),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=self.feats * 8, out_channels=self.feats * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.feats * 4),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=self.feats * 4, out_channels=self.feats * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.feats * 2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=self.feats * 2, out_channels=self.feats, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.feats),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=self.feats, out_channels=3, kernel_size=5, stride=3, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.generate(x)

#----------------------------------3. Discriminator Network Design-----------------------------#
class Dnet(nn.Module):
    def __init__(self, opt):
        super(Dnet, self).__init__()
        self.feats = opt.feats_channel
        self.discrim = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=self.feats, kernel_size=5, stride=3, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(in_channels=self.feats, out_channels=self.feats * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.feats * 2),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(in_channels=self.feats * 2, out_channels=self.feats * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.feats * 4),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(in_channels=self.feats * 4, out_channels=self.feats * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.feats * 8),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(in_channels=self.feats * 8, out_channels=1, kernel_size=4, stride=1, padding=0, bias=True)
        )

    def forward(self, x):
        return self.discrim(x).view(-1)

#---------------------------------4. Training Function---------------------------------#
def train_gan(opt):
    """
    Train the GAN network (Generator G and Discriminator D):
    Using WGAN loss and gradient penalty for training.
    """
    # 1. Load dataset
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(opt.img_size),
        torchvision.transforms.CenterCrop(opt.img_size),
        torchvision.transforms.ToTensor()
    ])

    dataset = torchvision.datasets.ImageFolder(root=opt.img_path, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, num_workers=0, drop_last=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Use CPU or CUDA
    # 2. Initialize generator and discriminator
    g_net = Gnet(opt).to(device)
    d_net = Dnet(opt).to(device)

    # 3. Define optimizers
    optimize_g = torch.optim.Adam(g_net.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optimize_d = torch.optim.Adam(d_net.parameters(), lr=2e-4, betas=(0.5, 0.999))

    # 4. Define loss function (WGAN loss + Gradient Penalty)
    def wgan_loss(real, fake):
        return -torch.mean(real) + torch.mean(fake)

    def compute_gradient_penalty(d_net, real_data, fake_data):
        batch_size, c, h, w = real_data.size()
        epsilon = torch.rand(batch_size, 1, 1, 1).to(device)
        interpolated_data = epsilon * real_data + (1 - epsilon) * fake_data
        interpolated_data.requires_grad_(True)

        d_interpolated = d_net(interpolated_data)
        gradients = torch.autograd.grad(outputs=d_interpolated, inputs=interpolated_data,
                                        grad_outputs=torch.ones(d_interpolated.size()).to(device),
                                        create_graph=True, retain_graph=True)[0]
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    # 5. Fixed noise samples for generating images
    fixed_noise = torch.randn(8, opt.noise_dim, 1, 1).to(device)

    # 6. Store generated images for GIF
    gif_images = {i: [] for i in range(8)}  # Each fixed noise corresponds to a GIF


    # Load weights
    try:
        g_net.load_state_dict(torch.load(opt.g_net_path))  # Load generator weights
        d_net.load_state_dict(torch.load(opt.d_net_path))  # Load discriminator weights
        print('Successfully loaded, continuing training')
    except:
        print('Loading failed, starting training from scratch')


    # 7. Training loop
    for epoch in range(opt.max_epoch):  # Total epochs
        for iteration, (img, _) in tqdm(enumerate(dataloader)):  # Iterate through dataset
            real_img = img.to(device)

            # Train the discriminator
            for _ in range(3):  # Train the discriminator 3 times for each step
                optimize_d.zero_grad()

                # Forward pass for discriminator
                real_output = d_net(real_img)
                fake_image = g_net(torch.randn(opt.batch_size, opt.noise_dim, 1, 1).to(device)).detach()  # Generate fake image
                fake_output = d_net(fake_image)

                # Calculate discriminator loss
                d_loss = wgan_loss(real_output, fake_output)

                # Compute gradient penalty
                gradient_penalty = compute_gradient_penalty(d_net, real_img, fake_image)
                d_loss += 10 * gradient_penalty  # Add weighted gradient penalty term

                d_loss.backward()
                optimize_d.step()

            # Train the generator
            optimize_g.zero_grad()
            fake_image = g_net(torch.randn(opt.batch_size, opt.noise_dim, 1, 1).to(device))
            fake_output = d_net(fake_image)
            g_loss = -torch.mean(fake_output)

            g_loss.backward()
            optimize_g.step()

            # Save generated images every 10 iterations
            if iteration % 10 == 0:
                # Save generated images
                vid_fake_image = g_net(torch.randn(opt.batch_size, opt.noise_dim, 1, 1).to(device))
                torchvision.utils.save_image(vid_fake_image.data[:8],
                                             "%s/%s_epoch_%d_iter_%d.png" % (opt.result_save_path, epoch, epoch, iteration),
                                             normalize=True)

                # Use fixed noise to generate images and add to GIF
                with torch.no_grad():
                    fixed_fake_images = g_net(fixed_noise)
                    for i, img in enumerate(fixed_fake_images):
                        gif_images[i].append(img.cpu().numpy())  # Save generated images for each noise

        # Save model after each epoch
        torch.save(d_net.state_dict(), opt.d_net_path)
        torch.save(g_net.state_dict(), opt.g_net_path)
        print(f"Epoch {epoch}/{opt.max_epoch} --- D Loss: {d_loss.item()} --- G Loss: {g_loss.item()}")

    # Save each fixed noise GIF
    for i in range(8):
        gif_filename = os.path.join(opt.gif_save_path, f"noise_{i}_generated_images.gif")
        imageio.mimsave(gif_filename,
                        [torchvision.transforms.ToPILImage()(torch.Tensor(img).clamp(0, 1)) for img in gif_images[i]],
                        duration=0.5)


if __name__ == "__main__":
    train_gan(opt)
