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

#--------------------------------1.参数配置---------------------------------#
class Config():
    result_save_path = 'results/'  # 生成图像保存的路径
    gif_save_path = 'gif_results/'  # GIF图像保存的路径
    evaluate_save_path = 'evaluate_results/'
    d_net_path = 'snapshots/dnet.pth'  # 判别网络权重文件保存的路径
    g_net_path = 'snapshots/gnet.pth'  # 生成网络权重文件保存的路径
    img_path = 'face/'  # 源图像文件路径

    img_size = 96  # 图像裁剪尺寸
    batch_size = 256  # 批数量
    max_epoch = 1000  # 循环轮次
    noise_dim = 100  # 初始噪声的通道维度
    feats_channel = 64  # 中间特征图维度

opt = Config()  # 类实例化

# 确保保存路径存在
if not os.path.exists('results'):
    os.mkdir('results')
if not os.path.exists('snapshots'):
    os.mkdir('snapshots')
if not os.path.exists(opt.gif_save_path):
    os.mkdir(opt.gif_save_path)
if not os.path.exists(opt.evaluate_save_path):
    os.mkdir(opt.evaluate_save_path)
#---------------------------------2.生成网络设计----------------------------------#
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

#----------------------------------3.判别网络设计-----------------------------#
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

#---------------------------------4.训练函数---------------------------------#
def train_gan(opt):
    """
    训练 GAN 网络（生成器 G 和 判别器 D）：
    使用 WGAN 损失和梯度惩罚进行训练。
    """
    # 1. 加载数据集
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(opt.img_size),
        torchvision.transforms.CenterCrop(opt.img_size),
        torchvision.transforms.ToTensor()
    ])

    dataset = torchvision.datasets.ImageFolder(root=opt.img_path, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, num_workers=0, drop_last=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 调用cpu或者cuda
    # 2. 初始化生成器和判别器
    g_net = Gnet(opt).to(device)
    d_net = Dnet(opt).to(device)

    # 3. 定义优化器
    optimize_g = torch.optim.Adam(g_net.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optimize_d = torch.optim.Adam(d_net.parameters(), lr=2e-4, betas=(0.5, 0.999))

    # 4. 定义损失函数（WGAN损失 + 梯度惩罚）
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

    # 5. 固定噪声样本用于生成图像
    fixed_noise = torch.randn(8, opt.noise_dim, 1, 1).to(device)

    # 6. 存储生成的图像用于GIF
    gif_images = {i: [] for i in range(8)}  # 每个固定噪声对应一个GIF


    # 加载权重
    try:
        g_net.load_state_dict(torch.load(opt.g_net_path))  # 载入权重文件
        d_net.load_state_dict(torch.load(opt.d_net_path))  # 载入判别网络权重文件
        print('加载成功，继续训练')
    except:
        print('加载失败，重新训练')


    # 7. 训练过程
    for epoch in range(opt.max_epoch):  # 总循环轮次
        for iteration, (img, _) in tqdm(enumerate(dataloader)):  # 遍历数据集
            real_img = img.to(device)

            # 训练判别网络
            for _ in range(3):  # 每次训练判别器3次
                optimize_d.zero_grad()

                # 判别器前向
                real_output = d_net(real_img)
                fake_image = g_net(torch.randn(opt.batch_size, opt.noise_dim, 1, 1).to(device)).detach()  # 生成假图像
                fake_output = d_net(fake_image)

                # 计算判别器损失
                d_loss = wgan_loss(real_output, fake_output)

                # 计算梯度惩罚
                gradient_penalty = compute_gradient_penalty(d_net, real_img, fake_image)
                d_loss += 10 * gradient_penalty  # 梯度惩罚项加权

                d_loss.backward()
                optimize_d.step()

            # 训练生成网络
            optimize_g.zero_grad()
            fake_image = g_net(torch.randn(opt.batch_size, opt.noise_dim, 1, 1).to(device))
            fake_output = d_net(fake_image)
            g_loss = -torch.mean(fake_output)

            g_loss.backward()
            optimize_g.step()

            # 每10次迭代保存生成的图像
            if iteration % 10 == 0:
                # 保存生成的图像
                vid_fake_image = g_net(torch.randn(opt.batch_size, opt.noise_dim, 1, 1).to(device))
                torchvision.utils.save_image(vid_fake_image.data[:8],
                                             "%s/%s_epoch_%d_iter_%d.png" % (opt.result_save_path, epoch, epoch, iteration),
                                             normalize=True)

                # 使用固定噪声生成图像并加入GIF
                with torch.no_grad():
                    fixed_fake_images = g_net(fixed_noise)
                    for i, img in enumerate(fixed_fake_images):
                        gif_images[i].append(img.cpu().numpy())  # 为每个噪声保存生成的图像

        # 每个 epoch 完成后保存模型
        torch.save(d_net.state_dict(), opt.d_net_path)
        torch.save(g_net.state_dict(), opt.g_net_path)
        print(f"Epoch {epoch}/{opt.max_epoch} --- D Loss: {d_loss.item()} --- G Loss: {g_loss.item()}")

    # 保存每个固定噪声的GIF
    for i in range(8):
        gif_filename = os.path.join(opt.gif_save_path, f"noise_{i}_generated_images.gif")
        imageio.mimsave(gif_filename,
                        [torchvision.transforms.ToPILImage()(torch.Tensor(img).clamp(0, 1)) for img in gif_images[i]],
                        duration=0.5)


if __name__ == "__main__":
    train_gan(opt)
