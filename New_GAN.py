import os
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import imageio

# -------------------------------1. 参数配置--------------------------------- #
class AdvancedConfig:
    adv_result_save_path = 'adv_results/'         # 生成图像保存的路径
    adv_gif_save_path = 'adv_gif_results/'       # GIF图像保存的路径
    adv_d_net_path = 'adv_snapshots/adv_dnet.pth'  # 判别网络权重文件保存的路径
    adv_g_net_path = 'adv_snapshots/adv_gnet.pth'  # 生成网络权重文件保存的路径
    adv_data_path = 'face/'                       # 数据路径


    adv_img_size = 96       # 图像尺寸
    adv_batch_size = 64     # 批量大小
    adv_max_epoch = 1000       # 训练轮数
    adv_noise_dim = 100     # 噪声向量维度
    feats_channel = 64      # 特征图维度
    w_dim = 128             # W空间维度

adv_opt = AdvancedConfig()

# 确保保存路径存在
os.makedirs(adv_opt.adv_result_save_path, exist_ok=True)
os.makedirs('adv_snapshots', exist_ok=True)
os.makedirs(adv_opt.adv_gif_save_path, exist_ok=True)


# ------------------映射网络（Mapping Network）------------------#
class MappingNetwork(nn.Module):
    def __init__(self, noise_dim, w_dim):
        super(MappingNetwork, self).__init__()
        self.mapping = nn.Sequential(
            nn.Linear(noise_dim, w_dim),
            nn.ReLU(True),
            nn.Linear(w_dim, w_dim),
            nn.ReLU(True),
            nn.Linear(w_dim, w_dim)
        )

    def forward(self, z):
        return self.mapping(z)

# ------------------自适应实例归一化（AdaIN）------------------#
class AdaIN(nn.Module):
    def __init__(self, w_dim, in_channels):
        super(AdaIN, self).__init__()
        self.w_dim = w_dim
        self.in_channels = in_channels

        # AdaIN的gamma和beta是通过w向量计算的
        self.gamma = nn.Linear(w_dim, in_channels)
        self.beta = nn.Linear(w_dim, in_channels)

    def forward(self, x, w):
        # 计算 gamma 和 beta
        gamma = self.gamma(w).view(-1, self.in_channels, 1, 1)
        beta = self.beta(w).view(-1, self.in_channels, 1, 1)

        # 归一化
        mean = x.mean(dim=[2, 3], keepdim=True)
        std = x.std(dim=[2, 3], keepdim=True)

        # AdaIN的标准化
        x = (x - mean) / (std + 1e-5)

        # 重新加上 gamma 和 beta
        x = gamma * x + beta
        return x

# ------------------生成器网络（Generator）------------------#
class AdvancedGnet(nn.Module):
    def __init__(self, opt):
        super(AdvancedGnet, self).__init__()
        self.feats = opt.feats_channel  # 从配置中获取特征图维度
        self.w_dim = opt.w_dim          # 从配置中获取W空间维度
        self.mapping_network = MappingNetwork(opt.adv_noise_dim, self.w_dim)

        # 生成器的第一层：将噪声向量映射到特征图
        self.fc = nn.Sequential(
            nn.Linear(opt.adv_noise_dim, self.feats * 8 * 4 * 4),  # 输出一个合适大小的特征向量
            nn.ReLU(True)
        )

        # 添加自适应实例归一化（AdaIN）层
        self.adain1 = AdaIN(self.w_dim, self.feats * 8)
        self.adain2 = AdaIN(self.w_dim, self.feats * 4)
        self.adain3 = AdaIN(self.w_dim, self.feats * 2)
        self.adain4 = AdaIN(self.w_dim, self.feats)

        # 后续的反卷积层
        self.conv_layers = nn.ModuleList([
            # 第一组：ConvTranspose -> BatchNorm -> ReLU
            nn.ConvTranspose2d(self.feats * 8, self.feats * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.feats * 4),
            nn.ReLU(inplace=True),

            # 第二组：ConvTranspose -> BatchNorm -> ReLU
            nn.ConvTranspose2d(self.feats * 4, self.feats * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.feats * 2),
            nn.ReLU(inplace=True),

            # 第三组：ConvTranspose -> BatchNorm -> ReLU
            nn.ConvTranspose2d(self.feats * 2, self.feats, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.feats),
            nn.ReLU(inplace=True),

            # 最后一层：ConvTranspose -> Tanh
            nn.ConvTranspose2d(self.feats, 3, 5, 3, 1, bias=False),
            nn.Tanh()  # 输出图像
        ])

    def forward(self, z):
        # 映射到W空间
        w = self.mapping_network(z)

        # 将噪声z映射到更大的特征图
        x = self.fc(z).view(-1, self.feats * 8, 4, 4)

        # 通过AdaIN层调整不同的层
        x = self.adain1(x, w)  # 调整特征图
        x = self.conv_layers[0](x)  # ConvTranspose2d: 512 -> 256
        x = self.conv_layers[1](x)  # BatchNorm2d: 256
        x = self.conv_layers[2](x)  # ReLU

        x = self.adain2(x, w)  # 调整特征图
        x = self.conv_layers[3](x)  # ConvTranspose2d: 256 -> 128
        x = self.conv_layers[4](x)  # BatchNorm2d: 128
        x = self.conv_layers[5](x)  # ReLU

        x = self.adain3(x, w)  # 调整特征图
        x = self.conv_layers[6](x)  # ConvTranspose2d: 128 -> 64
        x = self.conv_layers[7](x)  # BatchNorm2d: 64
        x = self.conv_layers[8](x)  # ReLU

        x = self.adain4(x, w)  # 调整特征图
        x = self.conv_layers[9](x)  # ConvTranspose2d: 64 -> 3
        x = self.conv_layers[10](x) # Tanh

        return x

# ------------------判别器网络（Discriminator）------------------#
class AdvancedDnet(nn.Module):
    def __init__(self, opt):
        super(AdvancedDnet, self).__init__()
        self.feats = opt.feats_channel  # 从配置中获取特征图维度
        self.discrim = nn.Sequential(
            nn.Conv2d(3, self.feats, 5, 3, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(self.feats, self.feats * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.feats * 2),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(self.feats * 2, self.feats * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.feats * 4),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(self.feats * 4, self.feats * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.feats * 8),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(self.feats * 8, 1, 4, 1, 0, bias=True)
        )

    def forward(self, x):
        return self.discrim(x).view(-1)

# ------------------训练函数（Training Function）------------------#
def train_advanced_gan(opt):
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize(opt.adv_img_size),
        transforms.CenterCrop(opt.adv_img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    dataset = torchvision.datasets.ImageFolder(opt.adv_data_path, transform)
    dataloader = DataLoader(dataset, batch_size=opt.adv_batch_size, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 初始化生成器和判别器
    g_net = AdvancedGnet(opt).to(device)
    d_net = AdvancedDnet(opt).to(device)

    # 优化器和损失函数
    optimizer_g = torch.optim.Adam(g_net.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(d_net.parameters(), lr=2e-4, betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()

    # 固定噪声用于生成GIF
    fixed_noise = torch.randn(8, opt.adv_noise_dim, device=device)
    gif_images = {i: [] for i in range(8)}  # 每个固定噪声对应一个GIF

    # 训练循环
    for epoch in range(1, opt.adv_max_epoch + 1):
        g_net.train()
        d_net.train()
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch [{epoch}/{opt.adv_max_epoch}]")
        for real_imgs, _ in progress_bar:
            real_imgs = real_imgs.to(device)
            batch_size = real_imgs.size(0)

            # 标签
            real_labels = torch.ones(batch_size, device=device)
            fake_labels = torch.zeros(batch_size, device=device)

            # ---------------------
            # 训练判别器
            # ---------------------
            optimizer_d.zero_grad()

            # 真实图像
            outputs_real = d_net(real_imgs)
            d_loss_real = criterion(outputs_real, real_labels)

            # 生成假图像
            z = torch.randn(batch_size, opt.adv_noise_dim, device=device)
            fake_imgs = g_net(z)
            outputs_fake = d_net(fake_imgs.detach())
            d_loss_fake = criterion(outputs_fake, fake_labels)

            # 总判别器损失
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_d.step()

            # ---------------------
            # 训练生成器
            # ---------------------
            optimizer_g.zero_grad()

            outputs_fake_for_g = d_net(fake_imgs)
            g_loss = criterion(outputs_fake_for_g, real_labels)  # 生成器希望判别器认为假图像为真
            g_loss.backward()
            optimizer_g.step()

            # 累积损失
            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            num_batches += 1

            # 更新进度条
            progress_bar.set_postfix({
                "D_Loss": f"{d_loss.item():.4f}",
                "G_Loss": f"{g_loss.item():.4f}"
            })

        # 计算平均损失
        avg_d_loss = epoch_d_loss / num_batches
        avg_g_loss = epoch_g_loss / num_batches
        print(f"Epoch [{epoch}/{opt.adv_max_epoch}] - D_Loss: {avg_d_loss:.4f}, G_Loss: {avg_g_loss:.4f}")

        # 每个 epoch 结束后保存生成图像
        try:
            with torch.no_grad():
                g_net.eval()
                generated_imgs = g_net(fixed_noise)
                torchvision.utils.save_image(
                    generated_imgs,
                    f"{opt.adv_result_save_path}/epoch_{epoch}.png",
                    normalize=True
                )
                for i, img in enumerate(generated_imgs):
                    gif_images[i].append(img.cpu())
            print(f"Saved generated images for epoch {epoch}.")
        except Exception as e:
            print(f"Error saving images at epoch {epoch}: {e}")


        # 保存模型权重
        try:
            torch.save(g_net.state_dict(), opt.adv_g_net_path)
            torch.save(d_net.state_dict(), opt.adv_d_net_path)
            print(f"Saved model checkpoints for epoch {epoch}.")
        except Exception as e:
            print(f"Error saving model checkpoints at epoch {epoch}: {e}")

    try:
        print("Saving GIFs...")
        for i in range(8):
            try:
                gif_path = os.path.join(opt.adv_gif_save_path, f"fixed_noise_{i}.gif")
                frames = [transforms.ToPILImage()(img) for img in gif_images[i]]

                # 创建每帧的持续时间列表
                durations = [0.5] * len(frames)
                if len(durations) > 0:
                    durations[-1] = 2.0  # 最后一帧持续2秒

                # 使用 durations 列表设置每帧的展示时间
                imageio.mimsave(gif_path, frames, duration=durations)
                print(f"Saved GIF: {gif_path}")
            except Exception as e:
                print(f"Error saving GIF for fixed_noise_{i}: {e}")
    except Exception as e:
        print(f"Error during GIF saving: {e}")

    print("Training completed.")

#--------------------------------7.运行训练----------------------------------#
if __name__ == "__main__":
    train_advanced_gan(adv_opt)
