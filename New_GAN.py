import os
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import imageio

# -------------------------------1. Parameter Configuration--------------------------------- #
class AdvancedConfig:
    adv_result_save_path = 'adv_results/'         # Path to save generated images
    adv_gif_save_path = 'adv_gif_results/'       # Path to save GIF images
    adv_d_net_path = 'adv_snapshots/adv_dnet.pth'  # Path to save the discriminator network weights
    adv_g_net_path = 'adv_snapshots/adv_gnet.pth'  # Path to save the generator network weights
    adv_data_path = 'face/'                       # Path to the dataset


    adv_img_size = 96       # Image size
    adv_batch_size = 64     # Batch size
    adv_max_epoch = 1000       # Number of training epochs
    adv_noise_dim = 100     # Noise vector dimension
    feats_channel = 64      # Feature map dimension
    w_dim = 128             # W-space dimension

adv_opt = AdvancedConfig()

# Ensure the save paths exist
os.makedirs(adv_opt.adv_result_save_path, exist_ok=True)
os.makedirs('adv_snapshots', exist_ok=True)
os.makedirs(adv_opt.adv_gif_save_path, exist_ok=True)


# ------------------Mapping Network------------------#
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

# ------------------Adaptive Instance Normalization (AdaIN)------------------#
class AdaIN(nn.Module):
    def __init__(self, w_dim, in_channels):
        super(AdaIN, self).__init__()
        self.w_dim = w_dim
        self.in_channels = in_channels

        # AdaIN's gamma and beta are calculated from the w vector
        self.gamma = nn.Linear(w_dim, in_channels)
        self.beta = nn.Linear(w_dim, in_channels)

    def forward(self, x, w):
        # Calculate gamma and beta
        gamma = self.gamma(w).view(-1, self.in_channels, 1, 1)
        beta = self.beta(w).view(-1, self.in_channels, 1, 1)

        # Normalize
        mean = x.mean(dim=[2, 3], keepdim=True)
        std = x.std(dim=[2, 3], keepdim=True)

        # AdaIN normalization
        x = (x - mean) / (std + 1e-5)

        # Add gamma and beta back
        x = gamma * x + beta
        return x

# ------------------Generator Network (Generator)------------------#
class AdvancedGnet(nn.Module):
    def __init__(self, opt):
        super(AdvancedGnet, self).__init__()
        self.feats = opt.feats_channel  # Get feature map dimension from config
        self.w_dim = opt.w_dim          # Get W-space dimension from config
        self.mapping_network = MappingNetwork(opt.adv_noise_dim, self.w_dim)

        # First layer of the generator: Map noise vector to feature map
        self.fc = nn.Sequential(
            nn.Linear(opt.adv_noise_dim, self.feats * 8 * 4 * 4),  # Output a feature vector of appropriate size
            nn.ReLU(True)
        )

        # Add Adaptive Instance Normalization (AdaIN) layers
        self.adain1 = AdaIN(self.w_dim, self.feats * 8)
        self.adain2 = AdaIN(self.w_dim, self.feats * 4)
        self.adain3 = AdaIN(self.w_dim, self.feats * 2)
        self.adain4 = AdaIN(self.w_dim, self.feats)

        # Subsequent deconvolution layers
        self.conv_layers = nn.ModuleList([
            # First group: ConvTranspose -> BatchNorm -> ReLU
            nn.ConvTranspose2d(self.feats * 8, self.feats * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.feats * 4),
            nn.ReLU(inplace=True),

            # Second group: ConvTranspose -> BatchNorm -> ReLU
            nn.ConvTranspose2d(self.feats * 4, self.feats * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.feats * 2),
            nn.ReLU(inplace=True),

            # Third group: ConvTranspose -> BatchNorm -> ReLU
            nn.ConvTranspose2d(self.feats * 2, self.feats, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.feats),
            nn.ReLU(inplace=True),

            # Final layer: ConvTranspose -> Tanh
            nn.ConvTranspose2d(self.feats, 3, 5, 3, 1, bias=False),
            nn.Tanh()  # Output image
        ])

    def forward(self, z):
        # Map to W space
        w = self.mapping_network(z)

        # Map noise z to a larger feature map
        x = self.fc(z).view(-1, self.feats * 8, 4, 4)

        # Adjust feature maps through AdaIN layers
        x = self.adain1(x, w)  # Adjust feature map
        x = self.conv_layers[0](x)  # ConvTranspose2d: 512 -> 256
        x = self.conv_layers[1](x)  # BatchNorm2d: 256
        x = self.conv_layers[2](x)  # ReLU

        x = self.adain2(x, w)  # Adjust feature map
        x = self.conv_layers[3](x)  # ConvTranspose2d: 256 -> 128
        x = self.conv_layers[4](x)  # BatchNorm2d: 128
        x = self.conv_layers[5](x)  # ReLU

        x = self.adain3(x, w)  # Adjust feature map
        x = self.conv_layers[6](x)  # ConvTranspose2d: 128 -> 64
        x = self.conv_layers[7](x)  # BatchNorm2d: 64
        x = self.conv_layers[8](x)  # ReLU

        x = self.adain4(x, w)  # Adjust feature map
        x = self.conv_layers[9](x)  # ConvTranspose2d: 64 -> 3
        x = self.conv_layers[10](x) # Tanh

        return x

# ------------------Discriminator Network (Discriminator)------------------#
class AdvancedDnet(nn.Module):
    def __init__(self, opt):
        super(AdvancedDnet, self).__init__()
        self.feats = opt.feats_channel  # Get feature map dimension from config
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

# ------------------Training Function------------------#
def train_advanced_gan(opt):
    # Data preprocessing
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

    # Initialize generator and discriminator
    g_net = AdvancedGnet(opt).to(device)
    d_net = AdvancedDnet(opt).to(device)

    # Optimizers and loss function
    optimizer_g = torch.optim.Adam(g_net.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(d_net.parameters(), lr=2e-4, betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()

    # Fixed noise for generating GIF
    fixed_noise = torch.randn(8, opt.adv_noise_dim, device=device)
    gif_images = {i: [] for i in range(8)}  # One GIF for each fixed noise

    # Training loop
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

            # Labels
            real_labels = torch.ones(batch_size, device=device)
            fake_labels = torch.zeros(batch_size, device=device)

            # ---------------------
            # Train Discriminator
            # ---------------------
            optimizer_d.zero_grad()

            # Real images
            outputs_real = d_net(real_imgs)
            d_loss_real = criterion(outputs_real, real_labels)

            # Fake images
            z = torch.randn(batch_size, opt.adv_noise_dim, device=device)
            fake_imgs = g_net(z)
            outputs_fake = d_net(fake_imgs.detach())
            d_loss_fake = criterion(outputs_fake, fake_labels)

            # Total discriminator loss
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_d.step()

            # ---------------------
            # Train Generator
            # ---------------------
            optimizer_g.zero_grad()

            outputs_fake_for_g = d_net(fake_imgs)
            g_loss = criterion(outputs_fake_for_g, real_labels)  # Generator wants discriminator to classify fake images as real
            g_loss.backward()
            optimizer_g.step()

            # Accumulate losses
            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            num_batches += 1

            # Update progress bar
            progress_bar.set_postfix({
                "D_Loss": f"{d_loss.item():.4f}",
                "G_Loss": f"{g_loss.item():.4f}"
            })

        # Average losses for the epoch
        avg_d_loss = epoch_d_loss / num_batches
        avg_g_loss = epoch_g_loss / num_batches
        print(f"Epoch [{epoch}/{opt.adv_max_epoch}] - D_Loss: {avg_d_loss:.4f}, G_Loss: {avg_g_loss:.4f}")

        # Save generated images at the end of each epoch
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


        # Save model weights
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

                # Create the duration list for each frame
                durations = [0.5] * len(frames)
                if len(durations) > 0:
                    durations[-1] = 2.0  # Last frame lasts for 2 seconds

                # Save GIF using the durations list
                imageio.mimsave(gif_path, frames, duration=durations)
                print(f"Saved GIF: {gif_path}")
            except Exception as e:
                print(f"Error saving GIF for fixed_noise_{i}: {e}")
    except Exception as e:
        print(f"Error during GIF saving: {e}")

    print("Training completed.")

#--------------------------------7. Run Training----------------------------------#
if __name__ == "__main__":
    train_advanced_gan(adv_opt)
