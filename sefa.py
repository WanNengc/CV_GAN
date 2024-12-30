import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
from GAN import Gnet


# Custom HTML visualization tool
class HtmlPageVisualizer:
    def __init__(self, num_rows, num_cols, viz_size=256):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.viz_size = viz_size
        self.headers = []
        self.cells = [[None for _ in range(num_cols)] for _ in range(num_rows)]

    def set_headers(self, headers):
        self.headers = headers

    def set_cell(self, row, col, text=None, image=None, highlight=False):
        cell_content = {}
        if text:
            cell_content['text'] = text
        if image:
            cell_content['image'] = image
        cell_content['highlight'] = highlight
        self.cells[row][col] = cell_content

    def save(self, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            f.write('<html><body><table border="1" style="border-collapse: collapse;">')

            if self.headers:
                f.write('<tr>')
                for header in self.headers:
                    f.write(f'<th>{header}</th>')
                f.write('</tr>')

            for row in self.cells:
                f.write('<tr>')
                for cell in row:
                    if cell is None:
                        f.write('<td></td>')
                    else:
                        style = 'background-color: yellow;' if cell.get('highlight', False) else ''
                        f.write(f'<td style="{style}">')
                        if 'text' in cell:
                            f.write(cell['text'])
                        if 'image' in cell:
                            image_path = cell['image']
                            f.write(f'<br><img src="{image_path}" width="{self.viz_size}" height="{self.viz_size}">')
                        f.write('</td>')
                f.write('</tr>')

            f.write('</table></body></html>')


# Image post-processing function
def postprocess(image_tensor):
    """
    Convert the generated image tensor to a visualizable format.
    """
    images = []
    for img in image_tensor:
        img = img.detach().cpu().numpy()
        img = ((img + 1) / 2 * 255).astype(np.uint8)  # Map values to [0, 255]
        img = np.transpose(img, (1, 2, 0))  # Convert to HWC format
        images.append(Image.fromarray(img))
    return images


# Weight factorization function
def factorize_dcgans(generator):
    """
    Factorize the generator weights of DCGAN to discover potential semantic directions.
    """
    weights = generator.generate[0].weight.data
    weights_flat = weights.view(weights.size(0), -1).cpu().numpy()
    u, s, vh = np.linalg.svd(weights_flat, full_matrices=False)
    boundaries = torch.tensor(u, dtype=torch.float32).to(weights.device)
    values = torch.tensor(s, dtype=torch.float32).to(weights.device)
    return boundaries, values


# Latent vector manipulation and image generation function
def generate_with_sefa(generator, boundaries, values, opt):
    """
    Use the SeFa method to manipulate the latent space of DCGAN and generate images.
    """
    device = next(generator.parameters()).device

    # Randomly generate latent vectors, averaging multiple sets of random latent vectors for stability
    codes = torch.mean(torch.stack([torch.randn(opt.num_samples, opt.noise_dim).to(device) for _ in range(5)]), dim=0)

    # Manipulation distances (expand the range)
    distances = np.linspace(opt.start_distance, opt.end_distance, opt.step)

    # Initialize HTML page for saving results
    vizer = HtmlPageVisualizer(num_rows=opt.num_semantics, num_cols=len(distances) + 1, viz_size=opt.viz_size)
    headers = [''] + [f'Distance {d:.2f}' for d in distances]
    vizer.set_headers(headers)

    # Iterate over semantic directions and distances to generate images
    for sem_id in tqdm(range(opt.num_semantics), desc='Semantic '):
        boundary = boundaries[sem_id] * 2.0  # Amplify the effect of boundaries
        vizer.set_cell(sem_id, 0, text=f'Semantic {sem_id:03d} ({values[sem_id]:.3f})', highlight=True)

        for col_id, d in enumerate(distances, start=1):
            temp_code = codes.clone()
            temp_code += boundary * d  # Manipulate the latent vector
            temp_code = temp_code.unsqueeze(-1).unsqueeze(-1)  # Adjust shape to fit the generator input

            # Generate the image using the generator
            fake_image = generator(temp_code)
            fake_image = postprocess(fake_image)[0]  # Image post-processing

            # Save image to relative path (HTML uses relative paths)
            image_relative_path = os.path.join(f'semantic_{sem_id}_dist_{col_id}.png')
            image_full_path = os.path.join(opt.save_dir, image_relative_path)
            fake_image.save(image_full_path)

            # Set the image path in the HTML page
            vizer.set_cell(sem_id, col_id, image=image_relative_path)

    # Save the results to an HTML file
    save_path = os.path.join(opt.save_dir, 'dcgan_sefa.html')
    vizer.save(save_path)
    print(f"Results saved to {save_path}")


# Main function
def main():
    class Opt:
        noise_dim = 100
        num_samples = 5
        num_semantics = 5
        start_distance = -3.0
        end_distance = 3.0
        step = 11
        viz_size = 256
        save_dir = './sefa_result'
        g_net_path = './snapshots/gnet.pth'
        feats_channel = 64
    opt = Opt()
    os.makedirs(opt.save_dir, exist_ok=True)

    if not os.path.exists(opt.g_net_path):
        raise FileNotFoundError(f"Generator checkpoint not found at {opt.g_net_path}")

    g_net = Gnet(opt)
    g_net.load_state_dict(torch.load(opt.g_net_path))
    print("Generator weights loaded successfully.")

    g_net.eval().cuda()

    boundaries, values = factorize_dcgans(g_net)
    generate_with_sefa(g_net, boundaries, values, opt)


if __name__ == '__main__':
    main()
