import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
from GAN import Gnet


# 自定义 HTML 可视化工具
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


# 图像后处理函数
def postprocess(image_tensor):
    """
    将生成的图像张量转换为可视化格式。
    """
    images = []
    for img in image_tensor:
        img = img.detach().cpu().numpy()
        img = ((img + 1) / 2 * 255).astype(np.uint8)  # 将值映射到 [0, 255]
        img = np.transpose(img, (1, 2, 0))  # 转换为 HWC 格式
        images.append(Image.fromarray(img))
    return images


# 权重分解函数
def factorize_dcgans(generator):
    """
    对 DCGAN 的生成器权重进行分解以发现潜在语义方向。
    """
    weights = generator.generate[0].weight.data
    weights_flat = weights.view(weights.size(0), -1).cpu().numpy()
    u, s, vh = np.linalg.svd(weights_flat, full_matrices=False)
    boundaries = torch.tensor(u, dtype=torch.float32).to(weights.device)
    values = torch.tensor(s, dtype=torch.float32).to(weights.device)
    return boundaries, values


# 潜在向量操控与图像生成函数
def generate_with_sefa(generator, boundaries, values, opt):
    """
    使用 SeFa 方法在 DCGAN 的潜在空间中操控并生成图像。
    """
    device = next(generator.parameters()).device

    # 随机生成潜在向量，平均多组随机潜在向量以提高稳定性
    codes = torch.mean(torch.stack([torch.randn(opt.num_samples, opt.noise_dim).to(device) for _ in range(5)]), dim=0)

    # 操控距离（扩大范围）
    distances = np.linspace(opt.start_distance, opt.end_distance, opt.step)

    # 保存结果的 HTML 页面初始化
    vizer = HtmlPageVisualizer(num_rows=opt.num_semantics, num_cols=len(distances) + 1, viz_size=opt.viz_size)
    headers = [''] + [f'Distance {d:.2f}' for d in distances]
    vizer.set_headers(headers)

    # 遍历语义方向和距离进行生成
    for sem_id in tqdm(range(opt.num_semantics), desc='Semantic '):
        boundary = boundaries[sem_id] * 2.0  # 放大边界的影响
        vizer.set_cell(sem_id, 0, text=f'Semantic {sem_id:03d} ({values[sem_id]:.3f})', highlight=True)

        for col_id, d in enumerate(distances, start=1):
            temp_code = codes.clone()
            temp_code += boundary * d  # 操控潜在向量
            temp_code = temp_code.unsqueeze(-1).unsqueeze(-1)  # 调整形状以适配生成器输入

            # 通过生成器生成图像
            fake_image = generator(temp_code)
            fake_image = postprocess(fake_image)[0]  # 图像后处理

            # 保存图像到相对路径（HTML 引用使用相对路径）
            image_relative_path = os.path.join(f'semantic_{sem_id}_dist_{col_id}.png')
            image_full_path = os.path.join(opt.save_dir, image_relative_path)
            fake_image.save(image_full_path)

            # 设置 HTML 页面中的图像路径
            vizer.set_cell(sem_id, col_id, image=image_relative_path)

    # 保存结果为 HTML
    save_path = os.path.join(opt.save_dir, 'dcgan_sefa.html')
    vizer.save(save_path)
    print(f"Results saved to {save_path}")


# 主函数
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
