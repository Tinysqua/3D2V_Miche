import os
import numpy as np
import torch
import trimesh
import sys;sys.path.append('./')
from torch.utils.data import DataLoader
from util.shapenet import DiyShapeNet
from util.datasets import load_list_from_txt, AxisScaling
from torch_cluster import fps
from tqdm import tqdm  # 导入tqdm
# 假设已经定义了DiagonalGaussianDistribution类和fps函数
def process_and_save_pointcloud(dataset, index, output_folder):
    # 获取点云数据
    surface, _, points, _ = dataset[index]
    points = points.unsqueeze(0)  # 将点云扩展到批次大小为1
    
    # 执行FPS采样
    B, N, D = points.shape
    # assert N == dataset.num_samples
    
    flattened = points.view(B*N, D)
    batch = torch.arange(B).to(points.device)
    batch = torch.repeat_interleave(batch, N)
    pos = flattened
    ratio = 1.0 * 512 / N  # 假设我们希望采样到pc_size个点
    idx = fps(pos, batch, ratio=ratio)
    sampled_pc = pos[idx]
    sampled_pc = sampled_pc.view(B, -1, 3)
    # 将采样后的点云转为numpy数组并保存为OBJ
    sampled_pc_np = sampled_pc.cpu().numpy().squeeze()
    
    # 使用trimesh创建点云并保存为OBJ文件
    pc_mesh = trimesh.points.PointCloud(sampled_pc_np)
    output_file = os.path.join(output_folder, f"sampled_crown_{index}.obj")
    pc_mesh.export(output_file)
# 创建数据集和数据加载器
dataset_folder = '/data7/haolin/TeethData/RD_3'
split = 'train'  # 或 'test'
crowns_list = load_list_from_txt('/data7/haolin/TeethData/RD_3/train.txt')
transform = AxisScaling((0.75, 1.25), True)
dataset = DiyShapeNet(dataset_folder, split='train', crowns_list=crowns_list, transform=transform, sampling=True, num_samples=1024, return_surface=True, surface_sampling=True, pc_size=2048)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
# 保存文件夹
output_folder = 'output/fps'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
# 使用tqdm包装循环，并限制循环次数
for idx in tqdm(range(min(10, len(dataset))), desc="Processing"):
    process_and_save_pointcloud(dataset, idx, output_folder)
    print(f"Saved sampled point cloud for index {idx}")