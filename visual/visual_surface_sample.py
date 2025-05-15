import os
import numpy as np
import torch
import sys;sys.path.append('./')
from util.shapenet import DiyShapeNet
from util.datasets import load_list_from_txt
import argparse
import trimesh

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/data7/haolin/TeethData/RD_3', help='dataset path')
    parser.add_argument('--output_dir', type=str, default='output/surface_points', help='output directory')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    val_crowns_name = load_list_from_txt(os.path.join(args.data_path, 'val.txt'))
    pc_size = 13984
    dataset = DiyShapeNet(args.data_path, split='test', crowns_list=val_crowns_name, transform=None, sampling=False, 
                          return_surface=True, surface_sampling=True, pc_size=pc_size)
    
    count = 0
    for i in range(len(dataset)):
        if count >= 20:
            break
        points, labels, surface, _ = dataset[i]
        surface_points = surface.numpy()
        point_cloud = trimesh.PointCloud(surface_points)
        point_cloud.export(os.path.join(args.output_dir, f'surface_{i}.ply'))
        count += 1

if __name__ == '__main__':
    main()
