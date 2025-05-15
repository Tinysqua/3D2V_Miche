# -*- coding: utf-8 -*-
import os
import shutil
import time
import argparse
from functools import partial
from scipy.spatial import cKDTree as KDTree

from einops import repeat, rearrange
import numpy as np
import trimesh
import cv2

import torch
import pytorch_lightning as pl

from michelangelo.models.tsal.inference_utils import extract_geometry
from util.datasets import load_list_from_txt
# from michelangelo.utils.misc import get_config_from_file, instantiate_from_config
from michelangelo.models.tsal.sal_perceiver import ShapeAsLatentPerceiver
from omegaconf import OmegaConf
import craftsman



def load_model(args):
    cfg = OmegaConf.load(args.cfg_path)
    model = craftsman.find(cfg['shape_model_type'])(**cfg['shape_model'], dtype=torch.float32)
    model.load_state_dict(torch.load(args.pth, map_location='cpu')['model'], strict=True)
    model = model.cuda()
    model = model.eval()

    return model, cfg

def load_surface(pointcloud_path, pc_size=8192, surface_sp_mode='random'):
    
    with np.load(pointcloud_path) as input_pc:
        surface = input_pc['points']
    
    if surface_sp_mode == 'random':
        rng = np.random.default_rng()
        ind = rng.choice(surface.shape[0], pc_size, replace=False)
        surface = torch.FloatTensor(surface[ind])
    elif surface_sp_mode == 'fps':
        import fpsample
        kdline_fps_samples_idx = fpsample.bucket_fps_kdline_sampling(surface[:, :3], pc_size, h=5)
        surface = torch.FloatTensor(surface[kdline_fps_samples_idx])
    surface = surface.unsqueeze(0).cuda()
    return surface

def cal_chamfer(pred_mesh:trimesh.Trimesh, gt_surface:torch.FloatTensor):
    pred = pred_mesh.sample(100000)
    tree = KDTree(pred)
    dist, _ = tree.query(gt_surface[0].cpu().numpy())
    d1 = dist
    gt_to_gen_chamfer = np.mean(dist)

    tree = KDTree(gt_surface[0].cpu().numpy())
    dist, _ = tree.query(pred)
    d2 = dist
    gen_to_gt_chamfer = np.mean(dist)

    cd = gt_to_gen_chamfer + gen_to_gt_chamfer
    return cd


def reconstruction(args, model:ShapeAsLatentPerceiver, cfg=None, bounds=(-1.00, -1.00, -1.00, 1.00, 1.00, 1.00), octree_depth=7, num_chunks=10000):

    mesh_list = load_list_from_txt(args.txt_path)
    parrent_path = os.path.join(os.path.dirname(args.txt_path), 'crown_4_pointcloud')
    gt_path = os.path.join(os.path.dirname(args.txt_path), 'watertight_meshes')
    for i, mesh_name in enumerate(mesh_list):
        if i+1 > 20:
            break
        mesh_path = os.path.join(parrent_path, mesh_name)
        gt_mesh_path = os.path.join(gt_path, mesh_name.replace('.npz', '.obj'))

        surface = load_surface(mesh_path, surface_sp_mode=cfg['dataset_opt']['surface_sp_mode'])
        
        # encoding
        # shape_embed, shape_latents = model.model.encode_shape_embed(surface, return_latents=True)    
        # shape_zq, posterior = model.model.shape_model.encode_kl_embed(shape_latents)
        latents, _, _ = model.encode(surface, None, sample_posterior=True)

        # decoding
        latents = model.decode(latents)
        geometric_func = partial(model.query_geometry, latents=latents)
        
        # reconstruction
        mesh_v_f, has_surface = extract_geometry(
            geometric_func=geometric_func,
            device=surface.device,
            batch_size=surface.shape[0],
            bounds=bounds,
            octree_depth=octree_depth,
            num_chunks=num_chunks,
        )
        recon_mesh = trimesh.Trimesh(mesh_v_f[0][0], mesh_v_f[0][1])
        
        # save
        os.makedirs(args.output_dir, exist_ok=True)
        recon_mesh.export(os.path.join(args.output_dir, f"crown_{i+1}.obj"))
        if args.save_gt:
            shutil.copy(gt_mesh_path, os.path.join(args.output_dir, f"gt_{i+1}.obj"))
        
        print(f'-----------------------------------------------------------------------------')
        print(f'>>> Finished and mesh saved in {os.path.join(args.output_dir, f"crown_{i+1}.obj")}')
        print(f'-----------------------------------------------------------------------------')

        chamfer_dis = cal_chamfer(recon_mesh, surface)

        print(f'-----------------------------------------------------------------------------')
        print(f'>>> Mesh to Gen chamfer distance: {chamfer_dis}')
        print(f'-----------------------------------------------------------------------------')


    
    return 0



task_dick = {
    'reconstruction': reconstruction,
}

if __name__ == "__main__":
    '''
    1. Reconstruct point cloud
    2. Image-conditioned generation
    3. Text-conditioned generation
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--pth", type=str, required=True)
    parser.add_argument("--task", type=str, default='reconstruction')
    parser.add_argument("--txt_path", type=str, default='./*.txt', help='Path to the txt file that record names of npz')
    parser.add_argument("--output_dir", type=str, default='./output')
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("--cfg_path", type=str, required=True)
    parser.add_argument('--save_gt', action='store_true')
    args = parser.parse_args()
    
    pl.seed_everything(args.seed)

    print(f'-----------------------------------------------------------------------------')
    print(f'>>> Running {args.task}')
    args.output_dir = os.path.join(args.output_dir, args.task)
    print(f'>>> Output directory: {args.output_dir}')
    print(f'-----------------------------------------------------------------------------')
    
    model, cfg = load_model(args)
    task_dick[args.task](args, model, cfg)