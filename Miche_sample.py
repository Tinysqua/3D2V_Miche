# -*- coding: utf-8 -*-
import os
from typing import *
import argparse
from functools import partial

import numpy as np
import trimesh
import cv2

import torch
import pytorch_lightning as pl

from michelangelo.models.tsal.inference_utils import extract_geometry
from util.datasets import load_list_from_txt
import craftsman
import crown_diffusion
from omegaconf import OmegaConf
from util.shapenet import filter_points, tooth_mapping
from Miche_eval_advanced import load_surface
from util.shapenet import fps_sample_points, random_sample_points


def load_models(args) -> Tuple[torch.nn.Module, torch.nn.Module]:
    cfg = OmegaConf.load(args.cfg_path)
    ae = craftsman.find(cfg['shape_model_type'])(**cfg['shape_model'], dtype=torch.float32)
    ae.load_state_dict(torch.load(args.ae_pth, map_location='cpu')['model'], strict=True)
    ae = ae.cuda()
    ae = ae.eval()

    diffusion_system = craftsman.find(cfg['system_type'])(**cfg['system'], ae=ae)
    model = craftsman.find(cfg['denoiser_model_type'])(cfg['denoiser_model'])
    model.load_state_dict(torch.load(args.pth, map_location='cpu')['model'], strict=True)
    model = model.cuda()
    model = model.eval()

    return model, diffusion_system, cfg

def prepare_crown_pointcloud(jaw_point_folder, mesh_name, jaw_size=2048, average_sp=True, surface_sp_mode=None):
    # Define the paths for the upper and lower jaw point clouds
    upper_jaw_path = os.path.join(jaw_point_folder, "upper_" + mesh_name)
    lower_jaw_path = os.path.join(jaw_point_folder, "lower_" + mesh_name)

    # Load the upper jaw point cloud
    with np.load(upper_jaw_path) as input_pc:
        upper_jaw = input_pc['points']

    # Load the lower jaw point cloud
    with np.load(lower_jaw_path) as input_pc:
        lower_jaw = input_pc['points']

    whole_jaw_np = np.concatenate([upper_jaw, lower_jaw], axis=0)
    whole_jaw_np = whole_jaw_np * 2.0  # Important!! 
    if average_sp:
        # Randomly sample points from the upper jaw
        rng = np.random.default_rng()
        ind = rng.choice(whole_jaw_np.shape[0], jaw_size, replace=False)
        whole_jaw = torch.from_numpy(whole_jaw_np[ind])

    else:
        in_range_mask = filter_points(whole_jaw_np)
        in_range_points = whole_jaw_np[in_range_mask]
        out_range_points = whole_jaw_np[~in_range_mask]
        # Sample points within the range
        if len(in_range_points) >= jaw_size:
            # in_range_sampled = in_range_points[np.random.default_rng().choice(len(in_range_points), jaw_size, replace=False)]
            in_range_sampled = fps_sample_points(in_range_points, jaw_size)
        else:
            in_range_sampled = in_range_points
            # out_range_to_add = out_range_points[np.random.default_rng().choice(len(out_range_points), jaw_size - len(in_range_points), replace=False)]
            out_range_to_add = fps_sample_points(out_range_points, jaw_size - len(in_range_points))
            in_range_sampled = np.concatenate([in_range_sampled, out_range_to_add], axis=0)
        whole_jaw = torch.from_numpy(in_range_sampled)

    return whole_jaw.unsqueeze(0).cuda()

def save_output(args, mesh_outputs, mesh_name):
    
    os.makedirs(args.output_dir, exist_ok=True)
    for i, mesh in enumerate(mesh_outputs):
        mesh.mesh_f = mesh.mesh_f[:, ::-1]
        mesh_output = trimesh.Trimesh(mesh.mesh_v, mesh.mesh_f)

        name = f"out_{mesh_name}.obj"
        mesh_output.export(os.path.join(args.output_dir, name), include_normals=True)

    print(f'-----------------------------------------------------------------------------')
    print(f'>>> Finished and mesh {name} saved in {args.output_dir}')
    print(f'-----------------------------------------------------------------------------')        

    return 0

def crown_mesh_generation(args, model:crown_diffusion.Crown_denoiser, diffusion_system:crown_diffusion.Crown_diffusion_system, cfg: dict, 
                          box_v=1.0, octree_depth=7): 
    mesh_list = load_list_from_txt(args.txt_path)
    parrent_path = os.path.join(os.path.dirname(args.txt_path), 'jaw_pointcloud')
    surface_path = os.path.join(os.path.dirname(args.txt_path), 'crown_4_pointcloud')
    for i, mesh_name in enumerate(mesh_list):
        if i+1 > 20:
            break
        jaw_pointcloud = prepare_crown_pointcloud(parrent_path, mesh_name, **cfg['dataset_opt'])

        surface = load_surface(os.path.join(surface_path, mesh_name))
        crown_cate = torch.LongTensor([tooth_mapping[mesh_name.split('_')[-1][:-4]]]).cuda()
        crown_cate = crown_cate.unsqueeze(0)
        # Generate the crown with classifier-free guidance
        decoded_latents = diffusion_system.sample(
            model, 
            jaw_pointcloud, 
            categories=crown_cate,
            sample_times=1, 
            steps=500,
            guidance_scale=args.guidance_scale
        )[0]
        mesh_output = diffusion_system.latent2mesh(decoded_latents, bounds=box_v, octree_depth=octree_depth)
        save_output(args, mesh_output, mesh_name[:-4])

        # Print the GT from VAE
        latents, _, _ = diffusion_system.shape_model.encode(surface, None, sample_posterior=True)
        latents = diffusion_system.shape_model.decode(latents)
        geometric_func = partial(diffusion_system.shape_model.query_geometry, latents=latents)

        mesh_v_f, has_surface = extract_geometry(
            geometric_func=geometric_func,
            device=surface.device,
            batch_size=surface.shape[0],
            bounds=(-1.00, -1.00, -1.00, 1.00, 1.00, 1.00),
            octree_depth=octree_depth,
            num_chunks=10000,
        )

        recon_mesh = trimesh.Trimesh(mesh_v_f[0][0], mesh_v_f[0][1])
        recon_mesh.export(os.path.join(args.output_dir, f"gt_{mesh_name[:-4]}.obj")) 


if __name__ == "__main__":
    '''
    Make crown generation from the upper/lower jaws
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--ae_pth", type=str, required=True)
    parser.add_argument("--pth", type=str, required=True)
    parser.add_argument("--txt_path", type=str, default='./*.txt', help='Path to the txt file that record names of npz')
    parser.add_argument("--output_dir", type=str, default='./output/eval')
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("--cfg_path", type=str, required=True)
    parser.add_argument("--guidance_scale", type=float, default=7.5, help='Classifier-free guidance scale (1.0 means no guidance)')
    args = parser.parse_args()
    
    pl.seed_everything(args.seed)

    print(f'-----------------------------------------------------------------------------')
    print(f'>>> Running diffusion')
    args.output_dir = os.path.join(args.output_dir, 'diffusion')
    print(f'>>> Output directory: {args.output_dir}')
    print(f'-----------------------------------------------------------------------------')
    
    model, diffusion_system, cfg = load_models(args)
    crown_mesh_generation(args, model, diffusion_system, cfg)