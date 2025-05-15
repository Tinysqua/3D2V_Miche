import os
import glob
import random

import yaml 

import torch
from torch.utils import data

import numpy as np


# tooth_mapping = {
#     '18': 0, '28': 0,
#     '17': 1, '27': 1,
#     '16': 2, '26': 2,
#     '15': 3, '25': 3,
#     '14': 4, '24': 4,
#     '13': 5, '23': 5,
#     '12': 6, '22': 6,
#     '11': 7, '21': 7,
#     '48': 8, '38': 8,
#     '47': 9, '37': 9,
#     '46': 10, '36': 10,
#     '45': 11, '35': 11,
#     '44': 12, '34': 12,
#     '43': 13, '33': 13,
#     '42': 14, '32': 14,
#     '41': 15, '31': 15
# }

tooth_mapping = {
    '11': 0,
    '12': 1,
    '13': 2,
    '14': 3,
    '15': 4,
    '16': 5,
    '17': 6,
    '18': 7,
    '21': 8,
    '22': 9,
    '23': 10,
    '24': 11,
    '25': 12,
    '26': 13,
    '27': 14,
    '28': 15,
    '31': 16,
    '32': 17,
    '33': 18,
    '34': 19,
    '35': 20,
    '36': 21,
    '37': 22,
    '38': 23,
    '41': 24,
    '42': 25,
    '43': 26,
    '44': 27,
    '45': 28,
    '46': 29,
    '47': 30,
    '48': 31
}




def filter_points(points, range_value=1.5):
    """Filter points within the specified range."""
    x_range = (-range_value, range_value)
    y_range = (-range_value, range_value)
    z_range = (-range_value, range_value)

    mask = (points[:, 0] >= x_range[0]) & (points[:, 0] <= x_range[1]) & \
           (points[:, 1] >= y_range[0]) & (points[:, 1] <= y_range[1]) & \
           (points[:, 2] >= z_range[0]) & (points[:, 2] <= z_range[1])
    return mask

def random_sample_points(points, sample_size):
    return points[np.random.default_rng().choice(len(points), sample_size, replace=False)]

def fps_sample_points(points, sample_size):
    import fpsample
    assert points.shape[1] == 3, "Shape of point cloud should be three"
    kdline_fps_samples_idx = fpsample.bucket_fps_kdline_sampling(points, sample_size, h=5)
    return points[kdline_fps_samples_idx]

class DiyShapeNet(data.Dataset):
    def __init__(self, dataset_folder, split, crowns_list, transform=None, sampling=True, num_samples=4096, return_surface=True, surface_sampling=True, pc_size=2048, replica=16):
        self.pc_size = pc_size

        self.transform = transform
        self.num_samples = num_samples
        self.sampling = sampling
        self.split = split

        self.dataset_folder = dataset_folder
        self.return_surface = return_surface
        self.surface_sampling = surface_sampling

        self.dataset_folder = dataset_folder
        self.point_folder = os.path.join(self.dataset_folder, 'crown_occ')
        self.mesh_folder = os.path.join(self.dataset_folder, 'crown_4_pointcloud')

        self.crowns_name = crowns_list

    
    def __getitem__(self, index):
        crown_name = self.crowns_name[index]
        point_path = os.path.join(self.point_folder, crown_name)

        try:
            with np.load(point_path) as data:
                vol_points = data['vol_points']
                vol_label = data['vol_label']
                near_points = data['near_points']
                near_label = data['near_label']
        except Exception as e:
            print(e)
            print(point_path)

        if self.return_surface:
            pc_path = os.path.join(self.mesh_folder, crown_name)
            with np.load(pc_path) as data:
                surface = data['points'].astype(np.float32)
                surface = surface * 1.0

            if self.surface_sampling:
                if self.surface_sampling == 'random':
                    ind = np.random.default_rng().choice(surface.shape[0], self.pc_size, replace=False)
                    surface = surface[ind]
                elif self.surface_sampling == 'fps':
                    import fpsample
                    kdline_fps_samples_idx = fpsample.bucket_fps_kdline_sampling(surface[:, :3], self.pc_size, h=5)
                    surface = surface[kdline_fps_samples_idx]
            surface = torch.from_numpy(surface)

        if self.sampling:
            ind = np.random.default_rng().choice(vol_points.shape[0], self.num_samples, replace=False)
            vol_points = vol_points[ind]
            vol_label = vol_label[ind]

            ind = np.random.default_rng().choice(near_points.shape[0], self.num_samples, replace=False)
            near_points = near_points[ind]
            near_label = near_label[ind]

        
        vol_points = torch.from_numpy(vol_points)
        vol_label = torch.from_numpy(vol_label).float()

        if self.split == 'train':
            near_points = torch.from_numpy(near_points)
            near_label = torch.from_numpy(near_label).float()


            points = torch.cat([vol_points, near_points], dim=0)
            labels = torch.cat([vol_label, near_label], dim=0)
        else:
            points = vol_points
            labels = vol_label

        if self.transform:
            surface, points = self.transform(surface, points)

        if self.return_surface:
            return points, labels, surface, 0
        else:
            return points, labels, 0
        
    

    def __len__(self):
        return len(self.crowns_name)
    

class CondShapeNet(data.Dataset):
    def __init__(self, dataset_folder, split, crowns_list, 
                 surface_sampling=True, pc_size=2048, 
                 crown_scale=1.0, jaw_scale = 2.0, jaw_size=2048, average_sp=True):
        self.pc_size = pc_size

        self.split = split

        self.dataset_folder = dataset_folder
        self.surface_sampling = surface_sampling

        self.dataset_folder = dataset_folder
        self.point_folder = os.path.join(self.dataset_folder, 'crown_occ')
        self.mesh_folder = os.path.join(self.dataset_folder, 'crown_4_pointcloud')
        self.jaw_point_folder = os.path.join(self.dataset_folder, 'jaw_pointcloud') 

        self.crowns_name = crowns_list
        self.crown_scale = crown_scale
        self.jaw_size = jaw_size
        self.average_sp = average_sp
        self.jaw_scale = jaw_scale

    
    def __getitem__(self, index):
        crown_name = self.crowns_name[index]

        # sample surface 
        pc_path = os.path.join(self.mesh_folder, crown_name)
        with np.load(pc_path) as data:
            surface = data['points'].astype(np.float32)
            surface = surface * self.crown_scale  # Important!! 

        if self.surface_sampling:
            if self.surface_sampling == 'random':
                ind = np.random.default_rng().choice(surface.shape[0], self.pc_size, replace=False)
                surface = surface[ind]
            elif self.surface_sampling == 'fps':
                import fpsample
                kdline_fps_samples_idx = fpsample.bucket_fps_kdline_sampling(surface[:, :3], self.pc_size, h=5)
                surface = surface[kdline_fps_samples_idx]
        surface = torch.from_numpy(surface)


        # sample condition surface 
        upper_jaw_path = os.path.join(self.jaw_point_folder, "upper_"+crown_name)
        lower_jaw_path = os.path.join(self.jaw_point_folder, "lower_"+crown_name)

        with np.load(upper_jaw_path) as data:
            upper_jaw = data['points'].astype(np.float32)
            upper_jaw = upper_jaw * self.jaw_scale  # Important!! 

        with np.load(lower_jaw_path) as data:
            lower_jaw = data['points'].astype(np.float32)
            lower_jaw = lower_jaw * self.jaw_scale  # Important!! 

        whole_jaw_np = np.concatenate([upper_jaw, lower_jaw], axis=0)
        whole_jaw = self.jaw_pc_sample(whole_jaw_np, average_sp=self.average_sp)


        crown_name = crown_name.split('.')[0]
        crown_info = dict(crown_name=crown_name, crown_cate=torch.LongTensor([tooth_mapping[crown_name.split('_')[-1]]]))

        return 0, 0, surface, whole_jaw, crown_info
        
    
    def jaw_pc_sample(self, whole_jaw_np, average_sp=True):
        whole_jaw = None
        if average_sp:
            ind = np.random.default_rng().choice(whole_jaw_np.shape[0], self.jaw_size, replace=False)
            whole_jaw = torch.from_numpy(whole_jaw_np[ind])
        else:
            in_range_mask = filter_points(whole_jaw_np)
            in_range_points = whole_jaw_np[in_range_mask]
            out_range_points = whole_jaw_np[~in_range_mask]
            # Sample points within the range
            if len(in_range_points) >= self.jaw_size:
                in_range_sampled = fps_sample_points(in_range_points, self.jaw_size)
            else:
                in_range_sampled = in_range_points
                out_range_to_add = fps_sample_points(out_range_points, self.jaw_size - len(in_range_points))
                in_range_sampled = np.concatenate([in_range_sampled, out_range_to_add], axis=0)

            whole_jaw = torch.from_numpy(in_range_sampled)

        return whole_jaw

    def __len__(self):
        return len(self.crowns_name)
