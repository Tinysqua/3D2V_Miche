import os
import torch
import random
from .shapenet import DiyShapeNet, CondShapeNet

def split_list(input_list, split_ratio=0.8):
    random.shuffle(input_list)
    
    split_index = int(len(input_list) * split_ratio)
    
    list1 = input_list[:split_index]
    list2 = input_list[split_index:]
    
    return list1, list2

def load_list_from_txt(filename):
    """
    Read the content from a txt file and convert it into a list
    """
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        lst = [line.strip() for line in lines]
    return lst

class AxisScaling(object):
    def __init__(self, interval=(0.75, 1.25), jitter=True):
        assert isinstance(interval, tuple)
        self.interval = interval
        self.jitter = jitter
        
    def __call__(self, surface, point):
        scaling = torch.rand(1, 3) * 0.5 + 0.75
        surface = surface * scaling
        point = point * scaling

        scale = (1 / torch.abs(surface).max().item()) * 0.999999
        surface *= scale
        point *= scale

        if self.jitter:
            surface += 0.005 * torch.randn_like(surface)
            surface.clamp_(min=-1, max=1)

        return surface, point


# def build_shape_surface_occupancy_dataset(split, args):
#     if split == 'train':
#         # transform = #transforms.Compose([
#         transform = AxisScaling((0.75, 1.25), True)
#         # ])
#         return ShapeNet(args.data_path, split=split, transform=transform, sampling=True, num_samples=1024, return_surface=True, surface_sampling=True, pc_size=args.point_cloud_size)
#     elif split == 'val':
#         # return ShapeNet(args.data_path, split=split, transform=None, sampling=True, num_samples=1024, return_surface=True, surface_sampling=True, pc_size=args.point_cloud_size)
#         return ShapeNet(args.data_path, split=split, transform=None, sampling=False, return_surface=True, surface_sampling=True, pc_size=args.point_cloud_size)
#     else:
#         return ShapeNet(args.data_path, split=split, transform=None, sampling=False, return_surface=True, surface_sampling=True, pc_size=args.point_cloud_size)

def build_shape_surface_occupancy_dataset(args, if_transform=True, surface_sp_mode=None):
    # crowns_name = [f for f in os.listdir(os.path.join(args.data_path, 'crown_occ'))]
    # train_crowns_name, val_crowns_name = split_list(crowns_name)
    train_crowns_name = load_list_from_txt(os.path.join(args.data_path, 'train.txt'))
    val_crowns_name = load_list_from_txt(os.path.join(args.data_path, 'val.txt'))

    transform = None
    if if_transform:
        transform = AxisScaling((0.75, 1.25), False)

    assert surface_sp_mode is not None, "Surface_sp_mode should not be None! Something wrong!"
    train_dataset = DiyShapeNet(args.data_path, split='train', crowns_list=train_crowns_name, transform=transform, sampling=True, num_samples=args.point_cloud_size // 2, 
                                return_surface=True, surface_sampling=surface_sp_mode, pc_size=args.point_cloud_size)
    val_dataset = DiyShapeNet(args.data_path, split='val', crowns_list=val_crowns_name, transform=None, sampling=True, num_samples=100000, 
                              return_surface=True, surface_sampling=surface_sp_mode, pc_size=args.point_cloud_size)
    return train_dataset, val_dataset

def conditional_build_shape_surface_occupancy_dataset(args, surface_sp_mode=None, **kwargs): 
    train_crowns_name = load_list_from_txt(os.path.join(args.data_path, 'train.txt'))
    val_crowns_name = load_list_from_txt(os.path.join(args.data_path, 'val.txt'))

    assert surface_sp_mode is not None, "Surface_sp_mode should not be None! Something wrong!"
    train_dataset = CondShapeNet(args.data_path, split='train', crowns_list=train_crowns_name, 
                                surface_sampling=surface_sp_mode, 
                                pc_size=args.point_cloud_size, **kwargs)
    val_dataset = CondShapeNet(args.data_path, split='val', crowns_list=val_crowns_name, 
                               surface_sampling=surface_sp_mode, 
                               pc_size=args.point_cloud_size, **kwargs)
    return train_dataset, val_dataset
    

# if __name__ == '__main__':
#     # m = ShapeNet('/home/zhanb0b/data/', 'train', transform=AxisScaling(), sampling=True, num_samples=1024, return_surface=True, surface_sampling=True)
#     m = ShapeNet('/home/zhanb0b/data/', 'train', transform=AxisScaling(), sampling=True, num_samples=1024, return_surface=True, surface_sampling=True)
#     p, l, s, c = m[0]
#     print(p.shape, l.shape, s.shape, c)
#     print(p.max(dim=0)[0], p.min(dim=0)[0])
#     print(p[l==1].max(axis=0)[0], p[l==1].min(axis=0)[0])
#     print(s.max(axis=0)[0], s.min(axis=0)[0])