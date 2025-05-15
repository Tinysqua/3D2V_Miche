from dataclasses import dataclass
# import sys;sys.path.append('./')
import torch
import torch.nn as nn
import math

from typing import Optional
from craftsman.utils.base import BaseModule
from craftsman.models.denoisers.utils import *
from michelangelo.models.modules.embedder import FourierEmbedder
from util.shapenet import tooth_mapping

class Crown_denoiser(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        pretrained_model_name_or_path: Optional[str] = None
        input_channels: int = 32
        output_channels: int = 32
        width: int = 768
        layers: int = 28
        heads: int = 16
        context_ln: bool = True
        init_scale: float = 0.25
        use_checkpoint: bool = False
        drop_path: float = 0.
        crown_cate: bool = False

    cfg: Config

    def configure(self) -> None:
        super().configure()

        # timestep embedding
        self.time_embed = TimestepEmbedder(self.cfg.width)

        # x embedding
        self.x_embed = nn.Linear(self.cfg.input_channels, self.cfg.width, bias=True)

        self.fourier_embedder = FourierEmbedder(num_freqs=8, include_pi=True)

        if self.cfg.context_ln:
            self.clip_embed = nn.Sequential(
                nn.LayerNorm(self.fourier_embedder.out_dim),
                nn.Linear(self.fourier_embedder.out_dim, self.cfg.width),
            )

        else:
            self.clip_embed = nn.Linear(self.fourier_embedder.out_dim, self.cfg.width)

        if self.cfg.crown_cate:
            unique_count = len(set(tooth_mapping.values()))
            self.category_emb = nn.Embedding(unique_count, self.cfg.width) # Since the numbers of teeth is fixed, so hard code 

        init_scale = self.cfg.init_scale * math.sqrt(1.0 / self.cfg.width)
        drop_path = [x.item() for x in torch.linspace(0, self.cfg.drop_path, self.cfg.layers)]
        self.blocks = nn.ModuleList([
            DiTBlock(
                    width=self.cfg.width, 
                    heads=self.cfg.heads, 
                    init_scale=init_scale, 
                    qkv_bias=self.cfg.drop_path, 
                    use_flash=True,
                    drop_path=drop_path[i]
            )
            for i in range(self.cfg.layers)
        ])

        self.t_block = nn.Sequential(
                        nn.SiLU(),
                        nn.Linear(self.cfg.width, 6 * self.cfg.width, bias=True)
                        )
        
        # final layer
        self.final_layer = T2IFinalLayer(self.cfg.width, self.cfg.output_channels)

        self.identity_initialize()

        if self.cfg.pretrained_model_name_or_path:
            print(f"Loading pretrained model from {self.cfg.pretrained_model_name_or_path}")
            ckpt = torch.load(self.cfg.pretrained_model_name_or_path, map_location="cpu")['state_dict']
            self.denoiser_ckpt = {}
            for k, v in ckpt.items():
                if k.startswith('denoiser_model.'):
                    self.denoiser_ckpt[k.replace('denoiser_model.', '')] = v
            self.load_state_dict(self.denoiser_ckpt, strict=False)

    def identity_initialize(self):
        for block in self.blocks:
            nn.init.constant_(block.attn.c_proj.weight, 0)
            nn.init.constant_(block.attn.c_proj.bias, 0)
            nn.init.constant_(block.cross_attn.c_proj.weight, 0)
            nn.init.constant_(block.cross_attn.c_proj.bias, 0)
            nn.init.constant_(block.mlp.c_proj.weight, 0)
            nn.init.constant_(block.mlp.c_proj.bias, 0)

    def forward(self,
                model_input: torch.FloatTensor,
                timestep: torch.LongTensor,
                point_cloud: torch.FloatTensor, 
                categories: torch.Tensor = None,
                unconditional: bool = False):

        r"""
        Args:
            model_input (torch.FloatTensor): [bs, n_data, c]
            timestep (torch.LongTensor): [bs,]
            point_cloud (torch.FloatTensor): [bs, context_tokens, 3] Since it is the point cloud for our project
            categories (torch.Tensor): [bs,] category labels
            unconditional (bool): Whether to do unconditional generation (for CFG)

        Returns:
            sample (torch.FloatTensor): [bs, n_data, c]
        """

        B, n_data, _ = model_input.shape

        # 1. time
        t_emb = self.time_embed(timestep)

        # 2. conditions projector
        if not unconditional:
            point_cloud = self.fourier_embedder(point_cloud)
            visual_cond = self.clip_embed(point_cloud)

            if self.cfg.crown_cate and categories is not None:
                visual_cond = torch.cat([self.category_emb(categories), visual_cond], dim=1)
        else:
            # For unconditional generation, use zero embeddings
            if self.cfg.crown_cate:
                # Create zero embeddings with same shape as category + point cloud embeddings
                visual_cond = torch.zeros((B, 1 + point_cloud.shape[1], self.cfg.width), device=model_input.device)
            else:
                # Create zero embeddings with same shape as point cloud embeddings only
                visual_cond = torch.zeros((B, point_cloud.shape[1], self.cfg.width), device=model_input.device)

        # 4. denoiser
        latent = self.x_embed(model_input)

        t0 = self.t_block(t_emb).unsqueeze(dim=1)

        for block in self.blocks:
            latent = auto_grad_checkpoint(block, latent, visual_cond, t0)

        latent = self.final_layer(latent, t_emb)

        return latent
    
if __name__ == "__main__":
    kwargs = dict(input_channels=512, output_channels=512, width=512, layers=20)
    model = Crown_denoiser(kwargs)
    pc_latent = torch.randn((4, 512, 512))
    time = torch.randn((4,))
    point_cloud = torch.randn((4, 2048, 3))
    output = model(pc_latent, time, point_cloud)
    print(output.shape)
