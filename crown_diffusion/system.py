import craftsman
import torch
from craftsman.utils.typing import *
from diffusers import DDIMScheduler
from tqdm import tqdm
from michelangelo.models.tsal.sal_perceiver import ShapeAsLatentPerceiver
import torch.nn.functional as F
from michelangelo.models.tsal.inference_utils import extract_geometry, geometry_iou
from functools import partial

class Latent2MeshOutput(object):

    def __init__(self):
        self.mesh_v = None
        self.mesh_f = None

def compute_snr(noise_scheduler, timesteps):
    """
    Computes SNR as per
    https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # Expand the tensors.
    # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # Compute SNR.
    snr = (alpha / sigma) ** 2
    return snr

def ddim_sample(ddim_scheduler: DDIMScheduler,
                diffusion_model: torch.nn.Module,
                shape: Union[List[int], Tuple[int]],
                cond: torch.FloatTensor,
                categories: Optional[torch.Tensor] = None,
                steps: int = 50,
                guidance_scale: float = 3.0,
                do_classifier_free_guidance: bool = True,
                generator: Optional[torch.Generator] = None,
                device: torch.device = "cuda",
                disable_prog: bool = True, 
                **kwargs):

    assert steps > 0, f"{steps} must > 0."

    # init latents
    bsz = cond.shape[0]
    if do_classifier_free_guidance:
        bsz = bsz // 2

    latents = torch.randn(
        (bsz, *shape),
        generator=generator,
        device=cond.device,
        dtype=cond.dtype,
    )
    # scale the initial noise by the standard deviation required by the scheduler
    latents = latents * ddim_scheduler.init_noise_sigma
    # set timesteps
    ddim_scheduler.set_timesteps(steps)
    timesteps = ddim_scheduler.timesteps.to(device)
    # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
    extra_step_kwargs = {
        "generator": generator
    }

    # reverse
    for i, t in enumerate(tqdm(timesteps, disable=disable_prog, desc="DDIM Sampling:", leave=False)):
        # expand the latents if we are doing classifier free guidance
        latent_model_input = (
            torch.cat([latents] * 2)
            if do_classifier_free_guidance
            else latents
        )
        # predict the noise residual
        timestep_tensor = torch.tensor([t], dtype=torch.long, device=device)
        timestep_tensor = timestep_tensor.expand(latent_model_input.shape[0])
        
        # Run both conditional and unconditional forward passes
        if do_classifier_free_guidance:
            # Unconditional forward pass
            noise_pred_uncond = diffusion_model(
                latent_model_input[:bsz], 
                timestep_tensor[:bsz], 
                cond[:bsz], 
                categories[:bsz] if categories is not None else None,
                unconditional=True,
                **kwargs
            )
            # Conditional forward pass
            noise_pred_cond = diffusion_model(
                latent_model_input[bsz:], 
                timestep_tensor[bsz:], 
                cond[bsz:],
                categories[bsz:] if categories is not None else None,
                unconditional=False,
                **kwargs
            )
            # Combine predictions
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        else:
            noise_pred = diffusion_model(
                latent_model_input, 
                timestep_tensor, 
                cond,
                categories if categories is not None else None,
                unconditional=False,
                **kwargs
            )

        # compute the previous noisy sample x_t -> x_t-1
        latents = ddim_scheduler.step(
            noise_pred, t, latents, **extra_step_kwargs
        ).prev_sample

        yield latents, t

class Crown_diffusion_system:
    def __init__(self, 
                 noise_scheduler_type: str = None, 
                 noise_scheduler: dict = None, 
                 denoise_scheduler_type: str = None, 
                 denoise_scheduler: dict = None, 
                 ae: ShapeAsLatentPerceiver = None, 
                 z_scale_factor: float = 1.0, 
                 snr_gamma: float = 5.0, 
                 loss_type: str = 'mse'):
        self.noise_scheduler = craftsman.find(noise_scheduler_type)(**noise_scheduler)
        self.denoise_scheduler = craftsman.find(denoise_scheduler_type)(**denoise_scheduler)
        self.shape_model = ae
        self.z_scale_factor = z_scale_factor
        self.snr_gamma = snr_gamma
        self.loss_type = loss_type

    def __call__(self, denoiser_model: torch.nn.Module, model_input: torch.FloatTensor, cond: torch.FloatTensor, **kwargs):
        with torch.no_grad():
            kl_embed, _, _ = self.shape_model.encode(model_input)
        latents = kl_embed * self.z_scale_factor

        # 3. sample noise that we"ll add to the latents
        noise = torch.randn_like(latents)
        bs = latents.shape[0]
        # 4. Sample a random timestep for each motion
        timesteps = torch.randint(
            0,
            self.noise_scheduler.num_train_timesteps,
            (bs,),
            device=latents.device,
        )
        timesteps = timesteps.long()

        # 5. add noise
        noisy_z = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # 6. diffusion model forward
        noise_pred = denoiser_model(noisy_z, timesteps, cond, **kwargs)

        # 7. compute loss
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise 
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Prediction Type: {self.noise_scheduler.prediction_type} not supported.")
        
        # 8. whether snr
        if self.snr_gamma == 0:
            if self.loss_type == "l1":
                loss = F.l1_loss(noise_pred, target, reduction="mean")
            elif self.loss_type in ["mse", "l2"]:
                loss = F.mse_loss(noise_pred, target, reduction="mean")
            else:
                raise ValueError(f"Loss Type: {self.loss_type} not supported.")
        else:
            # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
            # Since we predict the noise instead of x_0, the original formulation is slightly changed.
            # This is discussed in Section 4.2 of the same paper.
            snr = compute_snr(self.noise_scheduler, timesteps)
            mse_loss_weights = torch.stack([snr, self.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                dim=1
            )[0]
            if self.noise_scheduler.config.prediction_type == "epsilon":
                mse_loss_weights = mse_loss_weights / snr
            elif self.noise_scheduler.config.prediction_type == "v_prediction":
                mse_loss_weights = mse_loss_weights / (snr + 1)
            
            if self.loss_type == "l1":
                loss = F.l1_loss(noise_pred, target, reduction="none")
            elif self.loss_type in ["mse", "l2"]:
                loss = F.mse_loss(noise_pred, target, reduction="none")
            else:
                raise ValueError(f"Loss Type: {self.loss_type} not supported.")
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
            loss = loss.mean()
        return loss

    @torch.no_grad()
    def sample(self,
               denoiser_model: torch.nn.Module,
               sample_inputs: torch.FloatTensor,
               categories: Optional[torch.Tensor] = None,
               sample_times: int = 1,
               steps: Optional[int] = None,
               guidance_scale: Optional[float] = None,
               seed: Optional[int] = None,
               **kwargs):
        if guidance_scale is None:
            guidance_scale = 1.0
        do_classifier_free_guidance = guidance_scale != 1.0

        # Prepare conditions
        cond = sample_inputs
        if do_classifier_free_guidance:
            # Duplicate inputs for classifier-free guidance
            cond = torch.cat([cond] * 2)
            if categories is not None:
                categories = torch.cat([categories] * 2)

        outputs = []
        latents = None
        
        if seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(seed)
        else:
            generator = None

        for _ in range(sample_times):
            sample_loop = ddim_sample(
                self.denoise_scheduler,
                denoiser_model.eval(),
                shape=self.shape_model.latent_shape,
                cond=cond,
                categories=categories,
                steps=steps,
                guidance_scale=guidance_scale,
                do_classifier_free_guidance=do_classifier_free_guidance,
                device=self.shape_model.device,
                disable_prog=False,
                generator=generator, 
                **kwargs
            )
            for sample, t in sample_loop:
                latents = sample
            outputs.append(self.shape_model.decode(latents / self.z_scale_factor))

        return outputs
    
    def latent2mesh(self,
                    latents: torch.FloatTensor,
                    bounds: Union[Tuple[float], List[float], float] = 1.1,
                    octree_depth: int = 7,
                    num_chunks: int = 10000) -> List[Latent2MeshOutput]:

        """

        Args:
            latents: [bs, num_latents, dim]
            bounds:
            octree_depth:
            num_chunks:

        Returns:
            mesh_outputs (List[MeshOutput]): the mesh outputs list.

        """

        outputs = []

        geometric_func = partial(self.shape_model.query_geometry, latents=latents)

        # 2. decode geometry
        device = latents.device
        mesh_v_f, has_surface = extract_geometry(
            geometric_func=geometric_func,
            device=device,
            batch_size=len(latents),
            bounds=bounds,
            octree_depth=octree_depth,
            num_chunks=num_chunks,
            disable=False
        )

        # 3. decode texture
        for i, ((mesh_v, mesh_f), is_surface) in enumerate(zip(mesh_v_f, has_surface)):
            if not is_surface:
                outputs.append(None)
                continue

            out = Latent2MeshOutput()
            out.mesh_v = mesh_v
            out.mesh_f = mesh_f

            outputs.append(out)

        return outputs
    
    def cal_iou(self,
                latents: torch.FloatTensor,
                bounds: Union[Tuple[float], List[float], float] = 1.1,
                octree_depth: int = 7,
                num_chunks: int = 10000, 
                query_points: torch.Tensor = None, 
                occ_labels: torch.Tensor = None):
        geometric_func = partial(self.shape_model.query_geometry, latents=latents)

        device = latents.device
        iou = geometry_iou(
            geometric_func=geometric_func,
            device=device,
            batch_size=len(latents),
            bounds=bounds,
            octree_depth=octree_depth,
            num_chunks=num_chunks,
            disable=False, 
            query_points=query_points, 
            occ_labels=occ_labels
        )
        return iou