CUDA_VISIBLE_DEVICES=0 nohup python Miche_sample.py \
    --txt_path /data7/haolin/TeethData/RD_4/val.txt \
    --ae_pth output/ae/kl_d512_m512_l64_fpsmore/checkpoint-6880.pth \
    --pth output/diffusion/dit_1024_24_16_512_cfg/checkpoint-6999.pth \
    --cfg_path configs/config.yaml \
    --guidance_scale 1.0 \
    > eval.log 2>&1 &