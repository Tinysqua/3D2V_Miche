    CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 nohup torchrun \
    --nproc_per_node=7 Miche_main_class_cond.py \
    --accum_iter 2 --data_path /data7/haolin/TeethData/RD_4 \
    --config_path configs/config.yaml \
    --log_dir output/diffusion/dit_1024_24_16_512_cfg \
    --output_dir output/diffusion/dit_1024_24_16_512_cfg \
    --lr 0.0001 --pin_mem --dist_eval \
    --point_cloud_size 15360 --num_workers 32 \
    --resume output/diffusion/dit_1024_24_16_512_cfg/checkpoint-5760.pth \
    --ae-pth output/ae/kl_d512_m512_l64_fpsmore/checkpoint-6880.pth \
    --batch_size 7 \
    --epochs 7000 \
    --save_freq 40 \
    --warmup_epochs 100 > output.log 2>&1 &
    
