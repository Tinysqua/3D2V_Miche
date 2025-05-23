    CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 nohup torchrun \
    --nproc_per_node=7 Miche_main_ae.py \
    --accum_iter=2 \
    --config_path output/ae/kl_d512_m512_l64/config.yaml \
    --model kl_d512_m512_l64  \
    --resume output/ae/kl_d512_m512_l64/checkpoint-4240.pth \
    --output_dir output/ae/kl_d512_m512_l64_resume \
    --data_path /data7/haolin/TeethData/RD_3 \
    --log_dir output/ae/kl_d512_m512_l64_resume \
    --num_workers 60 \
    --point_cloud_size 8192 \
    --dist_eval \
    --batch_size 16 \
    --epochs 9000 \
    --kl_weight 0.001 \
    --warmup_epochs 100 > output.log 2>&1 &