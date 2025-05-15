CUDA_VISIBLE_DEVICES=0 nohup python \
    Miche_eval_advanced.py \
    --txt_path /data7/haolin/TeethData/RD_3/val.txt \
    --cfg_path output/ae/kl_d512_m512_l64/config.yaml \
    --pth output/ae/kl_d512_m512_l64_resume/checkpoint-8560.pth > eval.log 2>&1 &