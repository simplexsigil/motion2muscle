#!/bin/bash

export MIA_MOTION_DATA="/lsdf/users/dschneider-kf3609/workspace/HumanML3D/MIAHML3D"
export MIA_DATA="/lsdf/data/activity/MIADatasetOfficial"

# export CUDA_VISIBLE_DEVICES=3

python train_mia.py \
--batch-size 128 \
--lr 2e-4 \
--total-iter 300000 \
--lr-scheduler 200000 \
--depth 3 \
--out-dir output/MIA/transformer \
--dataname mint \
--vq-act relu \
--quantizer ema_reset \
--loss-vel 0.5 \
--recons-loss l1_smooth \
--exp-name output/MIA/transformer/8_trans_layers_256_width_28frames_2 \
--print-iter 50 \
--eval-iter 1000 \
--window-size 28 \
--width 256 \
--transformer_layer 8