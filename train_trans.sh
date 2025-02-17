#!/bin/bash

export MINT_DATA="/lsdf/data/activity/MuscleSim/musclesim_dataset"
export MOTION_DATA="/lsdf/users/dschneider-kf3609/workspace/HumanML3D/HumanML3D"
export BABEL_DATA="/lsdf/data/activity/BABEL/babel_v1-0_release"

# export CUDA_VISIBLE_DEVICES=3

python train_trans.py \
--batch-size 128 \
--lr 2e-4 \
--total-iter 300000 \
--lr-scheduler 200000 \
--depth 3 \
--out-dir output/transformer \
--dataname mint \
--vq-act relu \
--quantizer ema_reset \
--loss-vel 0.5 \
--recons-loss l1_smooth \
--exp-name utput/transformer/16_trans_layers_256_width_28frames_2 \
--print-iter 50 \
--eval-iter 1000 \
--window-size 28 \
--width 256 \
--muscle_subset MUSINT_402 \
--transformer_layer 16