#!/bin/bash

export MINT_DATA="/lsdf/data/activity/MuscleSim/musclesim_dataset"
export MOTION_DATA="/lsdf/users/dschneider-kf3609/workspace/HumanML3D/HumanML3D"
export BABEL_DATA="/lsdf/data/activity/BABEL/babel_v1-0_release"

export CUDA_VISIBLE_DEVICES=0

python3 ms_eval.py \
--batch-size 128 \
--lr 2e-4 \
--total-iter 3000000 \
--lr-scheduler 200000 \
--depth 3 \
--out-dir output/transformer/4_trans_layers_256_width/muscle_activation \
--dataname mint \
--vq-act relu \
--loss-vel 0.5 \
--recons-loss l1_smooth \
--exp-name 4_trans_layers_256_width \
--print-iter 50 \
--eval-iter 1000 \
--window-size 64 \
--width 256 \
--muscle_subset MUSINT_402 \
--resume-pth output/transformer/4_trans_layers_256_width/net_best_fid.pth \
--transformer_layer 4