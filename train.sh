#!/bin/bash

export MINT_DATA="/lsdf/data/activity/MuscleSim/musclesim_dataset"
export MOTION_DATA="/lsdf/users/dschneider-kf3609/workspace/HumanML3D/HumanML3D"
export BABEL_DATA="/lsdf/data/activity/BABEL/babel_v1-0_release"

export CUDA_VISIBLE_DEVICES=1

python3 train_vq_ms2.py \
--batch-size 128 \
--lr 2e-4 \
--total-iter 300000 \
--lr-scheduler 200000 \
--depth 3 \
--out-dir output/exp_transformer9 \
--dataname mint \
--vq-act relu \
--quantizer ema_reset \
--loss-vel 0.5 \
--recons-loss l1_smooth \
--exp-name transformer \
--print-iter 50 \
--eval-iter 1000 \
--window-size 64 \
--width 64 \
--muscle_subset MUSINT_402