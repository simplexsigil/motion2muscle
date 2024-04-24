import os
import json

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import models.vqvae as vqvae
import options.option_vq as option_vq
from dataset import dataset_MS
# import utils.eval_trans as eval_trans
import warnings

import numpy as np


warnings.filterwarnings("ignore")


import options.option_vq as option_vq


def test_vqvae(
    out_dir,
    val_loader,
    net,
    logger,
    savenpy=False,
):
    net.eval()
    nb_sample = 0

    muscle_ann_list = []
    muscle_pred_list = []

    R_precision = 0

    nb_sample = 0
    matching_score = 0

    # if savenpy:
    #     if not os.path.exists(os.path.join(out_dir, "muscle_acts3")):
    #         os.makedirs(os.path.join(out_dir, "muscle_acts3"))
    #     out_dir_ma = os.path.join(out_dir, "muscle_acts3")

    # Initialize a DataFrame to store metadata
    metadata = []

    for batch in val_loader:
        (action, motion, name, muscle_act_gt, time_start) = (
            batch["actions"],
            batch["motion"],
            batch["name"],
            batch["muscle_activation"],
            batch["time_start"],
        )

        motion = motion.cuda()
        m_length = torch.tensor([motion.shape[1]] * motion.shape[0])

        bs, seq = motion.shape[0], motion.shape[1]

        pred_muscle_eval = torch.zeros((bs, seq, muscle_act_gt.shape[-1])).cuda()

        for i in range(bs):
            muscle_act_pred, loss_commit, perplexity = net(motion[i : i + 1])
            cur_muscle_act_gt = muscle_act_gt[i : i + 1]

            # code = net.encode(motion[i : i + 1])
            # print(code)
            # exit()

            if savenpy:
                np.save(
                    os.path.join(out_dir, name[i].replace("/", "__I__") + "_gt.npy"),
                    cur_muscle_act_gt[:].cpu().numpy(),
                )
                np.save(
                    os.path.join(out_dir, name[i].replace("/", "__I__") + "_pred.npy"),
                    muscle_act_pred[:].detach().cpu().numpy(),
                )

                # Append metadata for the current batch item
                metadata.append({"name": name[i], "time_start": time_start[i], "action": action[i]})

            pred_muscle_eval[i : i + 1] = muscle_act_pred

        muscle_pred_list.append(pred_muscle_eval)
        muscle_ann_list.append(muscle_act_gt)




##### ---- Exp dirs ---- #####
args = option_vq.get_args_parser(parse=True)
torch.manual_seed(args.seed)

args.out_dir = os.path.join(args.out_dir, f"{args.exp_name}")
os.makedirs(args.out_dir, exist_ok=True)


##### ---- Logger ---- #####
import logging

logger = logging.getLogger()
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))


dataset_opt_path = "checkpoints/t2m/Comp_v6_KLD005/opt.txt"
args.nb_joints = 22

logger.info(f"Training on {args.dataname}, motions are with {args.nb_joints} joints")

"""
all_loader = dataset_MS.DATALoader(
    args.dataname,
    1,
    mode="all",
    window_size=args.window_size,
    unit_length=2**args.down_t,
    w_vectorizer=w_vectorizer,
)"""

##### ---- Dataloader ---- #####
test_loader = dataset_MS.DATALoader(
    args.dataname,
    128,
    mode="train",
    window_size=args.window_size,
    unit_length=2**args.down_t,
)

##### ---- Network ---- #####
net = vqvae.HumanVQVAE(
    args,  ## use args to define different parameters in different quantizers
    args.nb_code,
    args.code_dim,
    args.output_emb_width,
    args.down_t,
    args.stride_t,
    args.width,
    args.depth,
    args.dilation_growth_rate,
    args.vq_act,
    args.vq_norm,
)

if args.resume_pth:
    print("Loading model from file {}".format(args.resume_pth))
    logger.info("loading checkpoint from {}".format(args.resume_pth))
    ckpt = torch.load(args.resume_pth, map_location="cpu")
    net.load_state_dict(ckpt["net"], strict=True)
net.train()
net.cuda()

writer = SummaryWriter(args.out_dir)

##### ---- Optimizer & Scheduler ---- #####
optimizer = optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=args.gamma)

##### ---- eval ---- #####
fid = []
div = []
div_real = []
top1 = []
top2 = []
top3 = []
matching = []
repeat_time = 20

with torch.no_grad():
    for i in range(repeat_time):
        best_fid, best_div_real, best_div, best_top1, best_top2, best_top3, best_matching = test_vqvae(
            args.out_dir,
            test_loader,
            net,
            logger,
            savenpy=(i == 0),
        )
        fid.append(best_fid)
        div.append(best_div)
        div_real.append(best_div)
        top1.append(best_top1)
        top2.append(best_top2)
        top3.append(best_top3)
        matching.append(best_matching)