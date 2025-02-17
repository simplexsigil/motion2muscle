import os
import json

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import models.vqvae as vqvae
import utils.losses as losses
import options.option_vq as option_vq
import utils.utils_model as utils_model
from dataset import dataset_MS
import utils.eval_trans as eval_trans
from options.get_eval_option import get_opt
import warnings
from tqdm import tqdm
import numpy as np
import pandas as pd

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
        (motion, name, muscle_act_gt, time_start) = (
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
            muscle_act_pred = net(motion[i : i + 1])
            cur_muscle_act_gt = muscle_act_gt[i : i + 1]

            np.save(
                os.path.join(out_dir, name[i].replace("/", "__I__") + "_" + str(i) + "_gt.npy"),
                cur_muscle_act_gt[:].cpu().numpy(),
            )
            np.save(
                os.path.join(out_dir, name[i].replace("/", "__I__") + "_" + str(i) + "_pred.npy"),
                muscle_act_pred[0][:].detach().cpu().numpy(),
            )

            # print(f"Saved to {os.path.join(out_dir, name[i].replace("/", "__I__") + "_" + i + "_pred.npy")}")

            # Append metadata for the current batch item
            metadata.append({"name": name[i], "gt_name": name[i].replace("/", "__I__") + "_" + str(i) + "_gt.npy", "pred_name": name[i].replace("/", "__I__") + "_" + str(i) + "_pred.npy", "time_start": time_start[i]})

            pred_muscle_eval[i : i + 1] = muscle_act_pred[0]

        muscle_pred_list.append(pred_muscle_eval)
        muscle_ann_list.append(muscle_act_gt)

        temp_R, temp_match = eval_trans.calculate_R_precision_temporal(
            muscle_act_gt.cpu().numpy(), pred_muscle_eval.cpu().numpy(), top_k=3, sum_all=True
        )
        R_precision += temp_R
        matching_score += temp_match

        nb_sample += bs

        break

    muscle_ann_np = torch.cat(muscle_ann_list, dim=0).cpu().numpy()
    muscle_pred_np = torch.cat(muscle_pred_list, dim=0).cpu().numpy()

    metadata_df = pd.DataFrame(metadata)

    # Save the DataFrame
    if savenpy:
        metadata_df.to_csv(os.path.join(out_dir, "metadata.csv"), index=False)

    muscle_pred_np = np.nan_to_num(muscle_pred_np)

    muscle_ann_np = muscle_ann_np.reshape(-1, muscle_ann_np.shape[-1])
    muscle_pred_np = muscle_pred_np.reshape(-1, muscle_pred_np.shape[-1])

    gt_mu, gt_cov = eval_trans.calculate_activation_statistics(muscle_ann_np)
    mu, cov = eval_trans.calculate_activation_statistics(muscle_pred_np)

    diversity_real = eval_trans.calculate_diversity(muscle_ann_np, 300 if nb_sample > 300 else 100)
    diversity = eval_trans.calculate_diversity(muscle_pred_np, 300 if nb_sample > 300 else 100)

    R_precision = R_precision / nb_sample
    matching_score = matching_score / nb_sample

    fid = eval_trans.calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Test.:, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision. {R_precision}, matching_score_pred. {matching_score}"
    logger.info(msg)

    return fid, diversity_real, diversity, R_precision[0], R_precision[1], R_precision[2], matching_score


##### ---- Exp dirs ---- #####
args = option_vq.get_args_parser(parse=True)
torch.manual_seed(args.seed)

args.out_dir = os.path.join(args.out_dir, f"{args.exp_name}")
os.makedirs(args.out_dir, exist_ok=True)


##### ---- Logger ---- #####
import logging

logger = logging.getLogger()
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))


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
    10000,
    mode="val",
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
repeat_time = 1

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
print("final result:")
print("fid: ", sum(fid) / repeat_time)
print("div: ", sum(div) / repeat_time)
print("div real: ", sum(div_real) / repeat_time)
print("top1: ", sum(top1) / repeat_time)
print("top2: ", sum(top2) / repeat_time)
print("top3: ", sum(top3) / repeat_time)
print("matching: ", sum(matching) / repeat_time)

fid = np.array(fid)
div = np.array(div)
top1 = np.array(top1)
top2 = np.array(top2)
top3 = np.array(top3)
matching = np.array(matching)
msg_final = f"FID. {np.mean(fid):.3f}, conf. {np.std(fid)*1.96/np.sqrt(repeat_time):.3f}, Diversity Real. {np.mean(div_real):.3f}, conf. {np.std(div_real)*1.96/np.sqrt(repeat_time):.3f}, Diversity. {np.mean(div):.3f}, conf. {np.std(div)*1.96/np.sqrt(repeat_time):.3f}, TOP1. {np.mean(top1):.3f}, conf. {np.std(top1)*1.96/np.sqrt(repeat_time):.3f}, TOP2. {np.mean(top2):.3f}, conf. {np.std(top2)*1.96/np.sqrt(repeat_time):.3f}, TOP3. {np.mean(top3):.3f}, conf. {np.std(top3)*1.96/np.sqrt(repeat_time):.3f}, Matching. {np.mean(matching):.3f}, conf. {np.std(matching)*1.96/np.sqrt(repeat_time):.3f}"
logger.info(msg_final)