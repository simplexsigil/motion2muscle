import os
import json

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import models.m2m_transformer as motion_to_muscle
import options.option_vq as option_vq
from dataset import dataset_MS, dataset_MIA
import utils.eval_trans as eval_trans
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")


import options.option_vq as option_vq


def foward_data(
    out_dir,
    val_loader,
    net,
    savenpy=False,
):
    net.eval()

    # Initialize a DataFrame to store metadata
    metadata = []

    for batch in tqdm(val_loader):
        (motion, name, muscle_act_gt, time_start) = (
            batch["motion"],
            batch["name"],
            batch["muscle_activation"],
            batch["time_start"],
        )

        motion = motion.cuda()

        bs, seq = motion.shape[0], motion.shape[1]

        for i in range(bs):
            muscle_act_pred = net(motion[i : i + 1])
            cur_muscle_act_gt = muscle_act_gt[i : i + 1]

            out_file = os.path.join(out_dir, name[i])
            os.makedirs(os.path.dirname(out_file), exist_ok=True)

            np.save(
                out_file + "_" + str(i) + "_gt.npy",
                cur_muscle_act_gt[:].cpu().numpy(),
            )

            np.save(
                out_file + "_" + str(i) + "_pred.npy",
                muscle_act_pred[:].detach().cpu().numpy(),
            )

            # Append metadata for the current batch item
            metadata.append(
                {
                    "name": name[i],
                    "gt_name": name[i] + "_" + str(i) + "_gt.npy",
                    "pred_name": name[i] + "_" + str(i) + "_pred.npy",
                    "time_start": time_start[i],
                }
            )

    metadata_df = pd.DataFrame(metadata)

    # Save the DataFrame
    if savenpy:
        metadata_df.to_csv(os.path.join(out_dir, "metadata.csv"), index=False)


##### ---- Exp dirs ---- #####
args = option_vq.get_args_parser(parse=True)
torch.manual_seed(args.seed)

args.out_dir = os.path.join(args.out_dir, f"{args.exp_name}")
os.makedirs(args.out_dir, exist_ok=True)


##### ---- Logger ---- #####
import logging

logger = logging.getLogger()
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))


# w_vectorizer = WordVectorizer("./glove", "our_vab")

args.nb_joints = 22

logger.info(f"Training on {args.dataname}, motions are with {args.nb_joints} joints")


##### ---- Dataloader ---- #####
test_loader = dataset_MIA.MIADATALoader(
    args.dataname,
    3000,
    mode="val",
    window_size=args.window_size,
)

##### ---- Network ---- #####
input_width = 263
output_width = args.vq_dec_emb_width
net = motion_to_muscle.MotionToMuscleModel(
    input_width,
    output_width,
    args.width,
    num_layers=args.transformer_layer,
)

if args.resume_pth:
    print("Loading model from file {}".format(args.resume_pth))
    logger.info("loading checkpoint from {}".format(args.resume_pth))
    ckpt = torch.load(args.resume_pth, map_location="cpu")
    net.load_state_dict(ckpt["net"], strict=True)
net.train()
net.cuda()

writer = SummaryWriter(args.out_dir)

##### ---- eval ---- #####
with torch.no_grad():
    foward_data(
        args.out_dir,
        test_loader,
        net,
        savenpy=True,
    )
