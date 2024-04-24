import os
import json

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import models.vqvae as vqvae
import utils.losses as losses
import options.option_vq as option_vq
import utils.utils_model as utils_model
from dataset import dataset_MS, dataset_VQ, dataset_TM_eval
import utils.eval_trans as eval_trans
from options.get_eval_option import get_opt
from models.evaluator_wrapper import EvaluatorModelWrapper
import warnings
from tqdm import tqdm
from utils.profiler_utils import ProfilerContext

import time

warnings.filterwarnings("ignore")
from utils.word_vectorizer import WordVectorizer


def configure_dataset(args, logger):
    w_vectorizer = WordVectorizer("./glove", "our_vab")
    if args.dataname == "kit":
        dataset_opt_path = "checkpoints/kit/Comp_v6_KLD005/opt.txt"
        args.nb_joints = 21
    else:
        dataset_opt_path = "checkpoints/t2m/Comp_v6_KLD005/opt.txt"
        args.nb_joints = 22
    logger.info(f"Training on {args.dataname}, motions are with {args.nb_joints} joints")
    wrapper_opt = get_opt(dataset_opt_path, torch.device("cuda"))
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)
    return w_vectorizer, wrapper_opt, eval_wrapper


def initialize(args):
    torch.manual_seed(args.seed)
    args.out_dir = os.path.join(args.out_dir, f"{args.exp_name}")
    os.makedirs(args.out_dir, exist_ok=True)

    logger = utils_model.get_logger(args.out_dir)
    writer = SummaryWriter(args.out_dir)
    logger.info(json.dumps(vars(args), indent=2, sort_keys=True))
    return args, logger, writer


def initialize_model_and_optimizer(args, logger):
    net = vqvae.HumanVQVAE(
        args,
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
        logger.info(f"Loading checkpoint from {args.resume_pth}")
        ckpt = torch.load(args.resume_pth, map_location="cpu")
        net.load_state_dict(ckpt["net"], strict=True)
    net.train().cuda()

    optimizer = optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=args.gamma)

    return net, optimizer, scheduler


def warm_up_training(args, logger, writer, net, optimizer, train_loader_iter, global_metrics, Loss):

    iter_metrics = {
        "avg_recons": 0.0,
        "avg_perplexity": 0.0,
        "avg_commit": 0.0,
        "avg_data_time": 0.0,
        "avg_cuda_time": 0.0,
        "avg_forward_time": 0.0,
        "avg_backward_time": 0.0,
        "avg_lr": 0.0,
    }
    start_time = time.time()

    for nb_iter in range(1, args.warm_up_iter + 1):
        data_start_time = time.time()

        optimizer, current_lr = update_lr_warm_up(optimizer, nb_iter, args.warm_up_iter, args.lr)
        iter_metrics["avg_lr"] += current_lr

        out = next(train_loader_iter)

        iter_metrics["avg_data_time"] += time.time() - data_start_time

        forward_start_time = time.time()

        cuda_start_time = time.time()

        motion, gt_muscle_act = process_batch(out)

        iter_metrics["avg_cuda_time"] += time.time() - cuda_start_time

        if check_for_nans([motion, gt_muscle_act], logger, nb_iter):
            continue

        pred_muscle_act, loss_commit, perplexity = net(motion)

        iter_metrics["avg_forward_time"] += time.time() - forward_start_time

        loss = compute_loss(args, Loss, pred_muscle_act, gt_muscle_act, loss_commit)

        optimizer.zero_grad()
        backward_start_time = time.time()
        loss.backward()
        optimizer.step()

        iter_metrics["avg_backward_time"] += time.time() - backward_start_time

        # Accumulate metrics for logging
        iter_metrics["avg_recons"] += loss.item()
        iter_metrics["avg_perplexity"] += perplexity.item()
        iter_metrics["avg_commit"] += loss_commit.item()

        # Log periodically
        if nb_iter % args.print_iter == 0 or nb_iter == args.warm_up_iter:
            iter_metrics["iteration_time"] = time.time() - start_time

            for key in iter_metrics:
                iter_metrics[key] /= args.print_iter

            log_stats_with_timing(
                nb_iter=nb_iter,
                iter_metrics=iter_metrics,
                logger=logger,
                writer=writer,
                prefix="Warmup",
            )

            for key in iter_metrics:  # Reset metrics for the next iteration
                iter_metrics[key] = 0.0

            start_time = time.time()


def main_training_loop(
    args, logger, writer, net, optimizer, scheduler, train_loader_iter, val_loader, global_metrics, Loss
):
    iter_metrics = {
        "avg_recons": 0.0,
        "avg_perplexity": 0.0,
        "avg_commit": 0.0,
        "avg_data_time": 0.0,
        "avg_cuda_time": 0.0,
        "avg_forward_time": 0.0,
        "avg_backward_time": 0.0,
        "iteration_time": 0.0,
    }
    start_time = time.time()

    # Initial Validate
    validate_and_log(
        args=args,
        nb_iter=0,
        net=net,
        Loss=Loss,
        val_loader=val_loader,
        global_metrics=global_metrics,
        logger=logger,
        writer=writer,
    )

    profiler = ProfilerContext(name="train_loop", save_after=512)

    for nb_iter in range(1, args.total_iter + 1):
        with profiler:
            data_start_time = time.time()  # Start timing for data loading

            batch = next(train_loader_iter)

            iter_metrics["avg_data_time"] += time.time() - data_start_time  # Accumulate data loading time

            forward_start_time = time.time()  # Start timing for forward pass

            cuda_start_time = time.time()  # Start timing for cuda transfer

            motion, gt_muscle_act = process_batch(batch)

            iter_metrics["avg_cuda_time"] += time.time() - cuda_start_time

            if check_for_nans([motion, gt_muscle_act], logger, nb_iter):
                continue

            pred_muscle_act, loss_commit, perplexity = net(motion)

            loss = compute_loss(args, Loss, pred_muscle_act, gt_muscle_act, loss_commit)

            iter_metrics["avg_forward_time"] += time.time() - forward_start_time  # Accumulate forward pass time

            backward_start_time = time.time()  # Start timing for backward pass

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            iter_metrics["avg_backward_time"] += time.time() - backward_start_time

            # Accumulate metrics for logging
            iter_metrics["avg_recons"] += loss.item()
            iter_metrics["avg_perplexity"] += perplexity.item()
            iter_metrics["avg_commit"] += loss_commit.item()

            # Log periodically
            if nb_iter % args.print_iter == 0:
                iter_metrics["iteration_time"] = time.time() - start_time  # Calculate time for iterations

                for key in iter_metrics:
                    iter_metrics[key] /= args.print_iter

                log_stats_with_timing(
                    nb_iter=nb_iter,
                    iter_metrics=iter_metrics,
                    logger=logger,
                    writer=writer,
                    prefix="Train",
                )

                for key in iter_metrics:  # Reset metrics.
                    iter_metrics[key] = 0.0

                start_time = time.time()  # Reset start time for the next set of iterations

            # Validate and log periodically
            if nb_iter % args.eval_iter == 0:
                validate_and_log(
                    args=args,
                    nb_iter=nb_iter,
                    net=net,
                    Loss=Loss,
                    val_loader=val_loader,
                    global_metrics=global_metrics,
                    logger=logger,
                    writer=writer,
                )


def process_batch(batch):
    motion, gt_muscle_act = batch["motion"].cuda().float(), batch["muscle_activation"].cuda().float()
    return motion, gt_muscle_act


def check_for_nans(tensors, logger, nb_iter):
    if any(torch.isnan(tensor).any() for tensor in tensors):
        logger.info(f"NaN detected at iter {nb_iter}")
        return True
    return False


def compute_loss(args, Loss, pred_muscle_act, gt_muscle_act, loss_commit):
    loss_muscle_act = Loss.forward_naive(pred_muscle_act, gt_muscle_act)
    total_loss = loss_muscle_act + args.commit * loss_commit
    return total_loss


def update_lr_warm_up(optimizer, nb_iter, warm_up_iter, lr):

    current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

    return optimizer, current_lr


def validate_and_log(args, nb_iter, net, Loss, val_loader, global_metrics, logger, writer):
    logger.info("Performing evaluation at Iteration {}".format(nb_iter))

    best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger = (
        eval_trans.evaluation_vqvae(
            out_dir=args.out_dir,
            data_loader=val_loader,
            net=net,
            Loss=Loss,
            logger=logger,
            writer=writer,
            nb_iter=nb_iter,
            best_fid=global_metrics["best_fid"],
            best_iter=global_metrics["best_iter"],
            best_div=global_metrics["best_div"],
            best_top1=global_metrics["best_top1"],
            best_top2=global_metrics["best_top2"],
            best_top3=global_metrics["best_top3"],
            best_matching=global_metrics["best_matching"],
            args=args,
        )
    )

    global_metrics["best_fid"] = best_fid
    global_metrics["best_iter"] = best_iter
    global_metrics["best_div"] = best_div
    global_metrics["best_top1"] = best_top1
    global_metrics["best_top2"] = best_top2
    global_metrics["best_top3"] = best_top3
    global_metrics["best_matching"] = best_matching


def log_stats_with_timing(nb_iter, iter_metrics, logger, writer, prefix="Train"):

    logger.info(
        f"{prefix}. Iter {nb_iter} | "
        f"Commit {iter_metrics['avg_commit']:.3f} | "
        f"PPL    {iter_metrics['avg_perplexity']:.3f} | "
        f"Recons {iter_metrics['avg_recons']:.3f} | "
        + (f"LR {iter_metrics['avg_lr']:.3} s | " if "avg_lr" in iter_metrics else "")
        + f"DT {iter_metrics['avg_data_time']:.3} s | "
        f"CT {iter_metrics['avg_cuda_time']:.3} s | "
        f"FT {iter_metrics['avg_forward_time']:.3f} s | "
        f"BT {iter_metrics['avg_backward_time']:.3f} s | "
        f"IT {iter_metrics['iteration_time']:.3f} s"
    )
    for key, value in iter_metrics.items():
        if "time" in key:
            continue
        writer.add_scalar(f"Iter/{prefix}/{key}", value, nb_iter)


def log_iteration_stats_with_timing(
    nb_iter,
    avg_recons,
    avg_perplexity,
    avg_commit,
    avg_data_time,
    avg_forward_time,
    avg_backward_time,
    iteration_time,
    print_iter,
    logger,
    writer,
):
    avg_recons /= print_iter
    avg_perplexity /= print_iter
    avg_commit /= print_iter
    avg_data_time /= print_iter
    avg_forward_time /= print_iter
    avg_backward_time /= print_iter
    iteration_time /= print_iter
    logger.info(
        f"Train. Iter {nb_iter} : \t Commit. {avg_commit:.5f} \t PPL. {avg_perplexity:.2f} \t Recons.  {avg_recons:.5f} \t Data Time {avg_data_time:.5f} s \t Forward Time {avg_forward_time:.5f} s \t Backward Time {avg_backward_time:.5f} s \t Iteration Time {iteration_time:.5f} s"
    )
    writer.add_scalar("Timing/Data_Loading", avg_data_time, nb_iter)
    writer.add_scalar("Timing/Forward_Pass", avg_forward_time, nb_iter)
    writer.add_scalar("Timing/Backward_Pass", avg_backward_time, nb_iter)
    writer.add_scalar("Timing/Total_Iteration", iteration_time, nb_iter)


def prepare_dataloaders(args, w_vectorizer):
    train_loader = dataset_MS.DATALoader(
        args.dataname,
        args.batch_size,
        mode="train",
        window_size=args.window_size,
        unit_length=2**args.down_t,
        w_vectorizer=w_vectorizer,
        num_workers=args.num_workers,
        use_profiling=args.profiling,
    )

    val_loader = dataset_MS.DATALoader(
        args.dataname,
        args.batch_size,  # You might want to make this a command-line argument
        mode="val",
        window_size=args.window_size,
        unit_length=2**args.down_t,
        label_required=True,  # Assuming validation requires labels
        w_vectorizer=w_vectorizer,
        num_workers=args.num_workers,
        use_profiling=args.profiling,
    )

    train_loader_iter = dataset_MS.cycle(train_loader)

    return train_loader, val_loader, train_loader_iter


def main():
    # Parse arguments
    args = option_vq.get_args_parser()

    ProfilerContext.is_profiling_active = args.profiling

    # Initialize environment, logging, and writer
    args, logger, writer = initialize(args)

    # Configure dataset and related components
    w_vectorizer, wrapper_opt, eval_wrapper = configure_dataset(args, logger)

    # Initialize the model, optimizer, and scheduler
    net, optimizer, scheduler = initialize_model_and_optimizer(args, logger)

    # Prepare the data loaders
    train_loader, val_loader, train_loader_iter = prepare_dataloaders(args, w_vectorizer)

    # Define the loss function
    Loss = losses.ReConsLoss(args.recons_loss, args.nb_joints)

    global_metrics = {  # TODO: Load on training restart.
        "best_fid": 1000,
        "best_iter": 0,
        "best_div": 100,
        "best_top1": 0,
        "best_top2": 0,
        "best_top3": 0,
        "best_matching": 1000,
        "start_time": time.time(),
    }

    # Execute the warm-up training
    with ProfilerContext(name="warmup"):
        warm_up_training(args, logger, writer, net, optimizer, train_loader_iter, global_metrics, Loss)

    # Enter the main training loop
    main_training_loop(
        args,
        logger,
        writer,
        net,
        optimizer,
        scheduler,
        train_loader_iter,
        val_loader,
        global_metrics,
        Loss,
    )


if __name__ == "__main__":
    main()