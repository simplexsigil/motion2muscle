import os

import clip
import numpy as np
import torch
from scipy import linalg

import visualization.plot_3d_global as plot_3d
from utils.motion_process import recover_from_ric


def tensorborad_add_video_xyz(writer, xyz, nb_iter, tag, nb_vis=4, title_batch=None, outname=None):
    xyz = xyz[:1]
    bs, seq = xyz.shape[:2]
    xyz = xyz.reshape(bs, seq, -1, 3)
    plot_xyz = plot_3d.draw_to_batch(xyz.cpu().numpy(), title_batch, outname)
    plot_xyz = np.transpose(plot_xyz, (0, 1, 4, 2, 3))
    writer.add_video(tag, plot_xyz, nb_iter, fps=20)


@torch.no_grad()
def evaluation_vqvae(
    out_dir,
    data_loader,
    net,
    Loss,
    logger,
    writer,
    nb_iter,
    best_fid,
    best_iter,
    best_div,
    best_top1,
    best_top2,
    best_top3,
    best_matching,
    save=True,
    savenpy=False,
    prefix="Val",
    args=None,
):
    net.eval()
    nb_sample = 0

    muscle_ann_list = []
    muscle_pred_list = []

    R_precision_real = 0
    R_precision = 0

    nb_sample = 0
    matching_score_real = 0
    matching_score = 0

    perplexity_cum = 0
    loss_commit_cum = 0
    loss_pred_cum = 0
    total_loss = 0

    num_batches = len(data_loader)

    for batch in data_loader:
        (word_embeddings, pos_one_hots, caption, sent_len, motion, token, name, muscle_act_gt) = (
            batch["word_embeddings"],
            batch["pos_one_hots"],
            batch["caption"],
            batch["sent_len"],
            batch["motion"],
            batch["tokens"],
            batch["unique_name"],
            batch["muscle_activation"],
        )

        motion = motion.cuda()
        m_length = torch.tensor([motion.shape[1]] * motion.shape[0])
        word_embeddings = torch.tensor(np.nan_to_num(word_embeddings)).cuda()
        pos_one_hots = torch.tensor(pos_one_hots).cuda()
        sent_len = torch.tensor(sent_len).cuda()

        bs, seq = motion.shape[0], motion.shape[1]

        muscle_act_pred = torch.zeros((bs, seq, muscle_act_gt.shape[-1])).cuda()
        lc_b = 0
        perp_b = 0

        muscle_act_pred, lc_b, perp_b = net(motion)

        if savenpy:
            for i in range(bs):
                np.save(
                    os.path.join(out_dir, name[i].replace("/", "__I__") + "_gt.npy"), muscle_act_gt[i, :].cpu().numpy()
                )
                np.save(
                    os.path.join(out_dir, name[i].replace("/", "__I__") + "_pred.npy"),
                    muscle_act_pred[i, :].detach().cpu().numpy(),
                )

        muscle_act_pred = muscle_act_pred.cpu()

        loss_pred = Loss.forward_naive(muscle_act_pred, muscle_act_gt)

        perplexity_cum += perp_b
        loss_commit_cum += lc_b
        loss_pred_cum += loss_pred

        total_loss += loss_pred + args.commit * lc_b

        muscle_pred_list.append(muscle_act_pred)
        muscle_ann_list.append(muscle_act_gt)

        temp_R, temp_match = calculate_R_precision_temporal(
            muscle_act_gt.cpu().numpy(), muscle_act_pred.cpu().numpy(), top_k=3, sum_all=True
        )
        R_precision += temp_R
        matching_score += temp_match

        nb_sample += bs

    muscle_ann_np = torch.cat(muscle_ann_list, dim=0).cpu().numpy()
    muscle_pred_np = torch.cat(muscle_pred_list, dim=0).cpu().numpy()

    muscle_ann_np = muscle_ann_np.reshape(-1, muscle_ann_np.shape[-1])
    muscle_pred_np = muscle_pred_np.reshape(-1, muscle_pred_np.shape[-1])

    gt_mu, gt_cov = calculate_activation_statistics(muscle_ann_np)
    mu, cov = calculate_activation_statistics(muscle_pred_np)

    diversity_real = calculate_diversity(muscle_ann_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(muscle_pred_np, 300 if nb_sample > 300 else 100)

    R_precision = R_precision / nb_sample
    matching_score = matching_score / nb_sample

    total_loss = total_loss / num_batches
    loss_pred_cum = loss_pred_cum / num_batches
    loss_commit_cum = loss_commit_cum / num_batches
    perplexity_cum = perplexity_cum / num_batches

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = (
        f"--> \t Eva. Iter {nb_iter} : \n"
        f"FID. {fid:.2f} \n"
        f"Diversity Real. {diversity_real:.2f} \n"
        f"Diversity. {diversity:.2f} \n"
        f"R_precision. {R_precision} \n"
        f"matching_score_pred. {matching_score:2f} \n"
        f"Loss Pred. {loss_pred_cum:.5f} \n"
        f"Loss Commit. {loss_commit_cum:.5f} \n"
        f"Loss Total. {total_loss:.5f} \n"
        f"Perplexity. {perplexity_cum:.2f}"
    )

    logger.info(msg)

    writer.add_scalar(f"{prefix}/FID", fid, nb_iter)
    writer.add_scalar(f"{prefix}/Diversity", diversity, nb_iter)
    writer.add_scalar(f"{prefix}/top1", R_precision[0], nb_iter)
    writer.add_scalar(f"{prefix}/top2", R_precision[1], nb_iter)
    writer.add_scalar(f"{prefix}/top3", R_precision[2], nb_iter)
    writer.add_scalar(f"{prefix}/matching_score", matching_score, nb_iter)
    writer.flush()

    if fid < best_fid:
        msg = f"--> --> \t FID Improved from {best_fid:.2f} to {fid:.2f} !!!"
        logger.info(msg)
        best_fid, best_iter = fid, nb_iter
        if save:
            logger.info(f"Saving model to {os.path.join(out_dir, 'net_best_fid.pth')}")
            torch.save({"net": net.state_dict()}, os.path.join(out_dir, "net_best_fid.pth"))

    if abs(diversity_real - diversity) < abs(diversity_real - best_div):
        msg = f"--> --> \t Diversity Improved from {best_div:.2f} to {diversity:.2f} !!!"
        logger.info(msg)
        best_div = diversity
        if save:
            logger.info(f"Saving model to {os.path.join(out_dir, 'net_best_div.pth')}")
            torch.save({"net": net.state_dict()}, os.path.join(out_dir, "net_best_div.pth"))

    if R_precision[0] > best_top1:
        msg = f"--> --> \t Top1 Improved from {best_top1:.2f} to {R_precision[0]:.2f} !!!"
        logger.info(msg)
        best_top1 = R_precision[0]
        if save:
            logger.info(f"Saving model to {os.path.join(out_dir, 'net_best_top1.pth')}")
            torch.save({"net": net.state_dict()}, os.path.join(out_dir, "net_best_top1.pth"))

    if R_precision[1] > best_top2:
        msg = f"--> --> \t Top2 Improved from {best_top2:.2f} to {R_precision[1]:.2f} !!!"
        logger.info(msg)
        best_top2 = R_precision[1]

    if R_precision[2] > best_top3:
        msg = f"--> --> \t Top3 Improved from {best_top3:.2f} to {R_precision[2]:.2f} !!!"
        logger.info(msg)
        best_top3 = R_precision[2]

    if matching_score < best_matching:
        msg = f"--> --> \t matching_score Improved from {best_matching:.2f} to {matching_score:.2f} !!!"
        logger.info(msg)
        best_matching = matching_score
        if save:
            logger.info(f"Saving model to {os.path.join(out_dir, 'net_best_matching.pth')}")
            torch.save({"net": net.state_dict()}, os.path.join(out_dir, "net_best_matching.pth"))

    if save:
        logger.info(f"Saving model to {os.path.join(out_dir, 'net_last.pth')}")
        torch.save({"net": net.state_dict()}, os.path.join(out_dir, "net_last.pth"))

    net.train()
    return best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger


@torch.no_grad()
def evaluation_transformer(
    out_dir,
    val_loader,
    net,
    trans,
    logger,
    writer,
    nb_iter,
    best_fid,
    best_iter,
    best_div,
    best_top1,
    best_top2,
    best_top3,
    best_matching,
    clip_model,
    eval_wrapper,
    draw=True,
    save=True,
    savegif=False,
):

    trans.eval()
    nb_sample = 0

    draw_org = []
    draw_pred = []
    draw_text = []
    draw_text_pred = []

    motion_annotation_list = []
    motion_pred_list = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0

    nb_sample = 0
    for i in range(1):
        for batch in val_loader:
            word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token, name = batch

            bs, seq = pose.shape[:2]
            num_joints = 21 if pose.shape[-1] == 251 else 22

            text = clip.tokenize(clip_text, truncate=True).cuda()

            feat_clip_text = clip_model.encode_text(text).float()
            pred_pose_eval = torch.zeros((bs, seq, pose.shape[-1])).cuda()
            pred_len = torch.ones(bs).long()

            for k in range(bs):
                try:
                    index_motion = trans.sample(feat_clip_text[k : k + 1], False)
                except:
                    index_motion = torch.ones(1, 1).cuda().long()

                pred_pose = net.forward_decoder(index_motion)
                cur_len = pred_pose.shape[1]

                pred_len[k] = min(cur_len, seq)
                pred_pose_eval[k : k + 1, :cur_len] = pred_pose[:, :seq]

                if draw:
                    pred_denorm = val_loader.dataset.inv_transform(pred_pose.detach().cpu().numpy())
                    pred_xyz = recover_from_ric(torch.from_numpy(pred_denorm).float().cuda(), num_joints)

                    if i == 0 and k < 4:
                        draw_pred.append(pred_xyz)
                        draw_text_pred.append(clip_text[k])

            et_pred, em_pred = eval_wrapper.get_co_embeddings(
                word_embeddings, pos_one_hots, sent_len, pred_pose_eval, pred_len
            )

            if i == 0:
                pose = pose.cuda().float()

                et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
                motion_annotation_list.append(em)
                motion_pred_list.append(em_pred)

                if draw:
                    pose = val_loader.dataset.inv_transform(pose.detach().cpu().numpy())
                    pose_xyz = recover_from_ric(torch.from_numpy(pose).float().cuda(), num_joints)

                    for j in range(min(4, bs)):
                        draw_org.append(pose_xyz[j][: m_length[j]].unsqueeze(0))
                        draw_text.append(clip_text[j])

                temp_R, temp_match = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
                R_precision_real += temp_R
                matching_score_real += temp_match
                temp_R, temp_match = calculate_R_precision(
                    et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True
                )
                R_precision += temp_R
                matching_score_pred += temp_match

                nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Iter {nb_iter} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}"
    logger.info(msg)

    if draw:
        writer.add_scalar("./Test/FID", fid, nb_iter)
        writer.add_scalar("./Test/Diversity", diversity, nb_iter)
        writer.add_scalar("./Test/top1", R_precision[0], nb_iter)
        writer.add_scalar("./Test/top2", R_precision[1], nb_iter)
        writer.add_scalar("./Test/top3", R_precision[2], nb_iter)
        writer.add_scalar("./Test/matching_score", matching_score_pred, nb_iter)

        if nb_iter % 10000 == 0:
            for ii in range(4):
                tensorborad_add_video_xyz(
                    writer,
                    draw_org[ii],
                    nb_iter,
                    tag="./Vis/org_eval" + str(ii),
                    nb_vis=1,
                    title_batch=[draw_text[ii]],
                    outname=[os.path.join(out_dir, "gt" + str(ii) + ".gif")] if savegif else None,
                )

        if nb_iter % 10000 == 0:
            for ii in range(4):
                tensorborad_add_video_xyz(
                    writer,
                    draw_pred[ii],
                    nb_iter,
                    tag="./Vis/pred_eval" + str(ii),
                    nb_vis=1,
                    title_batch=[draw_text_pred[ii]],
                    outname=[os.path.join(out_dir, "pred" + str(ii) + ".gif")] if savegif else None,
                )

    if fid < best_fid:
        msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
        logger.info(msg)
        best_fid, best_iter = fid, nb_iter
        if save:
            torch.save({"trans": trans.state_dict()}, os.path.join(out_dir, "net_best_fid.pth"))

    if matching_score_pred < best_matching:
        msg = f"--> --> \t matching_score Improved from {best_matching:.5f} to {matching_score_pred:.5f} !!!"
        logger.info(msg)
        best_matching = matching_score_pred

    if abs(diversity_real - diversity) < abs(diversity_real - best_div):
        msg = f"--> --> \t Diversity Improved from {best_div:.5f} to {diversity:.5f} !!!"
        logger.info(msg)
        best_div = diversity

    if R_precision[0] > best_top1:
        msg = f"--> --> \t Top1 Improved from {best_top1:.4f} to {R_precision[0]:.4f} !!!"
        logger.info(msg)
        best_top1 = R_precision[0]

    if R_precision[1] > best_top2:
        msg = f"--> --> \t Top2 Improved from {best_top2:.4f} to {R_precision[1]:.4f} !!!"
        logger.info(msg)
        best_top2 = R_precision[1]

    if R_precision[2] > best_top3:
        msg = f"--> --> \t Top3 Improved from {best_top3:.4f} to {R_precision[2]:.4f} !!!"
        logger.info(msg)
        best_top3 = R_precision[2]

    if save:
        torch.save({"trans": trans.state_dict()}, os.path.join(out_dir, "net_last.pth"))

    trans.train()
    return best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger


@torch.no_grad()
def evaluation_transformer_test(
    out_dir,
    val_loader,
    net,
    trans,
    logger,
    writer,
    nb_iter,
    best_fid,
    best_iter,
    best_div,
    best_top1,
    best_top2,
    best_top3,
    best_matching,
    best_multi,
    clip_model,
    eval_wrapper,
    draw=True,
    save=True,
    savegif=False,
    savenpy=False,
):

    trans.eval()
    nb_sample = 0

    draw_org = []
    draw_pred = []
    draw_text = []
    draw_text_pred = []
    draw_name = []

    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0

    nb_sample = 0

    for batch in val_loader:

        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token, name = batch
        bs, seq = pose.shape[:2]
        num_joints = 21 if pose.shape[-1] == 251 else 22

        text = clip.tokenize(clip_text, truncate=True).cuda()

        feat_clip_text = clip_model.encode_text(text).float()
        motion_multimodality_batch = []
        for i in range(30):
            pred_pose_eval = torch.zeros((bs, seq, pose.shape[-1])).cuda()
            pred_len = torch.ones(bs).long()

            for k in range(bs):
                try:
                    index_motion = trans.sample(feat_clip_text[k : k + 1], True)
                except:
                    index_motion = torch.ones(1, 1).cuda().long()

                pred_pose = net.forward_decoder(index_motion)
                cur_len = pred_pose.shape[1]

                pred_len[k] = min(cur_len, seq)
                pred_pose_eval[k : k + 1, :cur_len] = pred_pose[:, :seq]

                if i == 0 and (draw or savenpy):
                    pred_denorm = val_loader.dataset.inv_transform(pred_pose.detach().cpu().numpy())
                    pred_xyz = recover_from_ric(torch.from_numpy(pred_denorm).float().cuda(), num_joints)

                    if savenpy:
                        np.save(os.path.join(out_dir, name[k] + "_pred.npy"), pred_xyz.detach().cpu().numpy())

                    if draw:
                        if i == 0:
                            draw_pred.append(pred_xyz)
                            draw_text_pred.append(clip_text[k])
                            draw_name.append(name[k])

            et_pred, em_pred = eval_wrapper.get_co_embeddings(
                word_embeddings, pos_one_hots, sent_len, pred_pose_eval, pred_len
            )

            motion_multimodality_batch.append(em_pred.reshape(bs, 1, -1))

            if i == 0:
                pose = pose.cuda().float()

                et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
                motion_annotation_list.append(em)
                motion_pred_list.append(em_pred)

                if draw or savenpy:
                    pose = val_loader.dataset.inv_transform(pose.detach().cpu().numpy())
                    pose_xyz = recover_from_ric(torch.from_numpy(pose).float().cuda(), num_joints)

                    if savenpy:
                        for j in range(bs):
                            np.save(
                                os.path.join(out_dir, name[j] + "_gt.npy"),
                                pose_xyz[j][: m_length[j]].unsqueeze(0).cpu().numpy(),
                            )

                    if draw:
                        for j in range(bs):
                            draw_org.append(pose_xyz[j][: m_length[j]].unsqueeze(0))
                            draw_text.append(clip_text[j])

                temp_R, temp_match = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
                R_precision_real += temp_R
                matching_score_real += temp_match
                temp_R, temp_match = calculate_R_precision(
                    et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True
                )
                R_precision += temp_R
                matching_score_pred += temp_match

                nb_sample += bs

        motion_multimodality.append(torch.cat(motion_multimodality_batch, dim=1))

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    multimodality = 0
    motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu().numpy()
    multimodality = calculate_multimodality(motion_multimodality, 10)

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Iter {nb_iter} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}, multimodality. {multimodality:.4f}"
    logger.info(msg)

    if draw:
        for ii in range(len(draw_org)):
            tensorborad_add_video_xyz(
                writer,
                draw_org[ii],
                nb_iter,
                tag="./Vis/" + draw_name[ii] + "_org",
                nb_vis=1,
                title_batch=[draw_text[ii]],
                outname=[os.path.join(out_dir, draw_name[ii] + "_skel_gt.gif")] if savegif else None,
            )

            tensorborad_add_video_xyz(
                writer,
                draw_pred[ii],
                nb_iter,
                tag="./Vis/" + draw_name[ii] + "_pred",
                nb_vis=1,
                title_batch=[draw_text_pred[ii]],
                outname=[os.path.join(out_dir, draw_name[ii] + "_skel_pred.gif")] if savegif else None,
            )

    trans.train()
    return (
        fid,
        best_iter,
        diversity,
        R_precision[0],
        R_precision[1],
        R_precision[2],
        matching_score_pred,
        multimodality,
        writer,
        logger,
    )


# (X - X_train)*(X - X_train) = -2X*X_train + X*X + X_train*X_train
def euclidean_distance_matrix(matrix1, matrix2):
    """
    Params:
    -- matrix1: N1 x D
    -- matrix2: N2 x D
    Returns:
    -- dist: N1 x N2
    dist[i, j] == distance(matrix1[i], matrix2[j])
    """
    assert matrix1.shape[1] == matrix2.shape[1]
    d1 = -2 * np.dot(matrix1, matrix2.T)  # shape (num_test, num_train)
    d2 = np.sum(np.square(matrix1), axis=1, keepdims=True)  # shape (num_test, 1)
    d3 = np.sum(np.square(matrix2), axis=1)  # shape (num_train, )
    dists = np.sqrt(d1 + d2 + d3)  # broadcasting
    return dists


def calculate_top_k(mat, top_k):
    size = mat.shape[0]
    gt_mat = np.expand_dims(np.arange(size), 1).repeat(size, 1)
    bool_mat = mat == gt_mat
    correct_vec = False
    top_k_list = []
    for i in range(top_k):
        #         print(correct_vec, bool_mat[:, i])
        correct_vec = correct_vec | bool_mat[:, i]
        # print(correct_vec)
        top_k_list.append(correct_vec[:, None])
    top_k_mat = np.concatenate(top_k_list, axis=1)
    return top_k_mat


def calculate_R_precision(embedding1, embedding2, top_k, sum_all=False):
    dist_mat = euclidean_distance_matrix(embedding1, embedding2)
    matching_score = dist_mat.trace()
    argmax = np.argsort(dist_mat, axis=1)
    top_k_mat = calculate_top_k(argmax, top_k)
    if sum_all:
        return top_k_mat.sum(axis=0), matching_score
    else:
        return top_k_mat, matching_score


def top_k_counts_from_distance_matrix(distance_matrix, max_k):
    # Argsort each row to get indices of sorted distances
    sorted_indices = np.argsort(distance_matrix, axis=1)

    # True labels are assumed to be the diagonal indices
    true_indices = np.arange(distance_matrix.shape[0])

    # Initialize an array to hold the counts for top-1 to top-max_k
    top_k_counts = np.zeros(max_k, dtype=int)

    # Check for each k from 1 to max_k
    for k in range(1, max_k + 1):
        # Check if the true index is within the top-k for each row
        top_k_matches = sorted_indices == true_indices[:, None]

        # Count the number of correct matches for this k
        top_k_counts[k - 1] = np.sum(top_k_matches)

    return top_k_matches


def calculate_R_precision_temporal(embedding1, embedding2, top_k, sum_all=False):
    B, T, D = embedding1.shape  # Assuming embedding1 and embedding2 have the same shape (B, T, D)
    accumulated_distance = np.zeros((B, B))

    # Calculate distance matrices for each time step and accumulate
    for t in range(T):
        dist_mat_t = euclidean_distance_matrix(embedding1[:, t, :], embedding2[:, t, :])
        accumulated_distance += dist_mat_t

    # Average the accumulated distance over the temporal dimension
    avg_distance = accumulated_distance / T

    matching_score = avg_distance.trace()
    argmax = np.argsort(avg_distance, axis=1)

    top_k_mat = calculate_top_k(argmax, top_k)
    if sum_all:
        return top_k_mat.sum(axis=0), matching_score
    else:
        return top_k_mat, matching_score


def calculate_multimodality(activation, multimodality_times):
    assert len(activation.shape) == 3
    assert activation.shape[1] > multimodality_times
    num_per_sent = activation.shape[1]

    first_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    second_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    dist = linalg.norm(activation[:, first_dices] - activation[:, second_dices], axis=2)
    return dist.mean()


def calculate_diversity(activation, diversity_times):
    assert len(activation.shape) == 2
    assert activation.shape[0] > diversity_times
    num_samples = activation.shape[0]

    first_indices = np.random.choice(num_samples, diversity_times, replace=False)
    second_indices = np.random.choice(num_samples, diversity_times, replace=False)
    dist = linalg.norm(activation[first_indices] - activation[second_indices], axis=1)
    return dist.mean()


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ("fid calculation produces singular product; " "adding %s to diagonal of cov estimates") % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def calculate_activation_statistics(activations):

    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mu, cov


def calculate_frechet_feature_distance(feature_list1, feature_list2):
    feature_list1 = np.stack(feature_list1)
    feature_list2 = np.stack(feature_list2)

    # normalize the scale
    mean = np.mean(feature_list1, axis=0)
    std = np.std(feature_list1, axis=0) + 1e-10
    feature_list1 = (feature_list1 - mean) / std
    feature_list2 = (feature_list2 - mean) / std

    dist = calculate_frechet_distance(
        mu1=np.mean(feature_list1, axis=0),
        sigma1=np.cov(feature_list1, rowvar=False),
        mu2=np.mean(feature_list2, axis=0),
        sigma2=np.cov(feature_list2, rowvar=False),
    )
    return dist