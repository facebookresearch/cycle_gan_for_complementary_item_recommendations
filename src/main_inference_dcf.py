# Copyright (c) 2015-present, Meta Platforms, Inc. and affiliates.
# All rights reserved.
import itertools
import logging
import os
import os.path as osp
from os.path import join as osj
from time import time

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from inferece_utils import calc_ndcg_per_category, calc_performance, get_dataloaders
from lit.eval_utils import generate_test_set_hot_labels, get_unique_asins
from lit.lit_dcf import LitDCF

logger = logging.getLogger(__name__)


def dcf_model_inference(lit_model, category_src, price_bin, img_emb, candidates):
    category_emb = lit_model.category_embs(category_src)
    price_emb = lit_model.price_embs(price_bin)
    src = lit_model.src_encoder(img_emb, category_emb, price_emb)

    # Fusion layer
    logits_i = lit_model.fusion_model(
        src.repeat(len(candidates), 1), candidates
    ).squeeze()
    return logits_i


def get_candidate_embeddings(lit_model, candidate_loader):
    candidates, asins, categories = [], [], []
    with torch.no_grad():
        for img_emb, price_bin, category, asin in tqdm(candidate_loader):

            category_emb = lit_model.category_embs(category)
            price_emb = lit_model.price_embs(price_bin)
            candidate = lit_model.candidate_encoder(img_emb, category_emb, price_emb)

            candidates.append(candidate.detach().cpu())
            asins.append(asin)
            categories.append(category.detach().cpu())

    candidates = torch.vstack(candidates)
    asins = np.array(list(itertools.chain(*asins)))
    categories = torch.hstack(categories)
    return candidates, categories, asins


def evaluate_dcf(
    lit_model, inference_loader, candidates, category_candidates, candidate_asins, cfg
):
    # Get valid cateogires
    hot_labels, logits, category_targets, asin_srcs = [], [], [], []
    for (
        img_emb,
        price_bin,
        category_src,
        asin_src,
        _,
        target_asins,
    ) in tqdm(inference_loader):

        logits_i = dcf_model_inference(
            lit_model, category_src, price_bin, img_emb, candidates
        )

        # Create ground true label
        target_asins = [asin_target[0] for asin_target in target_asins]
        hot_labels_i = np.in1d(
            candidate_asins,
            target_asins,
        )
        assert hot_labels_i.sum() > 0

        # Save
        hot_labels.append(hot_labels_i)
        logits.append(logits_i.squeeze())
        category_targets_i = category_candidates[hot_labels_i]
        category_targets.append(category_targets_i.unique().tolist())
        asin_srcs.append(asin_src)

    # Calcualte probability
    logits = torch.vstack(logits)
    hot_labels = np.vstack(hot_labels)

    # Retrival metrics
    calc_performance(hot_labels, logits, cfg)


# def evaluate_category_aware_dcf(lit_model, dataset, cfg, out_dir):
def evaluate_category_aware_dcf(lit_model, dataset, cfg, out_dir):
    path = cfg.model_weight_dir
    model_base_dir = osp.basename(path)
    asin_src = torch.load(osj(path, "asin_src.pth"))
    asin_pos = torch.load(osj(path, "asin_pos.pth"))
    category_src = torch.from_numpy(torch.load(osj(path, "category_src.pth")))
    category_pos = torch.from_numpy(torch.load(osj(path, "category_pos.pth")))
    src = torch.load(osj(path, "src.pth"))
    pos = torch.load(osj(path, "pos.pth"))
    set_name = torch.load(osj(path, "set_name.pth"))

    # Test sources: have a unique pair of (source,target-category)
    src_test = src[set_name == "test"]
    asin_src_test = asin_src[set_name == "test"]
    category_pos_test = category_pos[set_name == "test"]

    locs = list(zip(asin_src_test, category_pos_test))
    _, unique_idxs = np.unique(np.array(locs), axis=0, return_index=True)
    src_test, asin_src_test, category_pos_test = (
        src_test[unique_idxs],
        asin_src_test[unique_idxs],
        category_pos_test[unique_idxs],
    )

    # Candidate to compare with
    asins = np.hstack([asin_src, asin_pos])
    embs = torch.vstack([src, pos])
    categories = torch.hstack([category_src, category_pos])
    asins, embs, categories = get_unique_asins(asins, embs, categories)

    # Build hot label
    hot_labels, asin_src_test = generate_test_set_hot_labels(
        asin_src_test=asin_src_test,
        category_pos_test=category_pos_test.numpy(),
        fbt_by_asin_src=dataset.fbt_by_asin_src,
        asins=asins,
    )

    # Find distance of the candidates
    probs = []
    for src_test_i in tqdm(src_test):
        src_to_inference = src_test_i.repeat(len(embs), 1)
        logits = lit_model.fusion_model(src_to_inference, embs)
        prob = logits.sigmoid().squeeze()
        probs.append(prob)
    probs = torch.vstack(probs).cpu().numpy()

    # Constrain to target
    for n, cat in enumerate(category_pos_test):
        probs[n, categories != cat] = 0

    # Calculate retrival metrics
    calc_performance(hot_labels, probs, cfg)
    calc_ndcg_per_category(
        hot_labels, probs, asin_src_test, category_pos_test, model_base_dir, out_dir
    )


@hydra.main(version_base="1.2", config_path="../configs/", config_name="inference")
def execute_dcf_inference(cfg: DictConfig):
    t0 = time()
    out_dir = os.getcwd()
    os.chdir(hydra.utils.get_original_cwd())
    name = osp.basename(out_dir)
    pl.seed_everything(cfg.seed)

    wandb.init(
        project=cfg.wandb.project,
        dir=out_dir,
        config=OmegaConf.to_container(cfg),
        job_type="analysis",
        name="analysis_" + name,
    )

    logger.info(f"out_dir={out_dir}")
    logger.info(cfg)
    logger.info(f"{torch.backends.mps.is_available()=}")
    for category_name, model_dcf_weight_dir in zip(
        cfg.dcf_category_names, cfg.model_dcf_weight_dirs
    ):
        t1 = time()
        logger.info(category_name)

        # Dataset
        cfg.category_name = category_name
        cfg.model_weight_dir = model_dcf_weight_dir

        candidate_loader, inference_loader, dataset = get_dataloaders(cfg)

        # Load weights
        checkpoint = osj(cfg.model_weight_dir, "checkpoint.ckpt")
        logger.info(checkpoint)
        lit_model = LitDCF.load_from_checkpoint(checkpoint)
        lit_model.eval()
        torch.set_grad_enabled(False)

        candidates, category_candidates, candidate_asins = get_candidate_embeddings(
            lit_model, candidate_loader
        )

        evaluate_dcf(
            lit_model,
            inference_loader,
            candidates,
            category_candidates,
            candidate_asins,
            cfg,
        )
        evaluate_category_aware_dcf(
            lit_model,
            dataset,
            cfg,
            out_dir,
        )

        logger.info(f"Finish {category_name} in {time() - t1:.1f} s")
    logger.info(f"Finish execute_dcf_inference in {time() - t0:.1f} s")


if __name__ == "__main__":
    execute_dcf_inference()
