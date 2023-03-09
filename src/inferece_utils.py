# Copyright (c) 2015-present, Meta Platforms, Inc. and affiliates.
# All rights reserved.
import json
import logging

import pandas as pd
import torch
from sklearn.metrics import ndcg_score
from tqdm import tqdm
import numpy as np
from lit.eval_utils import calc_topk
from os.path import join as osj
from dataset_utils import FbtCandidateDataset, FbtDataset, FbtInferenceDataset, load_dfs
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def calc_performance(hot_labels, probs, cfg):
    logger.info(f"{cfg.model_weight_dir}")

    if isinstance(hot_labels, np.ndarray):
        hot_labels = torch.from_numpy(hot_labels)
    if isinstance(probs, np.ndarray):
        probs = torch.from_numpy(probs)

    logger.info("\nCoverage")
    _, sort_idxs = torch.sort(probs, dim=-1, descending=True)
    logger.info(
        f"TOP10 unique items {sort_idxs[:, :10].unique().shape}. {sort_idxs.shape=}"
    )

    logger.info("NDCG")
    ndcg_val_at_k = {}
    for k in cfg.top_k + [probs.shape[-1]]:
        ndcg_val_at_k[k] = ndcg_score(hot_labels, probs, k=k)
    logger.info(json.dumps(ndcg_val_at_k, sort_keys=True, indent=4))

    logger.info("\nTOPK")
    topk_d = calc_topk(probs, hot_labels, top_k_list=cfg.top_k)
    topk_d = {int(key.replace("topk/top", "")): value for key, value in topk_d.items()}
    logger.info(json.dumps(topk_d, sort_keys=True, indent=4))


def calc_ndcg_per_category(
    hot_labels, probs, asin_src_test, category_pos_test, model_base_dir, out_dir
):

    if isinstance(hot_labels, np.ndarray):
        hot_labels = torch.from_numpy(hot_labels)
    if isinstance(probs, np.ndarray):
        probs = torch.from_numpy(probs)

    ndcg_vals = [
        ndcg_score(hot_label_i.unsqueeze(0), prob_i.unsqueeze(0))
        for hot_label_i, prob_i in tqdm(zip(hot_labels, probs))
    ]
    out_path = osj(out_dir, f"{model_base_dir}.pkl")
    pd.DataFrame(
        {
            "asins": asin_src_test,
            "category_pos_test": category_pos_test,
            "ndcg": ndcg_vals,
        }
    ).to_pickle(out_path)
    logger.info(f"Finish predictor. {out_path}")


def get_dataloaders(cfg):
    fbt_df, emb_df = load_dfs(cfg)
    fbt_df_train, fbt_df_test = (
        fbt_df[fbt_df["set_name"] == "train"],
        fbt_df[fbt_df["set_name"] == "test"],
    )
    dataset = FbtDataset(fbt_df, emb_df)
    logger.info(f"{[len(fbt_df_train), len(fbt_df_test), len(dataset)]=}")

    candidate_dataset = FbtCandidateDataset(fbt_df, emb_df)
    inference_dataset = FbtInferenceDataset(fbt_df, emb_df)

    candidate_loader = DataLoader(
        candidate_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
        shuffle=False,
    )

    inference_loader = DataLoader(
        inference_dataset,
        batch_size=1,
        num_workers=0,
        pin_memory=True,
        shuffle=False,
    )
    return candidate_loader, inference_loader, dataset
