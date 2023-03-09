# Copyright (c) 2015-present, Meta Platforms, Inc. and affiliates.
# All rights reserved.
import json
import logging
import os
from os.path import join as osj
from time import time

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from sklearn.metrics import ndcg_score
from tqdm import tqdm

from dataset_utils import FbtDataset, load_dfs
from lit.eval_utils import (
    calc_topk,
    create_pop_cat_aware_predictor,
    create_pop_predictor,
    generate_test_set_hot_labels,
)
from inferece_utils import calc_ndcg_per_category, calc_performance, get_dataloaders

logger = logging.getLogger(__name__)

plt.style.use(["science", "ieee"])


def build_hot_labels_for_popularity_based_preictor(dataset, fbt_df):
    # Build hot labels
    t1 = time()
    fbt_df_test = fbt_df[fbt_df["set_name"] == "test"]
    fbt_by_asin_src = dataset.fbt_by_asin_src
    asins = np.unique(np.hstack([fbt_df.asin_src, fbt_df.asin_target]))
    indices = np.array(
        list(fbt_df_test.groupby(by=["asin_src", "category_int_target"]).indices.keys())
    )
    asin_src_test, category_pos_test = indices[:, 0], indices[:, 1].astype(int)
    hot_labels, asin_src_test = generate_test_set_hot_labels(
        asin_src_test, category_pos_test, fbt_by_asin_src, asins
    )
    return hot_labels, asins, asin_src_test, category_pos_test


def execute_analyse(cfg: DictConfig, out_dir):

    fbt_df, emb_df = load_dfs(cfg)
    fbt_df_train, fbt_df_test = (
        fbt_df[fbt_df["set_name"] == "train"],
        fbt_df[fbt_df["set_name"] == "test"],
    )
    dataset = FbtDataset(fbt_df, emb_df)
    logger.info(f"{[len(fbt_df_train), len(fbt_df_test), len(dataset)]=}")
    # Build hot labels
    t1 = time()
    (
        hot_labels,
        asins,
        asin_src_test,
        category_pos_test,
    ) = build_hot_labels_for_popularity_based_preictor(dataset, fbt_df)
    logger.info(f"{hot_labels.shape=} {time()-t1:.1f}s")

    # ------------------------------------------------
    # Most popular predictor
    # ------------------------------------------------
    logger.info("Most popular predictor")
    t1 = time()
    probs = create_pop_predictor(fbt_df_train, asins)

    # Predict based on popularity
    probs = torch.from_numpy(probs).repeat(hot_labels.shape[0], 1)

    # Calculate retrival metrics
    calc_performance(hot_labels, probs, cfg)

    # ------------------------------------------------
    # Most popular predictor: category aware
    # ------------------------------------------------
    logger.info("Most popular predictor: category aware")
    t1 = time()
    pred_dicts = create_pop_cat_aware_predictor(fbt_df_train, asins)
    probs = torch.tensor(
        np.vstack([pred_dicts[cat][np.newaxis, :] for cat in category_pos_test])
    )
    # Calculate retrival metrics
    calc_performance(hot_labels, probs, cfg)
    calc_ndcg_per_category(
        hot_labels,
        probs,
        asin_src_test,
        category_pos_test,
        f"pop_category_aware_pred.pkl_{cfg.category_name}",
        out_dir,
    )


@hydra.main(version_base="1.2", config_path="../configs/", config_name="inference")
def execute_most_pop_inference(cfg: DictConfig):
    out_dir = os.getcwd()
    os.chdir(hydra.utils.get_original_cwd())
    pl.seed_everything(cfg.seed)

    for category_name in cfg.most_pop_category_names:
        cfg.category_name = category_name
        execute_analyse(cfg, out_dir)


if __name__ == "__main__":
    execute_most_pop_inference()
