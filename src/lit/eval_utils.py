# Copyright (c) 2015-present, Meta Platforms, Inc. and affiliates.
# All rights reserved.
import logging

import numpy as np
import torch
import pandas as pd
from os.path import join as osj

logger = logging.getLogger(__name__)


def generate_test_set_hot_labels(
    asin_src_test: np.ndarray,
    category_pos_test: np.ndarray,
    fbt_by_asin_src: pd.DataFrame,
    asins: np.ndarray,
) -> torch.Tensor:
    """_summary_

    Args:
        asin_src_test (_type_): asin to use as source
        category_pos_test (_type_): target category of asin_src_test
        fbt_df (_type_): all dataset of also_buy
        asins (_type_): what to consider as also buy
    """
    hot_label_series = fbt_by_asin_src.apply(lambda x: np.in1d(asins, x))
    locs = list(zip(asin_src_test, category_pos_test))
    hot_labels = torch.from_numpy(np.vstack(hot_label_series.loc[locs].values))
    return hot_labels, asin_src_test


def vectorize_sort(x, permutation):
    # Order tensor by indecis
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    d1, d2 = x.size()
    ret = x[
        torch.arange(d1).unsqueeze(1).repeat((1, d2)).flatten(),
        permutation.flatten(),
    ].view(d1, d2)
    return ret


def get_unique_asins(asins, embs, category_pos):
    _, idx_of_unique = np.unique(asins, return_index=True)
    idx_of_unique = np.sort(idx_of_unique)
    asins_unique = asins[idx_of_unique]
    embs_unique = embs[idx_of_unique]
    category_pos_unique = category_pos[idx_of_unique]
    return asins_unique, embs_unique, category_pos_unique


def calc_topk(probs, hot_labels, top_k_list):
    _, sort_idxs = torch.sort(probs, dim=-1, descending=True)
    is_true = vectorize_sort(hot_labels, sort_idxs)

    topk_d = {}
    for k in top_k_list:
        topk = torch.any(is_true[:, :k], axis=-1).float().mean().item()
        topk_d[f"topk/top{k}"] = topk
    return topk_d


def create_pop_cat_aware_predictor(
    fbt_df: pd.DataFrame, candidate_asins: np.ndarray
) -> dict:

    # All asins probability are set to 0
    pred_init = pd.DataFrame(
        {"asins": candidate_asins, "freq": [0] * len(candidate_asins)}
    ).set_index("asins")

    pred_dicts = {}
    for category_int_target, df_gb in fbt_df.groupby(by=["category_int_target"]):
        s = df_gb["asin_target"].value_counts(ascending=False)

        pred = pred_init.copy()

        # Set prediction probability of an asins by its frequency
        pred["freq"].loc[s.index] = s.values
        pred = pred["freq"].to_numpy()
        pred_dicts[category_int_target] = pred / pred.sum()

    # pred_dicts = {category_int_target: [asin1_prob, asin2_prob, ... , asinsN_prob]}
    return pred_dicts


def create_pop_predictor(fbt_df: pd.DataFrame, candidate_asins: np.ndarray) -> dict:

    # All asins probability are set to 0
    pred = pd.DataFrame(
        {"asins": candidate_asins, "freq": [0] * len(candidate_asins)}
    ).set_index("asins")

    # asins by popularity
    s = fbt_df["asin_target"].value_counts(ascending=False)
    pred["freq"].loc[s.index] = s.values
    pred = pred["freq"].to_numpy()
    return pred / pred.sum()


def save_run_results(tensor_dict: dict, out_dir: str):
    for key, val in tensor_dict.items():
        torch.save(val, osj(out_dir, f"{key}.pth"))
