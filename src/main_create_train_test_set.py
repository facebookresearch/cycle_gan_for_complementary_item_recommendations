# Copyright (c) 2015-present, Meta Platforms, Inc. and affiliates.
# All rights reserved.
import logging
import os
import os.path as osp
import time

import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder

logger = logging.getLogger(__name__)


def remove_unique_categories(
    df: pd.DataFrame, category_freq_threshold: int = 20
) -> pd.DataFrame:
    categories = df.category.tolist()
    criterion, num_iter = 1, 0
    while criterion:
        categories = df.category.tolist()
        category_unique, category_count = np.unique(categories, return_counts=True)
        df["category_count"] = df.category.apply(
            lambda cat: category_count[cat == category_unique][0]
        )

        # If category is too rare: remove the leaf (..//..)
        df.category = df[["category", "category_count"]].apply(
            lambda row: row.category.rsplit("//", 1)[0]
            if row.category_count < category_freq_threshold
            else row.category,
            axis=1,
        )
        criterion = (df.category_count < category_freq_threshold).sum()

        print(f"{criterion=} {num_iter=}")
        num_iter += 1
    return df


@hydra.main(config_path="../configs/", config_name="create_train_test_sets")
def create_train_test_sets(cfg: DictConfig):
    t0 = time.time()
    logger.info(cfg)
    pl.seed_everything(1234)
    category_name = cfg.category_name
    out_dir = os.getcwd()
    processed_pkl_path = osp.join(
        cfg.data_processed_dir, f"{category_name}_processed_w_imgs.pkl"
    )
    out_pkl_path = osp.join(cfg.data_final_dir, f"{category_name}_sets.pkl")

    # Load data
    t1 = time.time()
    df = pd.read_pickle(processed_pkl_path)
    df.category = df.category.apply(lambda list_str: "//".join(list_str)).astype(str)

    emb_path = osp.join(cfg.data_final_dir, cfg.category_name + "_embeddings.pkl")
    df_embs = pd.read_pickle(emb_path)
    logger.info(f"Load df in {time.time()-t1:.1f} sec")

    # Original category
    category_unique, category_count = np.unique(
        df.category.tolist(), return_counts=True
    )
    idxs = np.argsort(category_count)[::-1]
    pd.DataFrame(
        {"category_name": category_unique[idxs], "count": category_count[idxs]}
    ).to_csv(osp.join(out_dir, "categories_pre.csv"), index=False)

    # Combine categories
    df_trans = df.copy()
    df_trans = remove_unique_categories(
        df_trans, category_freq_threshold=cfg.category_freq_threshold
    )

    # New categories
    category_unique, category_count = np.unique(
        df_trans.category.tolist(), return_counts=True
    )
    idxs = np.argsort(category_count)[::-1]
    pd.DataFrame(
        {"category_name": category_unique[idxs], "count": category_count[idxs]}
    ).to_csv(osp.join(out_dir, "categories_post.csv"), index=False)

    # Map categories to one hot
    df_trans["category_int"] = LabelEncoder().fit_transform(df_trans.category)
    df_fbt = df_trans.explode(column="also_buy").reset_index()
    df_merge = pd.merge(
        df_trans,
        df_fbt,
        how="inner",
        left_on="asin",
        right_on="also_buy",
        suffixes=["_src", "_target"],
    )
    df_merge = df_merge[
        [
            "asin_src",
            "category_src",
            "category_int_src",
            "price_src",
            "asin_target",
            "category_target",
            "category_int_target",
            "price_target",
        ]
    ]
    logger.info(f"[pre explode merge]=[{len(df_trans)} {len(df_fbt)} {len(df_merge)}]")

    # Keep only pairs with different category
    df_with_set_split = df_merge.copy()
    df_with_set_split = df_with_set_split[
        df_with_set_split.category_int_src != df_with_set_split.category_int_target
    ]

    # Keep only asins with images
    asins_to_keep = df_embs.asin.tolist()
    df_with_set_split = df_with_set_split[
        df_with_set_split["asin_src"].isin(asins_to_keep)
        & df_with_set_split["asin_target"].isin(asins_to_keep)
    ]

    # source in test won't be in target
    asin_srcs = df_with_set_split.asin_src.unique()
    asin_src_train, asin_src_test = train_test_split(
        asin_srcs, train_size=cfg.train_set_ratio, random_state=cfg.seed
    )
    df_with_set_split["set_name"] = None
    df_with_set_split["set_name"][
        df_with_set_split.asin_src.isin(asin_src_train)
    ] = "train"
    df_with_set_split["set_name"][
        df_with_set_split.asin_src.isin(asin_src_test)
    ] = "test"
    train_ratio = (df_with_set_split.set_name == "train").sum() / len(df_with_set_split)
    test_ratio = (df_with_set_split.set_name == "test").sum() / len(df_with_set_split)
    logger.info(f"{[train_ratio, test_ratio]=}. {df_with_set_split.shape=}")

    # Remove pairs in training were the target is a source in test
    fbt_df_train = df_with_set_split[df_with_set_split["set_name"] == "train"]
    fbt_df_test = df_with_set_split[df_with_set_split["set_name"] == "test"]
    len0 = len(fbt_df_train)
    fbt_df_train = fbt_df_train[~fbt_df_train.asin_target.isin(fbt_df_test.asin_src)]
    logger.info(
        f"Remove pairs in fbt_df_train. Size [pre post]=[{len0} {len(fbt_df_train)}]"
    )

    # Price bin
    est = KBinsDiscretizer(n_bins=cfg.price_n_bins, encode="ordinal")
    est.fit(
        np.hstack([fbt_df_train["price_src"], fbt_df_train["price_target"]])[
            :, np.newaxis
        ]
    )
    fbt_df_train["price_bin_src"] = (
        est.transform(fbt_df_train["price_src"].to_numpy()[:, np.newaxis])
        .squeeze()
        .astype(int)
    )
    fbt_df_train["price_bin_target"] = (
        est.transform(fbt_df_train["price_target"].to_numpy()[:, np.newaxis])
        .squeeze()
        .astype(int)
    )
    fbt_df_test["price_bin_src"] = (
        est.transform(fbt_df_test["price_src"].to_numpy()[:, np.newaxis])
        .squeeze()
        .astype(int)
    )
    fbt_df_test["price_bin_target"] = (
        est.transform(fbt_df_test["price_target"].to_numpy()[:, np.newaxis])
        .squeeze()
        .astype(int)
    )

    # Concate
    df_with_set_split = pd.concat([fbt_df_train, fbt_df_test])
    logger.info(f"{[len(fbt_df_train), len(fbt_df_test)]=}")
    df_with_set_split.to_pickle(out_pkl_path)
    logger.info(f"Finish in {time.time()-t0:.1f} sec. {out_pkl_path=}")


if __name__ == "__main__":
    create_train_test_sets()
