# Copyright (c) 2015-present, Meta Platforms, Inc. and affiliates.
# All rights reserved.
import gzip
import itertools
import json
import logging
import os.path as osp
import time

import hydra
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm
import numpy as np

logger = logging.getLogger(__name__)


def parse(path):
    g = gzip.open(path, "rb")
    for l in g:
        yield json.loads(l)


def getDF(path):
    i = 0
    df = {}
    for d in tqdm(parse(path)):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient="index")


def convert_price_string(x):
    x = x.replace("$", "").replace(" ", "").replace(",", "")
    price_range = x.split("-")
    for price_limit in price_range:
        try:
            float(price_limit)
        except:
            return pd.NA
    return np.array(price_range).astype(float).mean()


def verify_closed_set(fbt_set: pd.DataFrame):
    # Verify it is a closed set
    asins_also_buy = list(itertools.chain(*fbt_set.also_buy.tolist()))
    set1, set2 = set(asins_also_buy), set(fbt_set.asin.tolist())
    assert len(set1) == len(set2)


# Keep only also_buy items that exist in asins
def filter_also_buy(also_buy_list_, asins_):
    return list(set(also_buy_list_) & asins_)


def reduce_to_closed_set(fbt_df: pd.DataFrame) -> pd.DataFrame:
    # Keep only items that exist in also_buy
    start_len, finish_len = -1, 1
    fbt_df_start = fbt_df
    round_ = 0
    while start_len != finish_len:
        t1 = time.time()
        asins_also_buy = list(itertools.chain(*fbt_df_start.also_buy.tolist()))
        fbt_df_set_step0 = fbt_df_start[fbt_df_start.asin.isin(asins_also_buy)]

        asins = set(fbt_df_set_step0.asin.tolist())
        fbt_df_set_step0["also_buy"] = fbt_df_set_step0.also_buy.apply(
            lambda also_buy_list: filter_also_buy(also_buy_list, asins)
        )

        # Filter
        mask = fbt_df_set_step0["also_buy"].apply(
            lambda x: True if len(x) > 0 else False
        )
        fbt_df_set_step1 = fbt_df_set_step0[mask]
        fbt_df_finish = fbt_df_set_step1

        # Anlysis if some rows where reduced
        start_len, finish_len = len(fbt_df_start), len(fbt_df_finish)
        logger.info(
            f"reduce_to_closed_set: Round {round_}. [pre post]=[{start_len} {finish_len}]. {time.time()-t1:.2f} sec"
        )
        fbt_df_start = fbt_df_finish
        round_ += 1
    return fbt_df_start


@hydra.main(version_base="1.2", config_path="../configs/", config_name="process_meta")
def process_meta(cfg: DictConfig):
    t0 = time.time()
    logger.info(cfg)
    category_name = cfg.category_name
    out_path_pkl = osp.join(cfg.data_processed_dir, f"{category_name}_processed.pkl")
    meta_file_name = osp.join(cfg.data_raw_dir, f"meta_{category_name}.json.gz")
    fbt_file_name = osp.join(cfg.data_processed_dir, f"{category_name}_fbt.pkl")

    # Read meta file
    if not osp.exists(fbt_file_name):
        logger.info(f"Reading {meta_file_name}")
        t1 = time.time()
        fbt_df = getDF(meta_file_name)
        fbt_df.to_pickle(fbt_file_name)
        logger.info(f"getDF in {time.time()-t1:.1f} sec")
    t1 = time.time()
    fbt_df = pd.read_pickle(fbt_file_name)[
        ["asin", "category", "also_buy", "price", "imageURLHighRes"]
    ]
    logger.info(f"fbt_df read_pickle in {time.time()-t1:.1f} sec")

    # First cleaning meta
    t1 = time.time()
    len0 = len(fbt_df)
    fbt_df = fbt_df.dropna()
    fbt_df = fbt_df[fbt_df["imageURLHighRes"].apply(lambda urls: len(urls) > 0)]
    fbt_df["imageURLHighRes"] = fbt_df["imageURLHighRes"].apply(
        lambda urls: urls[0]
    )  # Keep only 1 url
    fbt_df["price"] = fbt_df["price"].apply(convert_price_string)
    fbt_df = fbt_df.dropna()
    logger.info(
        f"First cleaning meta in {time.time()-t1:1f} sec. [pre post]=[{len0} {len(fbt_df)}]"
    )

    # Megrge duplicate entries
    t1 = time.time()
    len0 = len(fbt_df)
    fbt_df = fbt_df.groupby(["asin"]).agg(
        {
            "also_buy": "sum",
            "category": "first",
            "price": "first",
            "imageURLHighRes": "first",
        },
        as_index=False,
    )
    fbt_df.also_buy = fbt_df.also_buy.apply(lambda also_buy: list(set(also_buy)))
    fbt_df = fbt_df.reset_index()
    logger.info(
        f"merge_duplicate_entries in {time.time() -t1:.1f} sec. [pre post]=[{len0} {len(fbt_df)}]"
    )

    # Keep only items that exist in also_buy
    t1 = time.time()
    len0 = len(fbt_df)
    fbt_set = reduce_to_closed_set(fbt_df)
    verify_closed_set(fbt_set)
    logger.info(
        f"reduce_to_closed_set in {time.time() -t1:.1f} sec. [pre post]=[{len0} {len(fbt_set)}]"
    )

    # Save dataframe
    t1 = time.time()
    len0 = len(fbt_set)
    fbt_set = fbt_set.dropna()
    fbt_set = fbt_set.reset_index()
    fbt_set["img_path"] = fbt_set["imageURLHighRes"].apply(
        lambda url: osp.join(cfg.data_raw_dir, category_name, osp.basename(url))
    )
    fbt_set.to_pickle(out_path_pkl)
    logger.info(
        f"saved dataframe in {time.time() -t1:.1f} sec. [pre post]=[{len0} {len(fbt_set)}]"
    )
    logger.info(out_path_pkl)
    logger.info(f"Finish in {time.time()-t0:.1f} sec")


if __name__ == "__main__":
    process_meta()
