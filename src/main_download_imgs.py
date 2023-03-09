# Copyright (c) 2015-present, Meta Platforms, Inc. and affiliates.
# All rights reserved.
import logging
import os
import os.path as osp
import time
import urllib.request

import hydra
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm
from main_process_meta import reduce_to_closed_set, verify_closed_set

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.2",config_path="../configs/", config_name="process_meta")
def download_imgs(cfg: DictConfig):
    logger.info(cfg)
    os.chdir(hydra.utils.get_original_cwd())
    img_out_dir = osp.join(cfg.data_raw_dir, cfg.category_name)
    os.makedirs(img_out_dir, exist_ok=True)

    t0 = time.time()
    fbt_file_name = osp.join(cfg.data_processed_dir, f"{cfg.category_name}_processed.pkl")
    fbt_file_name_out = osp.join(
        cfg.data_processed_dir, f"{cfg.category_name}_processed_w_imgs.pkl"
    )
    logger.info(cfg)

    # Load data
    t1 = time.time()
    fbt_df = pd.read_pickle(fbt_file_name)
    logger.info(f"fbt_df read_pickle in {time.time()-t1:.1f}")

    # Download images
    logger.info("Download images")
    succeeds = []
    for i, (url, img_path) in tqdm(
        enumerate(zip(fbt_df.imageURLHighRes, fbt_df.img_path)), total=len(fbt_df)
    ):
        succeed = True
        if not osp.exists(img_path):
            try:
                urllib.request.urlretrieve(url, img_path)
            except Exception as e:
                logger.info(f"[{i}/{len(fbt_df)}] Fail {img_path}")
                succeed = False

        succeeds.append(succeed)

    # Reduce set
    len0 = len(fbt_df)
    fbt_df = fbt_df.iloc[succeeds]
    logger.info(f"succeeds [pre post]=[{len0} {len(fbt_df)}]")

    len0 = len(fbt_df)
    fbt_df = reduce_to_closed_set(fbt_df)
    verify_closed_set(fbt_df)
    logger.info(f"reduce_to_closed_set [pre post]=[{len0} {len(fbt_df)}]")

    logger.info(f"Saving to {fbt_file_name_out=}")
    fbt_df.to_pickle(fbt_file_name_out)
    logger.info(f"Finish in {time.time()-t0:.1f} sec")


if __name__ == "__main__":
    download_imgs()
