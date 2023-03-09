# Copyright (c) 2015-present, Meta Platforms, Inc. and affiliates.
# All rights reserved.
import logging
import os
import os.path as osp
from time import time
import hydra
import pytorch_lightning as pl
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from dataset_utils import FbtDataset, load_dfs
from lit.lit_utils import LitFbt
from lit.lit_dcf import LitDCF
from lit.lit_pcomp import LitPcomp

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.2", config_path="../configs/", config_name="execute_train")
def execute_train(cfg: DictConfig):
    t0 = time()
    out_dir = os.getcwd()
    os.chdir(hydra.utils.get_original_cwd())
    name = osp.basename(out_dir)
    pl.seed_everything(cfg.seed)

    wandb.init(
        project=cfg.wandb.project,
        dir=out_dir,
        config=OmegaConf.to_container(cfg),
        job_type=cfg.method,
        name=name,
    )
    logger.info(f"out_dir={out_dir}")
    logger.info(cfg)
    logger.info(f"{torch.backends.mps.is_available()=}")

    # Dataset
    t1 = time()
    fbt_df, emb_df = load_dfs(cfg)
    trainset = FbtDataset(fbt_df[fbt_df["set_name"] == "train"], emb_df)
    testset = FbtDataset(fbt_df[fbt_df["set_name"] == "test"], emb_df)
    dataset_all = FbtDataset(fbt_df, emb_df)
    logger.info(f"[train test]=[{len(trainset)} {len(testset)}]. {time()-t1:.1f}s")

    train_loader = DataLoader(
        trainset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
        shuffle=True,
    )
    test_loader = DataLoader(
        testset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
        shuffle=False,
    )
    data_all_loader = DataLoader(
        dataset_all,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
        shuffle=False,
    )

    # Initalize model
    num_categories = fbt_df["category_int_src"].max() + 1
    num_price_bins = fbt_df[["price_bin_src", "price_bin_target"]].max().max() + 1
    img_emb_dim = len(emb_df.img_embedding.iloc[0])
    logger.info(f"{[num_categories, num_price_bins, img_emb_dim]=}")

    if cfg.method == "gan":

        lit_model = LitFbt(
            img_emb_dim,
            cfg.img_encoder_output_dim,
            num_categories,
            cfg.category_emb_size,
            num_price_bins,
            cfg.price_emb_size,
            cfg,
            out_dir,
        )
    elif cfg.method == "dcf":
        lit_model = LitDCF(
            img_emb_dim,
            cfg.img_encoder_output_dim,
            num_categories,
            cfg.category_emb_size,
            num_price_bins,
            cfg.price_emb_size,
            cfg,
            out_dir,
        )
    elif cfg.method == "pcomp":
        lit_model = LitPcomp(
            img_emb_dim,
            cfg.img_encoder_output_dim,
            num_categories,
            cfg.category_emb_size,
            num_price_bins,
            cfg.price_emb_size,
            cfg,
            out_dir,
        )

    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        min_epochs=cfg.epochs,
        gradient_clip_val=cfg.gradient_clip_val,
        gradient_clip_algorithm="value",
        devices=1,
        logger=WandbLogger(experimnet=wandb.run),
        callbacks=[LearningRateMonitor()],
        num_sanity_val_steps=0,
        default_root_dir=out_dir,
        accelerator="cpu",
        check_val_every_n_epoch=cfg.check_val_every_n_epoch,
    )
    t1 = time()
    trainer.fit(lit_model, train_loader, [test_loader, data_all_loader])
    logger.info(f"trainer.fit in {time()-t1:.1f} s")

    # Upload fiels
    wandb.save(osp.join(out_dir, "*.pth"))
    wandb.save(osp.join(out_dir, "*.ckpt"))
    logger.info(f"Finish execute_train in {time()-t0:.1f} s")


if __name__ == "__main__":
    execute_train()
