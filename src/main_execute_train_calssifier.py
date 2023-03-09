# Copyright (c) 2015-present, Meta Platforms, Inc. and affiliates.
# All rights reserved.
import logging
import os
import os.path as osp
import time

import hydra
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, TensorDataset

from lit.lit_utils import LitCategoryClassifier

logger = logging.getLogger(__name__)


@hydra.main(
    version_base="1.2",
    config_path="../configs/",
    config_name="execute_train_classifier",
)
def execute_train(cfg: DictConfig):
    t0 = time.time()
    out_dir = os.getcwd()
    os.chdir(hydra.utils.get_original_cwd())
    name = osp.basename(out_dir)
    pl.seed_everything(cfg.seed)

    wandb.init(
        project=cfg.wandb.project,
        dir=out_dir,
        config=OmegaConf.to_container(cfg),
        job_type="train_classifier",
        name=name,
    )
    logger.info(f"out_dir={out_dir}")
    logger.info(cfg)

    # Load data
    pkl_path = osp.join(cfg.data_dir, cfg.category_name + "_sets.pkl")
    fbt_df = pd.read_pickle(pkl_path)
    fbt_df = fbt_df[["asin_src", "category_int_src"]].drop_duplicates().reset_index()

    emb_path = osp.join(cfg.data_dir, cfg.category_name + "_embeddings.pkl")
    emb_df = pd.read_pickle(emb_path)

    df = pd.merge(fbt_df, emb_df, "inner", left_on="asin_src", right_on="asin")
    num_classes = df.category_int_src.max() + 1
    category_onehot = F.one_hot(
        torch.tensor(df["category_int_src"].tolist()), num_classes=num_classes
    )
    embs = torch.tensor(df["img_embedding"].tolist())
    input_emb_dim = embs.size(1)

    # Dataset
    dataset = TensorDataset(embs, category_onehot)
    len_train = int(len(dataset) * cfg.train_ratio)
    trainset, testset = torch.utils.data.random_split(
        dataset, [len_train, len(dataset) - len_train]
    )
    logger.info(f"Dataset size: [train test]=[{len(trainset)} {len(testset)}]")
    logger.info("Category bincount:")
    logger.info(torch.bincount(torch.tensor(fbt_df.category_int_src.tolist())))

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

    # Initalize model
    lit_model = LitCategoryClassifier(input_emb_dim, num_classes, cfg)

    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        min_epochs=cfg.epochs,
        gpus=1 if torch.cuda.is_available() else None,
        logger=WandbLogger(experimnet=wandb.run),
        callbacks=[LearningRateMonitor()],
        num_sanity_val_steps=0,
        default_root_dir=None,
        accelerator="mps",
    )
    trainer.fit(lit_model, train_loader, test_loader)

    logger.info(f"Finish execute_train in {time.time()-t0:.1f} sec")


if __name__ == "__main__":
    execute_train()
