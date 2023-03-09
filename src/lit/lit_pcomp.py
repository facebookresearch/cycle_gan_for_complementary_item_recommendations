# Copyright (c) 2015-present, Meta Platforms, Inc. and affiliates.
# All rights reserved.
import logging
import time
from os.path import join as osj
from time import time

import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from sklearn.metrics import ndcg_score
from torch import nn, optim
from torch.nn.functional import mse_loss
from .eval_utils import (
    calc_topk,
    generate_test_set_hot_labels,
    get_unique_asins,
    save_run_results,
)

from .lit_utils import LearnedEmbs, ImageEncoder

logger = logging.getLogger(__name__)


class LitPcomp(pl.LightningModule):
    def __init__(
        self,
        input_emb_dim: int,
        emb_dim: int,
        num_categories: int,
        category_emb_size: int,
        num_price_bins: int,
        price_emb_size: int,
        cfg,
        out_dir: str = ".",
    ):
        self.cfg = cfg
        self.out_dir = out_dir
        self.save_hyperparameters()
        super().__init__()

        # Architecture
        self.num_categories = num_categories
        self.category_embs = LearnedEmbs(num_categories, category_emb_size)
        self.price_embs = LearnedEmbs(num_price_bins, price_emb_size)
        self.img_encoder = ImageEncoder(
            input_emb_dim, category_emb_size, price_emb_size, emb_dim, cfg.dropout_rate
        )
        self.fbt_categories = LearnedEmbs(num_categories, emb_dim)

        logger.info(self.category_embs)
        logger.info(self.price_embs)
        logger.info(self.img_encoder)

        # Performance
        self.ndcg_val_best = 0.0
        wandb.define_metric("retrieval_metrics/ndcg", summary="max")

    def log(
        self,
        *args,
        **kwargs,
    ) -> None:
        kwargs["on_epoch"] = True
        kwargs["on_step"] = False
        return super().log(*args, **kwargs)

    def _loss_helper(
        self, batch, phase="train", batch_idx: int = 0, optimizer_idx: int = 0
    ):
        (
            img_emb_src,
            img_emb_pos,
            img_emb_neg,
            price_bin_src,
            price_bin_pos,
            price_bin_neg,
            category_src,
            category_pos,
            random_valid_category,
            asin_src,
            asin_pos,
            set_name,
        ) = batch
        category_src_emb = self.category_embs(category_src)
        category_dst_emb = self.category_embs(category_pos)

        price_src_emb = self.price_embs(price_bin_src)
        price_pos_emb = self.price_embs(price_bin_pos)
        price_neg_emb = self.price_embs(price_bin_neg)

        src = self.img_encoder(img_emb_src, category_src_emb, price_src_emb)
        pos = self.img_encoder(img_emb_pos, category_dst_emb, price_pos_emb)
        neg = self.img_encoder(img_emb_neg, category_dst_emb, price_neg_emb)

        src_fbt = (self.fbt_categories(category_pos) + 1) * src

        # Triplet
        zeros = torch.zeros_like(src_fbt)
        loss_pos = torch.maximum(
            zeros,
            self.cfg.epsilon
            - (self.cfg.lamb - mse_loss(src_fbt, pos, reduction="none")),
        ).mean()
        loss_neg = torch.maximum(
            zeros,
            self.cfg.epsilon
            + (self.cfg.lamb - mse_loss(src_fbt, neg, reduction="none")),
        ).mean()
        loss = 0.5 * (loss_pos + loss_neg)

        # Logger
        self.log(f"loss/{phase}/loss_pos", loss_pos)
        self.log(f"loss/{phase}/loss_neg", loss_neg)
        self.log(f"loss/{phase}", loss)

        # Logger
        src_fbt_std_mean = src_fbt.mean(axis=-1).mean()
        src_fbt_std = src_fbt.std(axis=-1).mean()
        src_fbt_avg_norm = torch.norm(src_fbt, dim=-1).mean()
        category_emb_mean = category_src_emb.mean(axis=-1).mean()
        category_emb_avg_norm = torch.norm(category_src_emb, dim=-1).mean()
        category_emb_max_val = torch.max(category_src_emb)

        epoch = float(self.trainer.current_epoch)
        self.log(f"epoch/{phase}", epoch)
        self.log(f"src_fbt/avg", src_fbt_std_mean)
        self.log(f"src_fbt/std", src_fbt_std)
        self.log(f"src_fbt/avg_norm", src_fbt_avg_norm)
        self.log(f"category_emb/avg", category_emb_mean)
        self.log(f"category_emb/avg_norm", category_emb_avg_norm)
        self.log(f"category_emb/max_val", category_emb_max_val)
        return {
            "loss": loss,
            "asin_src": asin_src,
            "asin_pos": asin_pos,
            "src": src.detach().cpu(),
            "src_fbt": src_fbt.detach().cpu(),
            "pos": pos.detach().cpu(),
            "category_src": category_src.detach().cpu(),
            "category_pos": category_pos.detach().cpu(),
            "set_name": set_name,
        }

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.cfg.milestones
        )
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx, optimizer_idx: int = 0):
        return self._loss_helper(
            batch, phase="train", batch_idx=batch_idx, optimizer_idx=optimizer_idx
        )

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self._loss_helper(
            batch, phase="test", batch_idx=batch_idx, optimizer_idx=0
        )

    def validation_epoch_end(self, outputs, phase: str = "test"):
        self._calc_retrival_metrics(outputs[-1], phase)

    def _calc_retrival_metrics(self, outputs, phase):
        t1 = time()

        # Get values from step
        epoch = int(self.trainer.current_epoch)
        src = torch.vstack([out["src"] for out in outputs])
        src_fbt = torch.vstack([out["src_fbt"] for out in outputs])
        pos = torch.vstack([out["pos"] for out in outputs])
        asin_src = np.hstack([out["asin_src"] for out in outputs])
        asin_pos = np.hstack([out["asin_pos"] for out in outputs])
        category_src = torch.hstack([out["category_src"] for out in outputs]).numpy()
        category_pos = torch.hstack([out["category_pos"] for out in outputs]).numpy()
        set_name = np.hstack([out["set_name"] for out in outputs])

        # Test sources: have a unique pair of (source,target-category)
        src_fbt_test = src_fbt[set_name == "test"]
        asin_src_test = asin_src[set_name == "test"]
        category_pos_test = category_pos[set_name == "test"]

        locs = list(zip(asin_src_test, category_pos_test))
        _, unique_idxs = np.unique(np.array(locs), axis=0, return_index=True)
        src_fbt_test, asin_src_test, category_pos_test = (
            src_fbt_test[unique_idxs],
            asin_src_test[unique_idxs],
            category_pos_test[unique_idxs],
        )

        # Candidate to compare with
        asins = np.hstack([asin_src, asin_pos])
        embs = torch.vstack([src, pos])
        categories = np.hstack([category_src, category_pos])
        asins, embs, categories = get_unique_asins(asins, embs, categories)

        # Build hot label
        t1 = time()
        fbt_by_asin_src = self.trainer.val_dataloaders[
            -1
        ].dataset.fbt_by_asin_src.copy()
        hot_labels, _ = generate_test_set_hot_labels(
            asin_src_test=asin_src_test,
            category_pos_test=category_pos_test,
            fbt_by_asin_src=fbt_by_asin_src,
            asins=asins,
        )
        logger.info(f"hot_labels in {time()-t1:.1f} s. {hot_labels.shape=}")

        # Find distance of the candidates
        t2 = time()
        dists = torch.cdist(src_fbt_test, embs, p=2)
        probs = torch.softmax(-dists, axis=-1)

        # Constrain to target cateogry
        for n, cat in enumerate(category_pos_test):
            probs[n, categories != cat] = 0

        # Calculate retrival metrics
        ndcg_val = ndcg_score(hot_labels, probs)
        self.logger.log_metrics({"retrieval_metrics/ndcg": ndcg_val}, step=epoch)
        logger.info(f"    {epoch=} {ndcg_val=:.6f} {probs.shape=}. {time()-t2:.1f}s")

        # TopK
        t2 = time()
        topk_d = calc_topk(probs, hot_labels, self.cfg.top_k)
        self.logger.log_metrics(topk_d, step=epoch)
        logger.info(f"{epoch=} _epoch_end_helper. {topk_d}. in {time()-t2:.1f} s")

        # Save tensors
        if self.ndcg_val_best < ndcg_val:
            self.ndcg_val_best = ndcg_val
            save_run_results(
                {
                    "src": src,
                    "src_fbt": src_fbt,
                    "pos": pos,
                    "asin_src": asin_src,
                    "asin_pos": asin_pos,
                    "category_src": category_src,
                    "category_pos": category_pos,
                    "set_name": set_name,
                },
                self.out_dir,
            )
            self.trainer.save_checkpoint(osj(self.out_dir, "checkpoint.ckpt"))
            logger.info(f"    {epoch=} {self.ndcg_val_best=:.5f}")
        logger.info(f"_epoch_end_helper. {time()-t1:.1f} s")
