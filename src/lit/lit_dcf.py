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

from .eval_utils import (
    calc_topk,
    generate_test_set_hot_labels,
    get_unique_asins,
    save_run_results,
)
from .lit_utils import LearnedEmbs

logger = logging.getLogger(__name__)


class ImageEncoder(nn.Module):
    def __init__(
        self,
        input_emb_dim: int,
        input_category_dim: int,
        input_price_dim: int,
        output_dim: int,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.input_emb_dim = input_emb_dim
        self.input_category_dim = input_category_dim
        self.input_price_dim = input_price_dim
        self.input_dim = input_emb_dim + input_category_dim + input_price_dim
        self.layers = nn.Sequential(
            nn.Linear(in_features=self.input_dim, out_features=64),
            nn.Dropout(p=dropout_rate),
            nn.LeakyReLU(),
            nn.Linear(in_features=64, out_features=output_dim),
            nn.LeakyReLU(),
        )

    def forward(self, img_embs, category_embs, price_embs):
        return self.layers(torch.hstack([img_embs, category_embs, price_embs]))


class FusionModel(nn.Module):
    def __init__(
        self,
        input_emb_dim: int,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.input_emb_dim = input_emb_dim
        self.layers = nn.Sequential(
            nn.Linear(in_features=self.input_emb_dim * 2, out_features=16),
            nn.Dropout(p=dropout_rate),
            nn.LeakyReLU(),
            nn.Linear(16, out_features=1),
        )

    def forward(self, src_embs, candidate_embs):
        embs = torch.hstack([src_embs, candidate_embs])
        return self.layers(embs)


class LitDCF(pl.LightningModule):
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
        self.src_encoder = ImageEncoder(
            input_emb_dim, category_emb_size, price_emb_size, emb_dim, 0.1
        )
        self.candidate_encoder = ImageEncoder(
            input_emb_dim, category_emb_size, price_emb_size, emb_dim, 0.1
        )
        self.fusion_model = FusionModel(emb_dim, 0.1)  # cfg.dropout_rate
        self.criterion = nn.BCEWithLogitsLoss()

        # Performance
        self.ndcg_val_best = 0.0
        wandb.define_metric("retrieval_metrics/ndcg", summary="max")

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

        img_emb_neg_easy = torch.roll(img_emb_src, 1)
        category_neg_easy = torch.roll(category_src, 1)
        price_bin_neg_easy = torch.roll(price_bin_src, 1)

        category_src_emb = self.category_embs(category_src)
        category_pos_emb = self.category_embs(category_pos)
        category_neg_emb = self.category_embs(category_pos)
        category_neg_emb_easy = self.category_embs(category_neg_easy)

        price_src_emb = self.price_embs(price_bin_src)
        price_pos_emb = self.price_embs(price_bin_pos)
        price_neg_emb = self.price_embs(price_bin_neg)
        price_neg_emb_easy = self.price_embs(price_bin_neg_easy)

        src = self.src_encoder(img_emb_src, category_src_emb, price_src_emb)
        pos = self.candidate_encoder(img_emb_pos, category_pos_emb, price_pos_emb)
        neg = self.candidate_encoder(img_emb_neg, category_neg_emb, price_neg_emb)
        neg_easy = self.candidate_encoder(
            img_emb_neg_easy, category_neg_emb_easy, price_neg_emb_easy
        )

        pred_pos = self.fusion_model(src, pos)
        pred_neg = self.fusion_model(src, neg)
        pred_neg_easy = self.fusion_model(src, neg_easy)

        target_pos = torch.ones_like(pred_pos)
        loss_pos = self.criterion(pred_pos.squeeze(), target_pos.squeeze())
        target_neg = torch.zeros_like(pred_neg)
        loss_neg = self.criterion(pred_neg.squeeze(), target_neg.squeeze())
        target_neg_easy = torch.zeros_like(pred_neg_easy)
        loss_neg_easy = self.criterion(
            pred_neg_easy.squeeze(), target_neg_easy.squeeze()
        )
        pred_pos = torch.sigmoid(pred_pos)
        pred_neg = torch.sigmoid(pred_neg)

        # Loss
        if self.cfg.hard_negative:
            loss = 0.5 * (loss_pos + loss_neg)
        else:
            loss = 0.5 * (loss_pos + loss_neg_easy)

        # Logger
        self.log(f"loss/{phase}", loss)
        self.log(f"clf/{phase}/acc/pred_pos", pred_pos.mean())
        self.log(f"clf/{phase}/acc/pred_neg", pred_neg.mean())
        self.log(f"clf/{phase}/loss/loss_pos", loss_pos)
        self.log(f"clf/{phase}/loss/loss_neg", loss_neg)

        # Logger
        category_emb_mean = category_src_emb.mean(axis=-1).mean()
        category_emb_avg_norm = torch.norm(category_src_emb, dim=-1).mean()
        category_emb_max_val = torch.max(category_src_emb)

        epoch = float(self.trainer.current_epoch)
        self.log(f"epoch/{phase}", epoch)
        self.log(f"category_emb/avg", category_emb_mean)
        self.log(f"category_emb/avg_norm", category_emb_avg_norm)
        self.log(f"category_emb/max_val", category_emb_max_val)
        return {
            "loss": loss,
            "asin_src": asin_src,
            "asin_pos": asin_pos,
            "src": src.detach().cpu(),
            "pos": pos.detach().cpu(),
            "category_src": category_src.detach().cpu(),
            "category_pos": category_pos.detach().cpu(),
            "category_neg": category_pos.detach().cpu(),
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
        pos = torch.vstack([out["pos"] for out in outputs])
        asin_src = np.hstack([out["asin_src"] for out in outputs])
        asin_pos = np.hstack([out["asin_pos"] for out in outputs])
        category_src = torch.hstack([out["category_src"] for out in outputs]).numpy()
        category_pos = torch.hstack([out["category_pos"] for out in outputs]).numpy()
        set_name = np.hstack([out["set_name"] for out in outputs])

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
        asins = asin_pos
        candidates = pos
        categories = category_pos
        asins, candidates, categories = get_unique_asins(asins, candidates, categories)

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

        # Find distance of the candidates: infernece of the source with each candidate. This is the row of dists
        t2 = time()
        probs = torch.vstack(
            [
                torch.sigmoid(
                    self.fusion_model(
                        (src_i.repeat(len(candidates), 1)), candidates
                    ).squeeze()
                )
                for src_i in src_test
            ]
        )

        # Constrain to target cateogry
        for n, cat in enumerate(category_pos_test):
            probs[n, categories != cat] = 0
        probs = probs / probs.sum(axis=1, keepdim=True)

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
