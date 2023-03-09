# Copyright (c) 2015-present, Meta Platforms, Inc. and affiliates.
# All rights reserved.
import logging
import time
from os.path import join as osj
from time import time
from itertools import chain

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

logger = logging.getLogger(__name__)


class LearnedEmbs(nn.Module):
    def __init__(self, num_classes, emb_size: int = 16):
        super().__init__()
        self.num_classes, self.emb_size = num_classes, emb_size
        self.embs = nn.Embedding(self.num_classes, self.emb_size, max_norm=1.0)

    def forward(self, idx):
        return self.embs(idx)


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
            nn.Linear(in_features=self.input_dim, out_features=256),
            nn.BatchNorm1d(256),
            nn.Dropout(p=dropout_rate),
            nn.LeakyReLU(),
            nn.Linear(in_features=256, out_features=64),
            nn.BatchNorm1d(64),
            nn.Dropout(p=dropout_rate),
            nn.LeakyReLU(),
            nn.Linear(in_features=64, out_features=output_dim),
        )

    def forward(self, img_embs, category_embs, price_embs):
        return self.layers(torch.hstack([img_embs, category_embs, price_embs]))


class CategoryClassifier(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        num_categories: int,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(emb_dim, 8),
            nn.Dropout(p=dropout_rate),
            nn.LeakyReLU(),
            nn.Linear(in_features=8, out_features=num_categories),
        )

    def forward(self, x):
        return self.layers(x)


class FbtAutoEncoder(nn.Module):
    def __init__(
        self,
        input_emb_dim: int,
        input_category_dim: int,
        dropout_rate: float = 0.1,
    ):
        # Transfomratoion: source embs + dst category -> dst embs
        super().__init__()
        self.input_emb_dim = input_emb_dim
        self.input_category_dim = input_category_dim
        self.input_dim = input_emb_dim + input_category_dim

        self.layers = nn.Sequential(
            nn.Linear(in_features=self.input_dim, out_features=8),
            nn.BatchNorm1d(8),
            nn.Dropout(p=dropout_rate),
            nn.LeakyReLU(),
            nn.Linear(in_features=8, out_features=self.input_emb_dim),
        )

    def forward(self, src_embs, dst_category):
        return self.layers(torch.hstack([src_embs, dst_category]))


class LitFbt(pl.LightningModule):
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
        self.fbt_ae = FbtAutoEncoder(emb_dim, cfg.category_emb_size, cfg.dropout_rate)
        self.fbt_ae_return = FbtAutoEncoder(
            emb_dim, cfg.category_emb_size, cfg.dropout_rate
        )
        self.clf = CategoryClassifier(emb_dim, num_categories, cfg.dropout_rate)
        logger.info(self.category_embs)
        logger.info(self.price_embs)
        logger.info(self.img_encoder)
        logger.info(self.fbt_ae)
        logger.info(self.clf)

        # Losses
        self.criterion_triplet = nn.TripletMarginLoss(
            margin=cfg.triplet_loss_margin, p=2
        )
        self.criterion_category = nn.CrossEntropyLoss()
        self.criterion_cycle = nn.MSELoss()

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

    def calc_triplet_acc(self, src_fbt, pos, neg):
        pos_dist = torch.norm(src_fbt - pos, dim=-1)
        neg_dist = torch.norm(src_fbt - neg, dim=-1)
        return (pos_dist < neg_dist).float().mean().item()

    def _clf_helper(self, emb, category):
        category_hat = self.clf(emb)
        loss = self.criterion_category(category_hat, category)

        category_pred = torch.argmax(category_hat, dim=-1)
        acc = (category_pred == category).float().mean()
        return loss, acc

    def _cycle_helper(self, src, category_src_emb, category_dst):
        category_dst_emb = self.category_embs(category_dst)
        src_fbt = self.fbt_ae(src, category_dst_emb)
        cycle = self.fbt_ae_return(src_fbt, category_src_emb)
        loss_cycle = self.criterion_cycle(cycle, src.detach())
        return loss_cycle

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

        if self.cfg.is_autoecoder_detach is True:
            src_fbt = self.fbt_ae(src.detach(), category_dst_emb)
        else:
            src_fbt = self.fbt_ae(src, category_dst_emb)

        # Train generator
        if optimizer_idx == 0:

            # Classifier
            loss_clf_src, acc_clf_src = self._clf_helper(src, category_src)
            loss_clf_pos, acc_clf_pos = self._clf_helper(pos, category_pos)
            loss_clf_src_fbt, acc_clf_src_fbt = self._clf_helper(src_fbt, category_pos)
            loss_clf = (1 / 3) * (loss_clf_src + loss_clf_src_fbt + loss_clf_pos)

            # Triplet
            loss_triplet = self.criterion_triplet(src_fbt, pos, neg)
            acc_triplet = self.calc_triplet_acc(src_fbt, pos, neg)

            # Cycle
            loss_cycle = self._cycle_helper(
                src, category_src_emb, random_valid_category
            )
            loss_cycle_labeled_pairs = self._cycle_helper(
                src, category_src_emb, category_pos
            )

            # Loss
            loss = (1 / 3) * (
                self.cfg.triplet_weight * loss_triplet
                + self.cfg.cycle_weight * loss_cycle
                + self.cfg.cycle_weight_labeled_pairs * loss_cycle_labeled_pairs
                + self.cfg.clf_weight * loss_clf
            )
            acc_genuine = (1 / 2) * (acc_clf_src + acc_clf_pos)

            # Logger
            self.log(f"loss/{phase}", loss)
            self.log(f"clf/{phase}/acc/acc_genuine", acc_genuine)
            self.log(f"clf/{phase}/acc/acc_clf_src_fbt", acc_clf_src_fbt)
            self.log(f"clf/{phase}/loss/loss_clf", loss_clf)
            self.log(f"triplet/{phase}/acc_triplet", acc_triplet)
            self.log(f"triplet/{phase}/loss_triplet", loss_triplet)
            self.log(f"cycle/{phase}/loss_cycle", loss_cycle)

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

        # Train discriminator
        if optimizer_idx == 1 and self.cfg.discriminator_weight > 0.0:

            loss_clf_src, acc_clf_src = self._clf_helper(src.detach(), category_src)
            loss_clf_src_fbt, acc_clf_src_fbt = self._clf_helper(
                src_fbt.detach(), category_pos
            )
            loss_clf_src_fbt = -loss_clf_src_fbt
            loss = (
                self.cfg.discriminator_weight * 0.5 * (loss_clf_src + loss_clf_src_fbt)
            )
            self.log(f"loss/{phase}/optimizer_idx_1/loss_clf_src", loss_clf_src)
            self.log(f"loss/{phase}/optimizer_idx_1/loss_clf_src_fbt", loss_clf_src_fbt)
            return loss

    def configure_optimizers_w_discriminator(self):
        optimizer = optim.Adam(
            chain(
                self.category_embs.parameters(),
                self.price_embs.parameters(),
                self.img_encoder.parameters(),
                self.fbt_ae.parameters(),
                self.fbt_ae_return.parameters(),
            ),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )
        optimizer_discriminator = optim.Adam(
            self.clf.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.cfg.milestones
        )
        lr_scheduler_discriminator = torch.optim.lr_scheduler.MultiStepLR(
            optimizer_discriminator, milestones=self.cfg.milestones
        )

        return [optimizer, optimizer_discriminator], [
            lr_scheduler,
            lr_scheduler_discriminator,
        ]

    def configure_optimizers_vanilla(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.cfg.milestones
        )
        return [optimizer], [lr_scheduler]

    def configure_optimizers(self):
        return (
            self.configure_optimizers_vanilla()
            if self.cfg.discriminator_weight > 0.0
            else self.configure_optimizers_w_discriminator()
        )

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
