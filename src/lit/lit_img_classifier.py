# Copyright (c) 2015-present, Meta Platforms, Inc. and affiliates.
# All rights reserved.
import logging

import pytorch_lightning as pl
import torch
from torch import nn, optim

logger = logging.getLogger(__name__)


class LitImgClassifier(pl.LightningModule):
    def __init__(self, input_dim, num_classes, cfg):
        super().__init__()
        self.cfg = cfg

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 16),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.1),
            nn.Linear(16, num_classes),
        )
        self.criterion_category = nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx, optimizer_idx):
        return self._loss_helper(batch, "train", optimizer_idx)

    def validation_step(self, batch, batch_idx: int):
        return self._loss_helper(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._loss_helper(batch, "test")

    def _loss_helper(self, batch, phase: str = "train"):
        emb, category = batch
        category_hat = self.classifier(emb)

        # Performance
        category_int = torch.argmax(category, dim=-1)
        category_int_pred = torch.argmax(category_hat, dim=-1)
        acc = (category_int_pred == category_int).float().mean()
        loss = self.criterion_category(category_hat, category.float())

        # Log
        self.log(f"loss/{phase}", loss)
        self.log(f"acc/{phase}", acc)
        self.log(f"epoch/{phase}", float(self.trainer.current_epoch))

        return {"loss": loss}

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )

        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.cfg.milestones
        )
        return [optimizer], [lr_scheduler]
