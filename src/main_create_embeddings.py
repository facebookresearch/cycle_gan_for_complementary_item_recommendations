# Copyright (c) 2015-present, Meta Platforms, Inc. and affiliates.
# All rights reserved.
import logging
import os
import os.path as osp
import time
import types
import hydra
import pandas as pd
import torch
import torchvision.models as models
from omegaconf import DictConfig
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset_utils import get_image_dataset
import torch.nn.functional as F
logger = logging.getLogger(__name__)


def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        features = torch.flatten(x, 1)
        x = self.fc(features)

        return x, features

@hydra.main(
    version_base="1.2", config_path="../configs/", config_name="create_embeddings"
)
def create_embeddings(cfg: DictConfig):
    t0 = time.time()
    logger.info(cfg)
    os.chdir(hydra.utils.get_original_cwd())
    pkl_path = osp.join(cfg.data_processed_dir, f"{cfg.category_name}_processed_w_imgs.pkl")
    out_path = osp.join(cfg.data_final_dir, f"{cfg.category_name}_embeddings.pkl")

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Load data
    df = pd.read_pickle(pkl_path)
    df = df[df["img_path"].apply(lambda path: osp.exists(path))]

    dataset = get_image_dataset(df)
    dataloader = DataLoader(
        dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=True
    )
    logger.info(f"Creating dataset: {len(dataset)=}")

    # Load pretrained model
    model = models.resnet152(pretrained=True)
    model.forward = types.MethodType(_forward_impl, model)
    model = model.to(device)

    embeddings = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            _, embedding = model(batch.to(device))
            embeddings.append(embedding.to("cpu"))
    embeddings = torch.vstack(embeddings)

    # Save to file
    df["img_embedding"] = embeddings.tolist()
    df = df[["asin", "img_path", "img_embedding"]]
    df.to_pickle(out_path)
    logger.info(f"Finish in {time.time()-t0:.1f} sec. {out_path=}")


if __name__ == "__main__":
    create_embeddings()
