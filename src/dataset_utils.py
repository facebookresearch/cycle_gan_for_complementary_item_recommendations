# Copyright (c) 2015-present, Meta Platforms, Inc. and affiliates.
# All rights reserved.
import logging
import os.path as osp
import random

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)


def load_dfs(cfg):
    pkl_path = osp.join(cfg["data_dir"], cfg["category_name"] + "_sets.pkl")
    fbt_df = pd.read_pickle(pkl_path)
    emb_path = osp.join(cfg["data_dir"], cfg["category_name"] + "_embeddings.pkl")
    emb_df = pd.read_pickle(emb_path)
    return fbt_df, emb_df


class ImgDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        x = Image.open(image_path)
        if self.transform is not None:
            x = self.transform(x)
        return x

    def __len__(self):
        return len(self.image_paths)


def repeate_to_rgb(x):
    if x.size(0) < 3:
        x = x.repeat(3, 1, 1)
    return x


def get_image_dataset(df):
    return ImgDataset(
        df.img_path.tolist(),
        transform=transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Lambda(repeate_to_rgb),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
    )


class FbtDataset(Dataset):
    def __init__(self, fbt_df: pd.DataFrame, df_embs: pd.DataFrame):

        # Database
        self.fbt_df = fbt_df
        self.fbt_df_price = (
            fbt_df[["asin_target", "price_bin_target"]]
            .drop_duplicates()
            .set_index("asin_target")
        )
        self.df_embs = df_embs.set_index("asin")

        # Map cateogry to asing: Needed for negative from the same target cateogry
        self.asin_per_categroy = fbt_df.groupby("category_int_target")[
            "asin_target"
        ].apply(list)

        # Valid categories
        self.valid_categories = self.fbt_df.groupby("category_int_src")[
            "category_int_target"
        ].apply(list)

        # For evaluation
        self.fbt_by_asin_src = fbt_df.groupby(["asin_src", "category_int_target"])[
            "asin_target"
        ].apply(list)

    def __getitem__(self, index):
        asin_src = self.fbt_df.iloc[index]["asin_src"]
        asin_pos = self.fbt_df.iloc[index]["asin_target"]
        category_src = self.fbt_df.iloc[index]["category_int_src"]
        category_pos = self.fbt_df.iloc[index]["category_int_target"]
        emb_src = torch.tensor(self.df_embs.loc[asin_src].img_embedding)
        emb_pos = torch.tensor(self.df_embs.loc[asin_pos].img_embedding)
        set_name = self.fbt_df.iloc[index]["set_name"]

        # Negative sample
        asin_neg = asin_pos
        while asin_pos == asin_neg:
            asins_in_category = self.asin_per_categroy.loc[category_pos]
            asin_neg = random.choice(asins_in_category)
        emb_neg = torch.tensor(self.df_embs.loc[asin_neg].img_embedding)

        # Price
        price_bin_src = self.fbt_df.iloc[index]["price_bin_src"]
        price_bin_pos = self.fbt_df.iloc[index]["price_bin_target"]
        fbt_df_price = self.fbt_df_price.copy()  # thread safe
        price_bin_neg = fbt_df_price.loc[asin_neg]["price_bin_target"]

        # Valid category:
        random_valid_category = random.sample(
            self.valid_categories.loc[category_src], 1
        )[0]

        return (
            emb_src,
            emb_pos,
            emb_neg,
            price_bin_src,
            price_bin_pos,
            price_bin_neg,
            category_src,
            category_pos,
            random_valid_category,
            asin_src,
            asin_pos,
            set_name,
        )

    def __len__(self):
        return len(self.fbt_df)


class FbtCandidateDataset(Dataset):
    def __init__(self, fbt_df: pd.DataFrame, df_embs: pd.DataFrame):

        self.fbt_df = fbt_df
        self.df_price = (
            fbt_df[["asin_target", "price_bin_target"]]
            .drop_duplicates()
            .set_index("asin_target")
        )
        self.df_embs = df_embs.set_index("asin")

        self.fbt_candidates = (
            fbt_df[["asin_target", "category_int_target"]]
            .drop_duplicates()
            .reset_index()
        )

    def __getitem__(self, index):
        asin_src = self.fbt_candidates.iloc[index]["asin_target"]
        category_src = self.fbt_candidates.iloc[index]["category_int_target"]
        emb_src = torch.tensor(self.df_embs.loc[asin_src].img_embedding)
        price_bin_src = self.df_price.loc[asin_src]["price_bin_target"]

        return (
            emb_src,
            price_bin_src,
            category_src,
            asin_src,
        )

    def __len__(self):
        return len(self.fbt_candidates)


class FbtInferenceDataset(Dataset):
    def __init__(self, fbt_df: pd.DataFrame, df_embs: pd.DataFrame):

        self.fbt_df_test = fbt_df[fbt_df["set_name"] == "test"]
        self.fbt_df_test = (
            self.fbt_df_test.groupby(["asin_src", "category_int_src"])["asin_target"]
            .agg(list)
            .reset_index()
        )

        # Valid categories to transfrom to
        fbt_df_train = fbt_df[fbt_df["set_name"] == "train"]
        self.valid_categories = fbt_df_train.groupby(["category_int_src"])[
            "category_int_target"
        ].agg(list)

        self.df_price = (
            fbt_df[["asin_src", "price_bin_src"]]
            .drop_duplicates()
            .set_index("asin_src")
        )
        self.df_embs = df_embs.set_index("asin")

    def __getitem__(self, index):
        asin_src = self.fbt_df_test.iloc[index]["asin_src"]
        category_src = self.fbt_df_test.iloc[index]["category_int_src"]
        emb_src = torch.tensor(self.df_embs.loc[asin_src].img_embedding)
        price_bin_src = self.df_price.loc[asin_src]["price_bin_src"]

        # Labels
        asin_targets = self.fbt_df_test.iloc[index]["asin_target"]

        # Categories to transform to
        valid_categories = self.valid_categories.loc[category_src]
        return (
            emb_src,
            price_bin_src,
            category_src,
            asin_src,
            valid_categories,
            asin_targets,
        )

    def __len__(self):
        return len(self.fbt_df_test)
