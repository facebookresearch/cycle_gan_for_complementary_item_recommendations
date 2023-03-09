# Cycle Generative Adversarial Networks for Complementary Item Recommendations

## Abstract
Complementary item recommendations are a ubiquitous feature of modern e-commerce sites. Such recommendations are highly effective when they are based on collaborative signals like co-purchase statistics. In certain online marketplaces, however, e.g., on online auction sites, constantly new items are added to the catalog. In such cases, complementary item recommendations are often based on item side-information due to a lack of interaction data. In this work, we propose a novel approach that can leverage both item side-information and labeled complementary item pairs to generate effective complementary recommendations for cold items, i.e., for items for which no co-purchase statistics yet exist. Given that complementary items typically have to be of a different category than the seed item, we technically maintain a latent space for each item category. Simultaneously, we learn to project distributed item representations into these category spaces to determine suitable recommendations. The main learning process in our architecture utilizes labeled pairs of complementary items. In addition, we adopt ideas from Cycle Generative Adversarial Networks (CycleGAN) to leverage available item information even in case no labeled data exists for a given item and category. Experiments on three e-commerce datasets show that our method is highly effective.

## Datasets

Datasets were downloaded from:
http://deepyeti.ucsd.edu/jianmo/amazon/index.html

Download the metadata file and place it under ./data/raw
```
├── LICENSE
├── README.md
├── configs
├── data
    ├── final
    ├── processed
    └── raw
        ├── meta_Clothing_Shoes_and_Jewelry.json.gz
        └── meta_Toys_and_Games.json.gz
        └── meta_Home_and_Kitchen.json.gz       
├── notebooks
├── outputs
├── requirements.txt
└── src
```

## Scripts to run experiments

### Clothing_Shoes_and_Jewelry
```
python main_process_meta.py category_name=Clothing_Shoes_and_Jewelry
python main_download_imgs.py category_name=Clothing_Shoes_and_Jewelry
python main_create_embeddings.py category_name=Clothing_Shoes_and_Jewelry
python main_create_train_test_set.py category_name=Clothing_Shoes_and_Jewelry
python main_execute_train.py category_name=Clothing_Shoes_and_Jewelry note="_triplet" discriminator_weight=0.0 clf_weight=0.0 cycle_weight=0.0
python main_execute_train.py category_name=Clothing_Shoes_and_Jewelry note="_triplet_and_cycle_" discriminator_weight=0.0 clf_weight=0.0 
python main_execute_train.py category_name=Clothing_Shoes_and_Jewelry note="_clf_and_cycle_" triplet_weight=0.0 discriminator_weight=0.0
python main_execute_train.py category_name=Clothing_Shoes_and_Jewelry note="_triplet_and_clf_"  discriminator_weight=0.0 cycle_weight=0.0
python main_execute_train.py category_name=Clothing_Shoes_and_Jewelry
python main_execute_train.py category_name=Clothing_Shoes_and_Jewelry method="dcf"
python main_execute_train.py category_name=Clothing_Shoes_and_Jewelry method="dcf" hard_negative=false
python main_execute_train.py category_name=Clothing_Shoes_and_Jewelry method="pcomp" 
```

### Toys_and_Games
```
python main_process_meta.py category_name=Toys_and_Games
python main_download_imgs.py category_name=Toys_and_Games
python main_create_embeddings.py category_name=Toys_and_Games
python main_create_train_test_set.py category_name=Toys_and_Games
python main_execute_train.py category_name=Toys_and_Games note="_triplet" discriminator_weight=0.0 clf_weight=0.0 cycle_weight=0.0 
python main_execute_train.py category_name=Toys_and_Games note="_triplet" discriminator_weight=0.0 clf_weight=0.0 cycle_weight=0.0
python main_execute_train.py category_name=Toys_and_Games note="_triplet_and_cycle_" discriminator_weight=0.0 clf_weight=0.0 
python main_execute_train.py category_name=Toys_and_Games note="_clf_and_cycle_" triplet_weight=0.0 discriminator_weight=0.0
python main_execute_train.py category_name=Toys_and_Games note="_triplet_and_clf_"  discriminator_weight=0.0 cycle_weight=0.0
python main_execute_train.py category_name=Toys_and_Games
python main_execute_train.py category_name=Toys_and_Games method="dcf"
python main_execute_train.py category_name=Toys_and_Games method="dcf" hard_negative=false
python main_execute_train.py category_name=Toys_and_Games method="pcomp" 
```

### Home_and_Kitchen
```
python main_process_meta.py category_name=Home_and_Kitchen
python main_download_imgs.py category_name=Home_and_Kitchen
python main_create_embeddings.py category_name=Home_and_Kitchen
python main_create_train_test_set.py category_name=Home_and_Kitchen
python main_execute_train.py category_name=Home_and_Kitchen note="_triplet" discriminator_weight=0.0 clf_weight=0.0 cycle_weight=0.0
python main_execute_train.py category_name=Home_and_Kitchen note="_triplet_and_cycle_" discriminator_weight=0.0 clf_weight=0.0 
python main_execute_train.py category_name=Home_and_Kitchen note="_clf_and_cycle_" triplet_weight=0.0 discriminator_weight=0.0
python main_execute_train.py category_name=Home_and_Kitchen note="_triplet_and_clf_"  discriminator_weight=0.0 cycle_weight=0.0
python main_execute_train.py category_name=Home_and_Kitchen
python main_execute_train.py category_name=Home_and_Kitchen method="dcf"
python main_execute_train.py category_name=Home_and_Kitchen method="dcf" hard_negative=false
python main_execute_train.py category_name=Home_and_Kitchen method="pcomp" 
```

# Contributing
See the CONTRIBUTING file for how to help out.

# License
This repository is CC BY-NC 4.0 licensed, as found in the LICENSE file.


