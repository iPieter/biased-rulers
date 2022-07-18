from transformers import BertTokenizer, BertModel
import torch
import pandas as pd
import numpy as np
import math


class Dataset:
    def __init__(self, male_list, female_list) -> None:
        self.male_list = male_list
        self.female_list = female_list


def load_dataset():
    df_professions = pd.read_csv("biased_rulers/data/professionsBLS2015.csv")

    df_professions2 = df_professions
    del df_professions2["label2"]
    del df_professions2["label3"]
    del df_professions2["label4"]
    del df_professions2["label5"]
    del df_professions2["none"]

    df_professions3 = df_professions2.groupby("label1", as_index=False).mean()

    ## use percentages
    # female_professions = df_professions3.loc[df_professions3['Women'] > 70]
    # male_professions = df_professions3.loc[df_professions3['Women'] < 30]

    # use top-k
    df_professions3 = df_professions3.sort_values("Women")
    female_professions = df_professions3[-30:]
    male_professions = df_professions3[:30]

    female_list_1 = female_professions.sample(frac=1, random_state=1)["label1"][
        :10
    ].tolist()
    male_list_1 = male_professions.sample(frac=1, random_state=1)["label1"][
        :10
    ].tolist()
    female_list_2 = female_professions.sample(frac=1, random_state=2)["label1"][
        :10
    ].tolist()
    male_list_2 = male_professions.sample(frac=1, random_state=2)["label1"][
        :10
    ].tolist()
    female_list_3 = female_professions.sample(frac=1, random_state=3)["label1"][
        :10
    ].tolist()
    male_list_3 = male_professions.sample(frac=1, random_state=3)["label1"][
        :10
    ].tolist()
    female_list_4 = female_professions.sample(frac=1, random_state=4)["label1"][
        :10
    ].tolist()
    male_list_4 = male_professions.sample(frac=1, random_state=4)["label1"][
        :10
    ].tolist()
    female_list_5 = female_professions.sample(frac=1, random_state=5)["label1"][
        :10
    ].tolist()
    male_list_5 = male_professions.sample(frac=1, random_state=5)["label1"][
        :10
    ].tolist()
    female_list_6 = female_professions.sample(frac=1, random_state=6)["label1"][
        :10
    ].tolist()
    male_list_6 = male_professions.sample(frac=1, random_state=6)["label1"][
        :10
    ].tolist()
    female_list_7 = female_professions.sample(frac=1, random_state=7)["label1"][
        :10
    ].tolist()
    male_list_7 = male_professions.sample(frac=1, random_state=7)["label1"][
        :10
    ].tolist()
    female_list_8 = female_professions.sample(frac=1, random_state=8)["label1"][
        :10
    ].tolist()
    male_list_8 = male_professions.sample(frac=1, random_state=8)["label1"][
        :10
    ].tolist()
    female_list_9 = female_professions.sample(frac=1, random_state=9)["label1"][
        :10
    ].tolist()
    male_list_9 = male_professions.sample(frac=1, random_state=9)["label1"][
        :10
    ].tolist()
    female_list_10 = female_professions.sample(frac=1, random_state=10)["label1"][
        :10
    ].tolist()
    male_list_10 = male_professions.sample(frac=1, random_state=10)["label1"][
        :10
    ].tolist()
    female_list_11 = female_professions.sample(frac=1, random_state=11)["label1"][
        :10
    ].tolist()
    male_list_11 = male_professions.sample(frac=1, random_state=11)["label1"][
        :10
    ].tolist()
    female_list_12 = female_professions.sample(frac=1, random_state=12)["label1"][
        :10
    ].tolist()
    male_list_12 = male_professions.sample(frac=1, random_state=12)["label1"][
        :10
    ].tolist()
    female_list_13 = female_professions.sample(frac=1, random_state=13)["label1"][
        :10
    ].tolist()
    male_list_13 = male_professions.sample(frac=1, random_state=13)["label1"][
        :10
    ].tolist()
    female_list_14 = female_professions.sample(frac=1, random_state=14)["label1"][
        :10
    ].tolist()
    male_list_14 = male_professions.sample(frac=1, random_state=14)["label1"][
        :10
    ].tolist()
    female_list_15 = female_professions.sample(frac=1, random_state=15)["label1"][
        :10
    ].tolist()
    male_list_15 = male_professions.sample(frac=1, random_state=15)["label1"][
        :10
    ].tolist()
    female_list_16 = female_professions.sample(frac=1, random_state=16)["label1"][
        :10
    ].tolist()
    male_list_16 = male_professions.sample(frac=1, random_state=16)["label1"][
        :10
    ].tolist()
    female_list_17 = female_professions.sample(frac=1, random_state=17)["label1"][
        :10
    ].tolist()
    male_list_17 = male_professions.sample(frac=1, random_state=17)["label1"][
        :10
    ].tolist()
    female_list_18 = female_professions.sample(frac=1, random_state=18)["label1"][
        :10
    ].tolist()
    male_list_18 = male_professions.sample(frac=1, random_state=18)["label1"][
        :10
    ].tolist()
    female_list_19 = female_professions.sample(frac=1, random_state=19)["label1"][
        :10
    ].tolist()
    male_list_19 = male_professions.sample(frac=1, random_state=19)["label1"][
        :10
    ].tolist()
    female_list_20 = female_professions.sample(frac=1, random_state=20)["label1"][
        :10
    ].tolist()
    male_list_20 = male_professions.sample(frac=1, random_state=20)["label1"][
        :10
    ].tolist()

    female_list = [
        female_list_1,
        female_list_2,
        female_list_3,
        female_list_4,
        female_list_5,
        female_list_6,
        female_list_7,
        female_list_8,
        female_list_9,
        female_list_10,
        female_list_11,
        female_list_12,
        female_list_13,
        female_list_14,
        female_list_15,
        female_list_16,
        female_list_17,
        female_list_18,
        female_list_19,
        female_list_20,
    ]
    male_list = [
        male_list_1,
        male_list_2,
        male_list_3,
        male_list_4,
        male_list_5,
        male_list_6,
        male_list_7,
        male_list_8,
        male_list_9,
        male_list_10,
        male_list_11,
        male_list_12,
        male_list_13,
        male_list_14,
        male_list_15,
        male_list_16,
        male_list_17,
        male_list_18,
        male_list_19,
        male_list_20,
    ]

    return Dataset(male_list, female_list)
