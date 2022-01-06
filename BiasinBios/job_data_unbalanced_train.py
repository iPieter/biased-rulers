import tensorflow as tf
import numpy as np
import torch
import random
import pandas as pd
import os, sys
import time
import datetime
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
# import job_data


df = pd.read_csv("preprocessed.csv")
df_job = pd.read_csv("occupation.csv", header=None)



for i in range(df["label"].nunique()):
    df.loc[df['label'] == i, 'prof'] = df_job.iloc[i][1]

gen_dist = {}
for i in range(df["label"].nunique()):
    gen_frac = (len(df[(df["label"]==i) & (df["gender"]=="F")]))/(len(df[df["label"]==i]))
    gen_dist[df_job.iloc[i][1]] = gen_frac
gen_dist

gen_dist = {}
for i in range(df["label"].nunique()):
    gen_frac = (len(df[(df["label"]==i) & (df["gender"]=="F")]))/(len(df[df["label"]==i]))
    gen_dist[df_job.iloc[i][1]] = gen_frac
gen_dist

gen_dist = dict(sorted(gen_dist.items(), key=lambda item: item[1]))
gen_dist

male_jobs = ["architect", "surgeon", "software_engineer", "pastor"]
female_jobs = ["nurse","model", "dietitian", "paralegal"]


bios_df = df[["raw_title", "gender", "label", "scrubbed", "bio", "swapped", "prof"]]

c0 = bios_df[bios_df['label'] == 1]
c1 = bios_df[bios_df['label'] == 25]
c2 = bios_df[bios_df['label'] == 24]
c3 = bios_df[bios_df['label'] == 16]
c4 = bios_df[bios_df['label'] == 13]
c5 = bios_df[bios_df['label'] == 12]
c6 = bios_df[bios_df['label'] == 7]
c7 = bios_df[bios_df['label'] == 15]

bios_df = df[["raw_title", "gender", "label", "scrubbed", "bio", "swapped", "prof"]]

from sklearn.utils import shuffle
final_bios_df = pd.concat([c0, c1, c2, c3, c4, c5, c6, c7])
final_bios_df = shuffle(final_bios_df, random_state=42)
