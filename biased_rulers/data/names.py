import re
import pandas as pd


def load_data():
    "Load gendered names from Lauscher et al. (2021) used by DisCo."

    df = pd.read_csv("biased_rulers/data/name_pairs.txt", sep="\t", header=None)
    return df