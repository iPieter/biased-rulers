import re
import pandas as pd


def load_data():
    "Load gendered nouns from Zhao et al. (2018) used by DisCo."

    df = pd.read_csv("biased_rulers/data/generalized_swaps.txt", sep="\t", header=None)
    return df