import re
import pandas as pd


def load_data():
    "Load gendered nouns from Zhao et al. (2018) used by DisCo."

    df = pd.read_csv("biased_rulers/data/generalized_swaps.txt", sep="\t", header=None)
    return df

def load_extra_data():
    "Load the extra list of gendered nouns from Zhao et al. (2018)."

    df = pd.read_csv("biased_rulers/data/extra_gendered_words.txt", sep="\t", header=None)
    return df