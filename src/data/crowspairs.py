import csv
import numpy as np
import pandas as pd

from transformers import AutoTokenizer, AutoModelForMaskedLM
from collections import defaultdict
from tqdm import tqdm


def read_data():
    """
    Load data into pandas DataFrame format.
    """

    df_data = pd.DataFrame(columns=["sent1", "sent2", "direction", "bias_type"])

    with open("src/data/crows_pairs_anonymized.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            direction, gold_bias = "_", "_"
            direction = row["stereo_antistereo"]
            bias_type = row["bias_type"]

            sent1, sent2 = "", ""
            if direction == "stereo":
                sent1 = row["sent_more"]
                sent2 = row["sent_less"]
            else:
                sent1 = row["sent_less"]
                sent2 = row["sent_more"]

            df_item = {
                "sent1": sent1,
                "sent2": sent2,
                "direction": direction,
                "bias_type": bias_type,
            }
            df_data = df_data.append(df_item, ignore_index=True)

    return df_data
