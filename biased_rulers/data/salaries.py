import re
import pandas as pd


def load_data():

    df = pd.read_csv("biased_rulers/data/employeesalaries2017.csv")
    df = df.sort_values("2017 Annual Salary")[-1000:]
    re_express = re.compile("\d")

    df["Job Title cleaned"] = df["Job Title"].str.lower().replace(re_express, "")

    return df
