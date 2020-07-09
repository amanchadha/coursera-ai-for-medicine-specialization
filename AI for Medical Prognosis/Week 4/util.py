import os

import pandas as pd

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__))
)

def load_data():
    df = pd.read_csv(os.path.join(__location__, 'pbc.csv'))
    df = df.drop('id', axis=1)
    df = df[df.status != 1]
    df.loc[:, 'status'] = df.status / 2.0
    df.loc[:, 'time'] = df.time / 365.0
    df.loc[:, 'trt'] = df.trt - 1
    df.loc[:, 'sex'] = df.sex.map({'f':0.0, 'm':1.0})
    df = df.dropna(axis=0)

    return df
