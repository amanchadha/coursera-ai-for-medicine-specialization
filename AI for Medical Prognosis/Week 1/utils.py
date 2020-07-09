import numpy as np
import pandas as pd
import sklearn


def generate_data(n=200):
    df = pd.DataFrame(
        columns=['Age', 'Systolic_BP', 'Diastolic_BP', 'Cholesterol']
    )
    df.loc[:, 'Age'] = np.exp(np.log(60) + (1 / 7) * np.random.normal(size=n))
    df.loc[:, ['Systolic_BP', 'Diastolic_BP', 'Cholesterol']] = np.exp(
        np.random.multivariate_normal(
            mean=[np.log(100), np.log(90), np.log(100)],
            cov=(1 / 45) * np.array([
                [0.5, 0.2, 0.2],
                [0.2, 0.5, 0.2],
                [0.2, 0.2, 0.5]]),
            size=n))
    return df


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def f(x):
    p = 0.4 * (np.log(x[0]) - np.log(60)) + 0.33 * (
            np.log(x[1]) - np.log(100)) + 0.3 * (
                np.log(x[3]) - np.log(100)) - 2.0 * (
                np.log(x[0]) - np.log(60)) * (
                np.log(x[3]) - np.log(100)) + 0.05 * np.random.logistic()
    if p > 0.0:
        return 1.0
    else:
        return 0.0


def load_data(n=200):
    np.random.seed(0)
    df = generate_data(n)
    for i in range(len(df)):
        df.loc[i, 'y'] = f(df.loc[i, :])
        X = df.drop('y', axis=1)
        y = df.y
    return X, y
