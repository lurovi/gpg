import random
import numpy as np
from sklearn.datasets import make_friedman1
from sklearn.preprocessing import StandardScaler, RobustScaler
import pandas as pd


class DatasetGenerator:
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def convert_text_dataset_to_csv(input_path: str, output_path: str, idx: int, scale: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
        file = open(input_path+'train'+str(idx), 'r')
        lines_train: list[str] = file.readlines()
        file.close()

        file = open(input_path+'test'+str(idx), 'r')
        lines_test: list[str] = file.readlines()
        file.close()

        n_features: int = int(lines_train[0].split()[0])
        
        lines_train = lines_train[2:]
        lines_test = lines_test[2:]

        d_train: dict[str, list[float]] = {str(k): [] for k in range(n_features)}
        d_train['target'] = []

        d_test: dict[str, list[float]] = {str(k): [] for k in range(n_features)}
        d_test['target'] = []

        for line in lines_train:
            l = line.split()
            for i in range(len(l)):
                if i == len(l) - 1:
                    d_train['target'].append(float(l[i]))
                else:
                    d_train[str(i)].append(float(l[i]))
        d_train: pd.DataFrame = pd.DataFrame(d_train)

        for line in lines_test:
            l = line.split()
            for i in range(len(l)):
                if i == len(l) - 1:
                    d_test['target'].append(float(l[i]))
                else:
                    d_test[str(i)].append(float(l[i]))
        d_test: pd.DataFrame = pd.DataFrame(d_test)

        if scale:
            y: np.ndarray = d_train['target'].to_numpy()
            d_train.drop('target', axis=1, inplace=True)
            X: np.ndarray = d_train.to_numpy()

            scaler: StandardScaler = StandardScaler()
            #scaler: RobustScaler = RobustScaler()
            scaler = scaler.fit(X)
            X = scaler.transform(X)
            d_train = pd.DataFrame(X)
            d_train.rename({i: str(i) for i in range(n_features)}, inplace=True)
            d_train["target"] = y

            y = d_test['target'].to_numpy()
            d_test.drop('target', axis=1, inplace=True)
            X = d_test.to_numpy()
            X = scaler.transform(X)
            d_test = pd.DataFrame(X)
            d_test.rename({i: str(i) for i in range(n_features)}, inplace=True)
            d_test["target"] = y

        d_train.to_csv(output_path+'train'+str(idx)+'.csv', index=False)
        d_test.to_csv(output_path+'test'+str(idx)+'.csv', index=False)
        return tuple[d_train, d_test]

    @staticmethod
    def read_csv_data(path: str, idx: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        d: pd.DataFrame = pd.read_csv(path+'train'+str(idx)+'.csv')
        y: np.ndarray = d['target'].to_numpy()
        d.drop('target', axis=1, inplace=True)
        X: np.ndarray = d.to_numpy()
        result: dict[str, tuple[np.ndarray, np.ndarray]] = {'train': (X, y)}
        d = pd.read_csv(path+'test'+str(idx)+'.csv')
        y = d['target'].to_numpy()
        d.drop('target', axis=1, inplace=True)
        X = d.to_numpy()
        result['test'] = (X, y)
        return result


