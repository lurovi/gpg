import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler


def read_csv_data(folder_path: str, dataset_name: str, idx: int, scale_strategy: str = 'no') -> dict[str, tuple[np.ndarray, np.ndarray]]:
    if not folder_path.endswith('/'):
        raise AttributeError(f'Provided folder path does not end with /.')
    if scale_strategy not in ('no', 'standard', 'robust', 'minmax'):
        raise AttributeError(f'{scale_strategy} is an invalid scale strategy. Allowed ones: no, standard, robust, minmax.')

    d: pd.DataFrame = pd.read_csv(folder_path+dataset_name+'/'+'train'+str(idx)+'.csv')
    y: np.ndarray = d['target'].to_numpy()
    d.drop('target', axis=1, inplace=True)
    X: np.ndarray = d.to_numpy()
    
    if scale_strategy == 'standard':
        data_scaler: StandardScaler = StandardScaler()
    elif scale_strategy == 'robust':
        data_scaler: RobustScaler = RobustScaler()
    elif scale_strategy == 'minmax':
        data_scaler: MinMaxScaler = MinMaxScaler()    

    if scale_strategy != 'no':
        data_scaler = data_scaler.fit(X)
        X = data_scaler.transform(X)

    result: dict[str, tuple[np.ndarray, np.ndarray]] = {'train': (X, y)}
    
    d = pd.read_csv(folder_path+dataset_name+'/'+'test'+str(idx)+'.csv')
    y = d['target'].to_numpy()
    d.drop('target', axis=1, inplace=True)
    X = d.to_numpy()

    if scale_strategy != 'no':
        X = data_scaler.transform(X)

    result['test'] = (X, y)
    
    return result

