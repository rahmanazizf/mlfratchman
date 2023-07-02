import pandas as pd
import numpy as np

class NearestNeighbors:
    def __init__(self, kneighbors: int = 5, power: int = 2, weight: str = 'uniform', 
                 standardize: bool = True) -> None:
        self.kneighbors = kneighbors
        self.power = power
        self.weight = weight
        self._input_data = np.array([])
        self._output_data = np.array([])
        self.standardize = standardize

    def fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame):
        """Menyimpan data ke dalam instance model"""
        if self.standardize:
            self._input_data = np.array(self._standardize(X_train.copy()))
        self._input_data = np.array(X_train.copy())
        self._output_data = np.array(y_train.copy())

    def _calculate_dist(self, X_target):
        """Menghitung jarak titik target terhadap semua titik data train"""
        return np.sum(abs(self._input_data - X_target)**self.power, axis = 1)**(1/self.power)

    def _standardize(self, X):
        """Melakukan standardisasi data input"""
        avg = np.mean(X, axis=0)
        stdev = np.std(X, axis=0)
        return (X - avg)/stdev