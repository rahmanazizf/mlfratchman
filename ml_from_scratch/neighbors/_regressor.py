from ._base import NearestNeighbors
import numpy as np
import pandas as pd

class KNNRegressor(NearestNeighbors):

    def __init__(self, kneighbors: int = 3, power: int = 2, weight: str = 'uniform', 
                 standardize: bool = True) -> None:
        super().__init__(kneighbors, power, weight, standardize)

    def _calculate_avg(self, output):
        return np.mean(output)

    def predict(self, X_target: pd.DataFrame):
        """Memprediksi output untuk data target"""
        pred_labels = []
        # hitung jarak
        for row in np.array(X_target):
            dist = self._calculate_dist(row)
            # urutkan dan ambil indeks baris
            sorted_indices = np.argsort(dist)[:self.kneighbors]
            # gunakan indeks baris untuk mengakses output/label
            neighbor_labels = self._output_data[sorted_indices]
            pred_labels.append(self._calculate_avg(neighbor_labels))
        return np.array(pred_labels).reshape(len(pred_labels),)