from ._base import NearestNeighbors
import numpy as np
import pandas as pd

class KNNClassifier(NearestNeighbors):

    def __init__(self, kneighbors: int = 3, power: int = 2, weight: str = 'uniform', 
                 standardize: bool = True) -> None:
        super().__init__(kneighbors, power, weight, standardize)

    def predict(self, X_target: pd.DataFrame):
        """Memprediksi output untuk data target"""
        pred_labels = []
        # hitung jarak
        for row in np.array(X_target):
            dist = self._calculate_dist(row)
            # urutkan dan ambil indeks baris
            sorted_indices = np.argsort(dist)[:self.kneighbors]
            # gunakan indeks baris untuk mengakses output/label
            neighborcls = self._output_data[sorted_indices]
            pred_labels.append(self._predict_proba(neighborcls))
        return np.array(pred_labels).reshape(len(pred_labels),)
        

    def _predict_proba(self, ncls: np.array):
        """Memprediksi label berdasarkan probabilitas
        Params
        ncls: <array> sebanyak n-kelas tetangga terdekat dari data target
        Return
        prediksi kelas <int>
        """
        label_proba = []
        for label in np.unique(self._output_data):
            label_proba.append([label, np.sum(ncls == label)/self.kneighbors])
        proba = np.array(label_proba)[:, 1]
        return np.argmax(proba)
