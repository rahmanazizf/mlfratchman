from ._base import NearestNeighbors
import numpy as np

class KNNRegressor(NearestNeighbors):
    def __init__(self, k: int = 5, power: int = 2, standardize: bool = True) -> None:
        super().__init__(k, standardize, power)

    def _average(self, X_test:np.ndarray):
        nearest_distances = super()._calc_dist(X_test)
        nearest_output = self._output_data[list(
            np.argsort(nearest_distances)[:self.k])]
        return np.mean(nearest_output)

    def predict(self, X_test: np.ndarray):
        y_pred = np.array([])
        for row in X_test:
            y_pred = np.append(y_pred, [self._average(row)])
        return y_pred
