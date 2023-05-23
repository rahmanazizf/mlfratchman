import numpy as np
import pandas as pd


class NearestNeighbors:

    DIST_TYPE = ('euclidean', 'manhattan')

    def __init__(self, n_neighbors: int = 3, standardize: bool = True, p:int=2) -> None:
        self._input_data = np.array([])
        self._output_data = np.array([])
        self.n_neighbors = n_neighbors
        self.standardize = standardize
        self.p = p

    def fit(self, X: np.ndarray, y: np.ndarray):
        self._input_data = np.array(X).reshape(len(X), -1)
        self._output_data = np.array(y).reshape(len(y), -1)

    def _euclidean(self, X_train: np.ndarray, X_target: np.ndarray) -> np.ndarray:
        return (X_train - X_target)**2

    def _manhattan(self, X_train: np.ndarray, X_target: np.ndarray) -> np.ndarray:
        return abs(X_train - X_target)

    def _calc_dist(self, X_target: np.ndarray, X_train: np.ndarray) -> np.ndarray:
        """Menghitung jarak setiap titik terhadap target"""
        assert X_target.shape == (
            1, 2), f"Expected target shape (1, 2), got {X_target.shape}"
        dist = (abs(X_train - X_target))**self.p
        row_sum = np.sum(dist, axis=1)
        return row_sum**0.5

    def _predict_proba(self, X_test: np.ndarray):
        nearest_distances = self._calc_dist(X_test, self._input_data)
        n_proba = self._output_data[list(
            np.argsort(nearest_distances)[:self.n_neighbors])]
        df = pd.DataFrame(n_proba.value_counts(normalize=True)).reset_index()
        return np.array(df).T

    def _standardize(self):
        pass


def main():
    nn = NearestNeighbors(n_neighbors=5)
    inp = np.array(range(10))
    out = np.array(range(11, 21))
    print(inp.size)
    nn.fit(inp, out)
    print(nn._input_data)
    print(nn._output_data)


if __name__ == '__main__':
    main()
