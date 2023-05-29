import numpy as np
import pandas as pd


class NearestNeighbors:

    def __init__(self, k: int = 3, standardize: bool = True, power: int = 2):
        self._input_data = []
        self._output_data = []
        self.k = k
        self.p = power
        self.standardize = standardize

    # TODO: tambahkan weight
    def weight(self, X_target):
        dist = np.sum((abs(self._input_data - X_target))**self.p, axis=1)**(1/self.p)
        return 1/(dist)**2

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        """Fitting data training yang akan digunakan untuk prediksi
        X: <ndarray> data input
        y: <ndarray> data output
        """
        self._input_data = np.array(X).reshape(len(X), -1)
        self._output_data = np.array(y).reshape(len(y), -1)
        assert self._input_data.shape[0] == self._output_data.shape[0], \
            f"X and y have different shape, X: {self._input_data.shape}; y: {self._output_data.shape}"
        if not self.standardize:
            return
        self._input_data = self._standardize(self._input_data)

    def _calc_dist(self, X_target: np.ndarray) -> np.ndarray:
        """Menghitung jarak setiap titik data training terhadap target
        X_target: <ndarray> data target (data yang ingin diprediksi)
        return
            <ndarray> jarak setiap titik data training terhadap target
        """
        assert X_target.shape == (self._input_data.shape[1], 
                                  ), f"Expected target shape ({self._input_data.shape[0]},), got {X_target.shape}"
        dist = (abs(self._input_data - X_target))**self.p
        row_sum = np.sum(dist, axis=1)
        return row_sum**(1/self.p)

    def _predict_proba(self, X_test: np.ndarray):
        nearest_distances = self._calc_dist(X_test)
        n_proba = self._output_data[list(
            np.argsort(nearest_distances)[:self.k])]
        # df = pd.DataFrame(n_proba).value_counts(normalize=True).reset_index()
        # return np.array(df).T
        df_proba = pd.DataFrame(n_proba).value_counts(normalize = True)
        cls_proba = np.zeros(shape=(len(self.output_classes()), 2))
        for i, cls in enumerate(self.output_classes()):
            cls_proba[0, i] = cls
            cls_proba[1, i] = df_proba.get(cls, float(0))

        return cls_proba
    
    def output_classes(self):
        return np.unique(self._output_data)

    def _standardize(self, input_data: np.ndarray) -> np.ndarray:
        """Standardize input data
        input_data: <ndarray> input data with n-rows and m-features
        return
            <ndarray> standardized data
        """
        avg = np.mean(input_data, axis=0)
        stdev = np.std(input_data, axis=0)
        return (self._input_data - avg)/stdev

def main():
    nn = NearestNeighbors(k=5)
    inp = np.array(range(10))
    out = np.array(range(11, 21))
    print(inp.size)
    nn.fit(inp, out)


if __name__ == '__main__':
    main()
