from neighbors._base import NearestNeighbors
import numpy as np
import pandas as pd

class KNNClassifier(NearestNeighbors):

    def __init__(self, **kwrds) -> None:
        super().__init__(**kwrds)

    def _predict_proba(self, X_test: np.ndarray):
        nearest_distances = self._calc_dist(X_test)
        n_proba = self._output_data[list(
            np.argsort(nearest_distances)[:self.k])]
        df_proba = pd.DataFrame(n_proba).value_counts(normalize = True)
        cls_proba = np.zeros(shape=(len(self.output_classes()), 2))
        for i, cls in enumerate(self.output_classes()):
            cls_proba[0, i] = cls
            cls_proba[1, i] = df_proba.get(cls, float(0))

        return cls_proba

    def predict(self, X_test: np.ndarray):
        y_pred = np.array([])
        for row in X_test:
            neighbors_proba = self._predict_proba(row)
            y_pred = np.append(y_pred, [neighbors_proba[0, np.argmax(neighbors_proba[1, :])]])
        return y_pred