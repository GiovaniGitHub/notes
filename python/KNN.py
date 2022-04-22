import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn import neighbors

from utils import split_dataset

DISTANCES_TYPES = [
    "braycurtis",
    "canberra",
    "chebyshev",
    "cityblock",
    "correlation",
    "cosine",
    "dice",
    "euclidean",
    "hamming",
    "jaccard",
    "jensenshannon",
    "kulsinski",
    "mahalanobis",
    "matching",
    "minkowski",
    "rogerstanimoto",
    "russellrao",
    "seuclidean",
    "sokalmichener",
    "sokalsneath",
    "sqeuclidean",
    "yule",
]


class KNearestNeighbors:
    def __init__(self, X, y, n_neighborhood=5):
        self.n_neighborhood = n_neighborhood
        self.X = X
        self.y = y

    def __get_labels_neighborhood(self, row, distance):
        """Get labels in neighborhood of row vector.

        Args:
            row (list): Vector that will classified.
            distance (str): Type of distance.

        Returns:
            double: class choiced.
        """
        try:
            labels_neighborhood = np.array(
                sorted(
                    np.append(cdist(self.X, [row], distance), self.y, axis=1),
                    key=lambda l: l[0],
                )
            )[: self.n_neighborhood, -1]

            return max(set(labels_neighborhood), key=list(labels_neighborhood).count)
        except Exception as e:
            raise e

    def predict(self, X_classifier, distance):
        return [self.__get_labels_neighborhood(row, distance) for row in X_classifier]


if __name__ == "__main__":
    PATH_FILE = (
        "/home/nobrega/Dados/Documentos/Estudos/notes/dataset/knn_classification.csv"
    )
    df = pd.read_csv(PATH_FILE)
    df.pop("id")
    y = df["class"].values
    y = y.reshape(len(y), 1)
    df.pop("class")

    X = df.values

    X_train, y_train, X_test, y_test = split_dataset(X, y, 0.7)

    n_neighbors = 15

    knn = KNearestNeighbors(X_train, y_train, n_neighbors)
    for type_distance in DISTANCES_TYPES:
        try:
            y_hat = np.array(knn.predict(X_test, type_distance))
            y_test = np.ndarray.flatten(np.array(y_test))
            
            acc = sum(((y_hat == y_test)*1.0))/len(y_test)
            print(f"Type of Distance {type_distance} Acurácia: {acc}")
        except:
            print(f"This distance type {type_distance} can't calculate")