from sklearn.base import TransformerMixin
import numpy as np


class PCA(TransformerMixin):

    def __init__(self, n_components=None):
        self.n_components = n_components  # dimension that the data has to be reduced to

    def fit(self, X):
        if self.n_components is None:  # if the number of components is not mentioned keep the same dimension as the data
            self.n_components = X.shape[1]

        self.cov_mat = np.cov(X.T)  # covariance matrix
        self.eigen_vals, eigen_vecs = np.linalg.eig(self.cov_mat)  # get the eigen values and vectors

        # forms a list of tuples in which each tuple has an eigen value and a vector
        eigen_pairs = [(self.eigen_vals[i], eigen_vecs[:, i]) for i in range(len(self.eigen_vals))]
        eigen_pairs.sort(key=lambda k: k[0],
                         reverse=True)  # sorts the list in descending order based on the eigen value
        self.explained_variance_ratio = [self.eigen_vals[i] / np.sum(self.eigen_vals) for i in
                                         range(len(self.eigen_vals))]

        self.W = np.empty(shape=(X.shape[1], self.n_components))
        for i in range(self.n_components):
            self.W[:, i] = eigen_pairs[i][1]  # gets the first n_components eigen vectors into the W matrix
        return self

    def transform(self, X):
        return X.dot(self.W)  # transforms a d-dimensional data into n_components-dimensional data.

    def inverse_transform(self, X):
        return X.dot(self.W.T)


class LDA(TransformerMixin):

    def __init__(self, n_components=None):
        self.n_components = n_components

    def _calculate_mean(self, X, y):
        self.mean_vectors = []
        for label in np.unique(y):
            self.mean_vectors.append(np.mean(X[y == label], axis=0))

    def _within_class(self, X, y):
        d = X.shape[1]
        S_W = np.zeros(shape=(d, d))
        for label, mean_vec in zip(np.unique(y), self.mean_vectors):
            class_scatter = np.cov(X[y == label].T)
            S_W += class_scatter
        return S_W

    def _between_class(self, X, y):
        self.d = X.shape[1]
        mean_overall = np.mean(X, axis=0)
        S_B = np.zeros(shape=(self.d, self.d))
        for label, mean_vec in zip(np.unique(y), self.mean_vectors):
            n = len(X[y == label])
            mean_vec = mean_vec.reshape(self.d, 1)
            S_B += n * (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)
        return S_B

    def _compute_eigen_pairs(self):

        eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(self.S_W).dot(self.S_B))
        eigen_pairs = [(eigen_vals[i], eigen_vecs[i]) for i in range(len(eigen_vals))]
        self.eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)
        self.transform_matrix = np.empty(shape=(self.d, self.n_components))
        for i in range(self.n_components):
            self.transform_matrix[:, i] = self.eigen_pairs[i][1]

    def fit(self, X, y):
        if self.n_components is None:
            self.n_components = X.shape[1]
        self._calculate_mean(X, y)
        self.S_W = self._within_class(X, y)
        self.S_B = self._between_class(X, y)
        self._compute_eigen_pairs()
        return self

    def transform(self, X):
        return X.dot(self.transform_matrix)