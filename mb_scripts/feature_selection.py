from sklearn.base import TransformerMixin
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class UnivariateFeatureSelection(TransformerMixin):

    def __init__(self, n_features, type_of_model, scoring):

        if type_of_model == "c":
            scoring_methods = {"chi2": chi2, "f_classif": f_classif, "mutual_info_classif": mutual_info_classif}
        elif type_of_model == "r":
            scoring_methods = {"f_regression": f_regression, "mutual_info_regression": mutual_info_regression}

        if isinstance(n_features, int):
            self.selection = SelectKBest(scoring_methods[scoring], k=n_features)
        elif isinstance(n_features, float):
            self.selection = SelectPercentile(scoring_methods[scoring],
                                              percentile=(n_features * 100))

    def fit(self, X, y):
        return self.selection.fit(X, y)

    def transform(self, X):
        return self.selection.transform(X)

def plot_importance_rf(df, cols=None, type_of_model="c"):
    from sklearn import ensemble

    if cols != None:
        data = df[cols].copy()
    else:
        data = df.copy()

    if type_of_model == "c":
        rf = ensemble.RandomForestClassifier()
    else:
        rf = ensemble.RandomForestRegressor()

    rf = rf.fit(data.iloc[:, :-1], data.iloc[:, -1])
    imps = rf.feature_importances_
    idxs = np.argsort(imps)
    cols = df.columns

    sns.set_style("whitegrid")
    plt.title("FEATURE IMPORTANCES", size=20)
    plt.barh(range(len(idxs)), imps[idxs])
    plt.yticks(range(len(idxs)), [cols[i] for i in idxs])
    plt.xlabel("RANDOM FOREST IMPORTANCE")
    plt.show();