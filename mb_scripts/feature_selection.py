from sklearn.base import TransformerMixin
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile
from mb_scripts.useful_scripts import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, mean_squared_error
from itertools import combinations


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


def plot_importance_rf(df, cols=None, type_of_problem="classification"):

    """
    This function is used to plot the importance of each feature for a given dataset using RandomForest.
    df -> dataset, type pandas.DataFrame
    cols -> the features for which importance has to be plotted. If None all the features are chosen
    type_of_problem -> specifies whether the problem is a classification or regression problem
    """
    from sklearn import ensemble

    if cols != None:
        data = df[cols].copy()
    else:
        data = df.copy()

    if type_of_problem == "classification":
        rf = ensemble.RandomForestClassifier()
    elif type_of_problem == "regression":
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
    plt.show()


class CombinationFeatureSelection:

    def __init__(self, model, type_of_problem="classification", num_features=3):
        self.model = model
        self.type_of_problem = type_of_problem
        self.num_features = num_features

    def select_features(self, df, test_size=0.3, random_state=42):
        cols = df.columns
        cols = cols[:-1]
        features = []
        scores = []
        best_features = []
        best_score = 0
        for x in combinations(cols, self.num_features):
            data = df.loc[:, x]
            if self.type_of_problem == "classification":
                X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-1], data.iloc[:, -1],
                                                                    test_size=test_size, stratify=df.iloc[:, -1],
                                                                    random_state=random_state)
                y_train = y_train.astype(int)
                y_test = y_test.astype(int)
                model = self.model
                model = model.fit(X_train, y_train)
                pred = model.predict_proba(X_test)
                pred = np.argmax(pred, axis=1)
                score = f1_score(y_test, pred, average="weighted")
                print(f"For features: {x}, f1_score is {score}")
                features.append(x)
                scores.append(score)
                if score > best_score:
                    best_score = score
                    best_features = x
            else:
                X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-1], data.iloc[:, -1],
                                                                    test_size=test_size,
                                                                    random_state=random_state)
                model = self.model
                model = model.fit(X_train, y_train)
                pred = model.predict(X_test)
                rmse = np.sqrt(mean_squared_error(y_test, pred))
                print(f"For features: {x} rmse is {rmse}")
                features.append(x)
                scores.append(rmse)
                if rmse < best_score:
                    best_score = rmse
                    best_features = x
        return best_features, best_score