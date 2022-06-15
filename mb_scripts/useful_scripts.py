import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .metrics import precision_binary,recall_binary,accuracy_score,f1_score

def train_test_split(X,y,test_size=0.2,random_state=42):

    """
    Accepts only a dataframe or a numpy array as input.
    :param X: input data X
    :param y: input data y
    :param test_size: specifies the size of the test dataset.
    :param random_state: seed for shuffling the data
    :return: X_train,X_test,y_train,y_test
    """

    np.random.seed(random_state)
    shuffled_index = np.random.permutation(len(X))
    train_indices = shuffled_index[:int(len(X)*(1-test_size))]
    test_indices = shuffled_index[int(len(X)*(1-test_size)):]
    if type(X)==type(pd.DataFrame(data={1:[2,3]})):
        X_train,X_test,y_train,y_test = X.iloc[train_indices],X.iloc[test_indices],y.iloc[train_indices],y.iloc[test_indices]
        return X_train, X_test, y_train, y_test
    elif type(X)==type(np.array([1,2])):
        X_train,X_test,y_train,y_test = X[train_indices],X[test_indices],y[train_indices],y[test_indices]
        return X_train, X_test, y_train, y_test
    else:
        raise TypeError("Only dataframes and numpy arrays are accepted as input")


def classifiers_metrics(models,X,y,test_size=0.1,random_state=42):
    """
    :param models: a list or a numpy array consisting of classification models
    :param X: The whole feature set. It need not be split into training and test sets
    :param y: The whole true target labels.
    :param test_size: Size of the test data
    :param random_state: Specifies the random seed for splitting the dataset
    :return: returns a dataframe consisting of precision, recall, f1_score and accuracy of all the classifiers passed
    """
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=test_size,random_state=random_state)
    precision_list,recall_list,accuracy_list,f1_list = [],[],[],[]

    if type(models)!=type([1,2,3]) and type(models)!=type(np.array([1,2,3])):
        raise TypeError("models should be of type list or numpy array")

    for model in models:
        model = model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        precision_list.append(precision_binary(y_test,y_pred))
        recall_list.append(recall_binary(y_test,y_pred))
        accuracy_list.append(accuracy_score(y_test,y_pred))
        f1_list.append(f1_score(y_test,y_pred))

    metric_df = pd.DataFrame(index=models,data={"Precision":precision_list,
                                                "Recall":recall_list,
                                                "Accuracy":accuracy_list,
                                                "F1 Score":f1_list})
    return metric_df

def gini_impurity(value_set):
  # node impurity measurement for decision tree
  total_num = np.sum(value_set)
  gini = 1
  for j in value_set:
    gini -= (j/total_num)**2
  return np.round(gini,3)

def entropy(value_set):
  # node impurity measurement for decision tree
  total_num = np.sum(value_set)
  ig = 0
  smoothing_term = 10e-7
  for x in value_set:
    p = (x+smoothing_term)/total_num
    ig -= p*np.log2(p)
  return np.round(ig,3)

def generate_n_folds(data,n_splits=5):

    """
    :param data: input dataframe which should be the trainig set
    :param n_splits: number of folds required
    :return: returns the input df with an extra column "kfold" specifying the fold a particular instance belongs to
    """

    from sklearn import model_selection
    import pandas as pd
    import numpy as np

    training_data = data.copy()
    training_data[:,"kfold"] = -1*np.ones(len(data),dtype=np.int32)
    training_data = training_data.sample(frac=1)
    kf = model_selection.KFold(n_splits=n_splits)

    for fold,(trn_,val_) in enumerate(kf.split(X=training_data)):
        training_data.loc[val_,"kfold"] = fold

    return training_data

def base_models_cv(models,param_grids,X,y,cv=5):

    """
    :param models: list of estimators
    :param param_grids: list of dictionaries of params for each model
    :param X: training instances with their features
    :param y: target labels
    :param cv: number of folds for cross validation
    :return: prints the best params and scores for each estimator
    """

    from sklearn.model_selection import GridSearchCV

    for i,model in enumerate(models):
        gs = GridSearchCV(estimator=model,param_grid=param_grids[i],cv=cv)
        gs = gs.fit(X,y)
        print(f"Best parameters for {model}\n")
        for j in gs.best_params_.keys():
            print(f"{j:{15}}{gs.best_params_[j]}")
        print(f"Best score: {gs.best_score_}")
        print("-------------------------------")

def reduce_memory(data,memory_size_int=8,memory_size_float=16):
    import numpy as np
    import pandas as pd
    for col in data.columns:
        if str(data[col].dtype)[:1] == 'i':
            data[col] = data[col].astype(np.int8)
        elif str(data[col].dtype)[:1] == 'f':
            data[col] = data[col].astype(np.float16)
    return data

def generate_time_series(batch_size,n_steps):
  freq1,freq2,offset1,offset2 = np.random.rand(4,batch_size,1)
  time = np.linspace(0,1,n_steps)
  series = 0.5*np.sin((time-offset1)*(freq1*10+10))
  series += 0.2*np.sin((time-offset2)*(freq2*20+20))
  series += 0.1*(np.random.rand(batch_size,n_steps)-0.5)
  return series[...,np.newaxis]
