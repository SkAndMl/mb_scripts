import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_decision_boundary(classifier,X,y,resolution=0.02,markers=None,colors=None):
    """
    This is a function that is used to visualize the boundaries predicted by classifiers to classify the training data.
    This function only takes uses two features even if more than two are given.
    :param classifier: classifier model that is used to predict the labels
    :param X: training data
    :param y: training label
    :param resolution: resolution of the plot
    :param markers: markers for different classes
    :param colors: colors for different classes
    :return: a figure consisting of the boundaries for each class
    """

    import matplotlib.pyplot as plt
    import seaborn as sns

    if markers==None:
        markers = ['*','s','o']
    if colors==None:
        colors = ['blue','red','orange']

    x_min,x_max = X[:,0].min()-0.1,X[:,0].max()+0.1  # x-axis range
    y_min,y_max = X[:,1].min()-0.1,X[:,1].max()+0.1  # y_axis range

    xx,yy = np.meshgrid(np.arange(x_min,x_max,resolution),
                        np.arange(y_min,y_max,resolution))  # creating a 2x2 array for the figure

    y_unq = np.unique(y)    
    classifier = classifier.fit(X,y)
    Z = classifier.predict(np.c_[np.ravel(xx),np.ravel(yy)])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx,yy,Z)  # the contour plot
    

    for i,j in enumerate(y_unq):
        plt.scatter(X[y==j,0],X[y==j,1],color=colors[i],marker=markers[i],label=j)

    plt.legend()
    plt.show()


def count_plot(df,hue=None,cols=None):
  import seaborn as sns
  import pandas as pd
  import matplotlib.pyplot as plt
  sns.set_style("whitegrid")
  if cols==None:
    cols = []
    for i in df.columns:
      if(df[i].nunique()<10):
        cols.append(i)
  fig,axes = plt.subplots(nrows=len(cols),ncols=1,figsize=(10,5))
  for i,col in enumerate(cols):
    if hue:
      sns.countplot(data=df,x=col,ax=axes[i],hue=hue)
    else:
      sns.countplot(data=df,x=col,ax=axes[i])
  fig.tight_layout()
  plt.show()

def plot_2(x,y,z,label_y,label_z,title):
    import matplotlib.pyplot as plt
    plt.plot(x,y,'b-',label=label_y)
    plt.plot(x,z,'r--',label=label_z)
    plt.title(title)
    plt.legend()
def train_validation_curve_for_rf(X,y,val_size=0.3,rs=42,epochs=10,n_estimators=False,max_depth=True,
                                 max_depth_start=6,n_estimators_start=100,criterion='gini'):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score,precision_score,recall_score
    from useful_scripts import train_test_split
    X_tr,X_val,y_tr,y_val = train_test_split(X,y,test_size=val_size,random_state=rs)
    acc_tr,acc_val = [0.5],[0.5]
    pr_tr,pr_val = [],[]
    rec_tr,rec_val = [],[]
    if max_depth:
        for depth in range(max_depth_start,max_depth_start+epochs):
            rf = RandomForestClassifier(criterion=criterion,max_depth=depth,n_estimators=100)
            rf = rf.fit(X_tr,y_tr)
            pred_tr = rf.predict(X_tr)
            pred_val = rf.predict(X_val)
            acc_tr.append(accuracy_score(y_tr,pred_tr))
            acc_val.append(accuracy_score(y_val,pred_val))
            pr_tr.append(precision_score(y_tr,pred_tr))
            pr_val.append(precision_score(y_val,pred_val))
            rec_tr.append(recall_score(y_tr,pred_tr))
            rec_val.append(recall_score(y_val,pred_val))
        fig,ax = plt.subplots(nrows=3,figsize=(10,8))
        plt.sca(ax[0])
        plot_2([x for x in range(max_depth_start,max_depth_start+epochs)],acc_tr,acc_val,'train','val','train/val accuracy')
        plt.sca(ax[1])
        plot_2([x for x in range(max_depth_start,max_depth_start+epochs)],pr_tr,pr_val,'train','val','train/val precision')
        plt.sca(ax[2])
        plot_2([x for x in range(max_depth_start,max_depth_start+epochs)],rec_tr,rec_val,'train','val','train/val recall')
    elif n_estimators:
        for n_estimator in range(n_estimators_start,n_estimators_start+epochs):
            rf = RandomForestClassifier(criterion=criterion,max_depth=6,n_estimators=n_estimator)
            rf = rf.fit(X_tr,y_tr)
            pred_tr = rf.predict(X_tr)
            pred_val = rf.predict(X_val)
            acc_tr.append(accuracy_score(y_tr,pred_tr))
            acc_val.append(accuracy_score(y_val,pred_val))
            pr_tr.append(precision_score(y_tr,pred_tr))
            pr_val.append(precision_score(y_val,pred_val))
            rec_tr.append(recall_score(y_tr,pred_tr))
            rec_val.append(recall_score(y_val,pred_val))
        fig,ax = plt.subplots(nrows=3,figsize=(10,8))
        plt.sca(ax[0])
        plot_2([x for x in range(n_estimators_start,n_estimators_start+epochs)],
               acc_tr,acc_val,'train','val','train/val accuracy')
        plt.sca(ax[1])
        plot_2([x for x in range(n_estimators_start,n_estimators_start+epochs)],
               pr_tr,pr_val,'train','val','train/val precision')
        plt.sca(ax[2])
        plot_2([x for x in range(n_estimators_start,n_estimators_start+epochs)],
               rec_tr,rec_val,'train','val','train/val recall')
