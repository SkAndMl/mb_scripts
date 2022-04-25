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

    classifier = classifier.fit(X,y)
    Z = classifier.predict(np.c_[np.ravel(xx),np.ravel(yy)])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx,yy,Z)  # the contour plot

    for i in np.unique(y):
        plt.scatter(X[y==i,0],X[y==i,1],color=colors[i],marker=markers[i],label=i)

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