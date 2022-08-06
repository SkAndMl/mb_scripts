from sklearn import model_selection,ensemble,base
import pandas as pd

class WeightedEnsembleClassifier(base.BaseEstimator):

    def __init__(self,models,cv_strategy):

        super(WeightedEnsembleClassifier,self).__init__()
        if not isinstance(models,list):
            models = (models,)
        
        self.models = models
        self.cv_strategy = cv_strategy
    
    def fit(self,X,y):

        if isinstance(X,pd.DataFrame):
            X = X.values
        
        if isinstance(y,pd.DataFrame) or isinstance(y,pd.Series):
            y = y.values

        cv_score_dict = {i:[] for i in range(len(self.models))}

        for train_ind,val_ind in self.cv_strategy.split(X,y):
            X_train,X_test = X[train_ind],X[val_ind]
            y_train,y_test = y[train_ind],y[val_ind]

            for i in range(len(self.models)):
                self.models[i] = self.models[i].fit(X_train,y_train)
                pred_proba = self.models[i].predict_proba(X_test)[:,1]
                cv_score_dict[i].append(metrics.roc_auc_score(y_test,pred_proba))
        
        tot_wt = 0
        for i in range(len(self.models)):
            cv_score_dict[i] = np.array(cv_score_dict[i]).mean()
            tot_wt += cv_score_dict[i]
        
        for i in range(len(self.models)):
            cv_score_dict[i] /= tot_wt
        
        self.wts_dict = cv_score_dict
        return self
    
    def predict(self,X):

        preds = np.zeros(shape=(len(X)))
        for i in range(len(self.models)):
            preds += self.wts_dict[i]*self.models[i].predict_proba(X)[:,1]
        
        return preds


class StackAndBlend(base.BaseEstimator):

    def __init__(self,models,blender,cv_strategy):
        
        super(StackAndBlend,self).__init__()
        if not isinstance(models,list):
            models = (models,)
        
        self.cv_strategy = cv_strategy
        self.models = models
        self.blender = blender
    
    def fit(self,X,y):

        val_inds = []

        if isinstance(X,pd.DataFrame):
            X = X.values
        if isinstance(y,pd.DataFrame) or isinstance(y,pd.Series):
            y = y.values

    
        pred_matrix = np.empty(shape=(len(X),len(self.models)))
        for train_ind,val_ind in self.cv_strategy.split(X,y):
            X_train,X_val = X[train_ind],X[val_ind]
            y_train,y_val = y[train_ind],y[val_ind]
            
            

            for i in range(len(self.models)):
                self.models[i] = self.models[i].fit(X_train,y_train)
                pred_matrix[len(val_inds):len(val_inds)+len(val_ind),i] = self.models[i].predict_proba(X_val)[:,1]
            
            val_inds.extend(list(val_ind))
        
        pred_df = pd.DataFrame(data=pred_matrix[:len(val_inds),:],columns=[f"pred_{i}" for i in range(len(self.models))])
        pred_df.index = val_inds
        pred_df.sort_index(inplace=True)
        val_inds.sort()
        pred_df["target"] = y[val_inds]
        self.pred_df = pred_df
        
        self.blender = self.blender.fit(self.pred_df.drop("target",axis=1),
                                        self.pred_df.target)
        
        return self
    
    def predict(self,X):
        
        if isinstance(X,pd.DataFrame):
            X = X.values
        
        pred_matrix = np.empty(shape=(len(X),len(self.models)))

        for i in range(len(self.models)):
            pred_matrix[:,i] = self.models[i].predict_proba(X)[:,1]
        
        y_final_pred = self.blender.predict_proba(pred_matrix)[:,1]
        return y_final_pred
class WeightedClassifier(base.BaseEstimator):

    def __init__(self,models,cv_strategy):

        super(WeightedClassifier,self).__init__()
        if not isinstance(models,list):
            models = (models,)
        
        self.models = models
        self.cv_strategy = cv_strategy
    
    def fit(self,X,y):

        if isinstance(X,pd.DataFrame):
            X = X.values
        
        if isinstance(y,pd.DataFrame) or isinstance(y,pd.Series):
            y = y.values

        cv_score_dict = {i:[] for i in range(len(self.models))}

        for train_ind,val_ind in self.cv_strategy.split(X,y):
            X_train,X_test = X[train_ind],X[val_ind]
            y_train,y_test = y[train_ind],y[val_ind]

            for i in range(len(self.models)):
                self.models[i] = self.models[i].fit(X_train,y_train)
                pred_proba = self.models[i].predict_proba(X_test)[:,1]
                cv_score_dict[i].append(metrics.roc_auc_score(y_test,pred_proba))
        
        tot_wt = 0
        for i in range(len(self.models)):
            cv_score_dict[i] = np.array(cv_score_dict[i]).mean()
            tot_wt += cv_score_dict[i]
        
        for i in range(len(self.models)):
            cv_score_dict[i] /= tot_wt
        
        self.wts_dict = cv_score_dict
        return self
    
    def predict(self,X):

        preds = np.zeros(shape=(len(X)))
        for i in range(len(self.models)):
            preds += self.wts_dict[i]*self.models[i].predict_proba(X)[:,1]
        
        return preds
class WeightedClassifier(base.BaseEstimator):

    def __init__(self,models,cv_strategy):

        super(WeightedClassifier,self).__init__()
        if not isinstance(models,list):
            models = (models,)
        
        self.models = models
        self.cv_strategy = cv_strategy
    
    def fit(self,X,y):

        if isinstance(X,pd.DataFrame):
            X = X.values
        
        if isinstance(y,pd.DataFrame) or isinstance(y,pd.Series):
            y = y.values

        cv_score_dict = {i:[] for i in range(len(self.models))}

        for train_ind,val_ind in self.cv_strategy.split(X,y):
            X_train,X_test = X[train_ind],X[val_ind]
            y_train,y_test = y[train_ind],y[val_ind]

            for i in range(len(self.models)):
                self.models[i] = self.models[i].fit(X_train,y_train)
                pred_proba = self.models[i].predict_proba(X_test)[:,1]
                cv_score_dict[i].append(metrics.roc_auc_score(y_test,pred_proba))
        
        tot_wt = 0
        for i in range(len(self.models)):
            cv_score_dict[i] = np.array(cv_score_dict[i]).mean()
            tot_wt += cv_score_dict[i]
        
        for i in range(len(self.models)):
            cv_score_dict[i] /= tot_wt
        
        self.wts_dict = cv_score_dict
        return self
    
    def predict(self,X):

        preds = np.zeros(shape=(len(X)))
        for i in range(len(self.models)):
            preds += self.wts_dict[i]*self.models[i].predict_proba(X)[:,1]
        
        return preds
