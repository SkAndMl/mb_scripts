![2022-04-25 (3)](https://user-images.githubusercontent.com/86184014/165095198-1f90196a-f4c5-42f5-92ed-60850a8386d5.png)
<br>
Creator : Sathya Krishnan Suresh<br>
## DESCRIPTION
This is a python package to quicken the modelling and data analysis process.<br><br>
PyPi : https://pypi.org/project/mb-scripts/

## Version 0.1.1 additions
1. **feature selection**<br>
Feature selection sub-package has been added which consists of UnivariateFeatureSelection and 
CombinationFeatureSelection.
2. **generate_time_series**<br>
This function can be used to test out new Recurrent Neural Architectures.
3. **Plotting feature importances using RandomForest**<br>
This function helps selecting the features that are most correlated with the target variable with 
the help of RandomForest
4. **Decomposition**<br>
The new model in 🔥Decomposition🔥 contains standard decomposition methods that help with 
dimensionality reduction.
5. **metrics update**<br>
A lot of regression metrics have been added along with multiclass precision metrics
variations.


## Installing and using mb_scripts
The package can be downloaded using `pip install mb-scripts==<latest_version>`<br><br>
Latest version : **0.1.0**.<br><br>
Once you have installed mb_scripts you can begin using it.<br><br> Here are some examples for using mb_scripts.<br><br>
`train_validation_curve_for_rf` - used to monitor `RandomForestClassifier`'s overfitting
![2022-05-07](https://user-images.githubusercontent.com/86184014/167239436-a77b2773-072e-4b66-b4ab-2b48089c9606.png)
<br><br>
`plot_decision_boundary` is used to visually look at the decision boundary of classification functions<br><br>
![2022-04-25 (1)](https://user-images.githubusercontent.com/86184014/165075925-daa9cdf5-cbe0-41fe-85fa-39395d4cf027.png)
<br><br>
`classifiers_metrics` returns a dataframe that consists of precision, recall, accuracy_score and f1_score for all the classification models passed.<br><br>
![2022-04-25 (2)](https://user-images.githubusercontent.com/86184014/165077324-b64aeb9f-170e-4630-a17e-5a0a9174a79e.png)
<br><br>

I am writing scripts regularly so the versions will keep changing for the next one month. Stay tuned.
