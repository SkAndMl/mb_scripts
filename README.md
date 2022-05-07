![2022-04-25 (3)](https://user-images.githubusercontent.com/86184014/165095198-1f90196a-f4c5-42f5-92ed-60850a8386d5.png)<br>
Creator : Sathya Krishnan Suresh<br>
## DESCRIPTION
This is a python package to quicken the modelling and data analysis process.<br><br>
PyPi : https://pypi.org/project/mb-scripts/

## Version 0.1.0 additions
1. **plot_pr_roc_curve**<br>
This function is used to plot precision, recall and roc measures of an estimator 
passed to it in separate subplots.
2. **tt_with_sc**<br>
You can now comfortably split the data into training and test sets and simultaneously
scaler the numerical features in your data.
3. **resnet**<br>
RESNET-34 CNN image classification architecture has been added to the already existing typicall cnn
architecture and lecun5 cnn architecture.
4. **dnn_scripts**<br>
The new module in 🔥mb_scripts🔥 that will contain functions for building dense models
with Tensorflow
5. **metrics update**<br>
A lot of regression metrics have been added along with multiclass precision metrics
variations.
6. **plotting update**<br>
`plot_2` and `train_validation_curve_for_rf` have been added and they will be super useful for
monitoring overfitting when you are training a RandomForestClassifier model


## Installing and using mb_scripts
The package can be downloaded using `pip install mb-scripts==<latest_version>`<br><br>
Latest version : **0.1.0**.<br><br>
Once you have installed mb_scripts you can begin using it.<br><br> Here are some examples for using mb_scripts.<br><br>
`plot_decision_boundary` is used to visually look at the decision boundary of classification functions<br><br>
![2022-04-25 (1)](https://user-images.githubusercontent.com/86184014/165075925-daa9cdf5-cbe0-41fe-85fa-39395d4cf027.png)<br><br>
`classifiers_metrics` returns a dataframe that consists of precision, recall, accuracy_score and f1_score for all the classification models passed.<br><br>
![2022-04-25 (2)](https://user-images.githubusercontent.com/86184014/165077324-b64aeb9f-170e-4630-a17e-5a0a9174a79e.png)<br><br>

You can also find a couple of cnn image classification models in `mb_scripts.cnn_image_architecture`.<br>

I am writing scripts regularly so the versions will keep changing for the next one month. Stay tuned.
