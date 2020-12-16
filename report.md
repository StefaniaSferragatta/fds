
## Report final project 

# Project Title: Predictors of mental health illness

## Team members: Martina Betti, Giacomo Ruà, Gaurav Ramse, Stefania Sferragatta, Mert Yildiz

---------------------------------------------------------------------------------------------------------------

### Introduction
We decided to focus on tuning of hyperparameters rather than on implementing from scratch each method.

## Evaluating models 

### 1. Logistic regression classifier

The first machine learning method used by the author is Logistic Regression through the sklearn module. This algorithm does not take any hyperparameter in input, hence we could not provide any tuning optimization. However, we provide our own version of the code for two main purposes: the first is to evaluate whether our implementation is suitable also for a greater number of features, secondly we want to compare the accuracy obtained with our model to that obtained with the built in function. The optimization algorithm we choose is the gradient ascent.
First of all we adapt the training set to the required input format, then we add an additional function (coefficients_sgd) in order to get the optimal starting value for the theta parameter (*theta0*). For the learning rate and for the number of epochs to be used in the regression, we initially put into practice what we learned from the previous homework by using the best combination for these two parameters. 
In order to obtain our final prediction, we want to classify each sample according to the log likelihood obtained with the product of the theta final vector and each sample features. We assume that if the log-likelihood is greater or equal than 0.5, then we classify one sample as "treatment yes", otherwise "treatment no". 
For comparison purposes, we applied the same evaluation methods that the author provide in the "evalModelClass" to our model.  As we can see in the table above, the model produced by our code has an accuracy of 0.74. In general our code performed a bit worse than the built in function in all the evaluation methods. 

|                         | Built-in           | Our code           |
| ----------------------- | ------------------ | ------------------ |
| ClassificationmAccuracy | 0.7936507936507936 | 0.7433862433862434 |
| Classification Error    | 0.2063492063492064 | 0.2566137566137566 |
| False Positive Rate     | 0.2617801047120419 | 0.2931937172774869 |
| Precision               | 0.7942436374835513 | 0.7437774729120586 |
| AUC score               | 0.7942436374835513 | 0.7437774729120586 |


### 2. KNeighbors classifier
The k-Nearest Neighbors is an algorithm that works on the entire training dataset, but when a prediction is required the k-most similar records to a new record are located and used for the prediction.
The main steps of this method are:
1. We calculate the euclidean distance between two rows in a dataset, where the rows are mostly made up of numbers;
2. The neighbors for a row in the test set are the k closest instances, as defined by our distance measure;
3. We sort all of the records in the training dataset by their distance to the new row;
4. We select the top k to return as the most similar neighbors. In the case of classification, we can return the most represented class among the neighbors.

|                       |Built-in           |Our code          |
|-----------------------|-------------------|------------------|
|Classification Accuracy| 0.8174603174603174|0.8121693121693122|
|Classification Error   | 0.1825396825396825|0.1878306878306878|
|False Positive Rate    | 0.2774869109947644|0.2774869109947644|
|Precision              | 0.7633928571428571|0.7612612612612613|
|AUC Score              | 0.8184757958395162|0.8131282022566285|

In order to tune the hyperparameters for the KNN built-in function the parameters that we have considered are:
 - k: number of clusters. The optimal k from the author analysis was 21, but since we have changed the set of parameters used for the classification, the new optimal number of clusters in a range from 1 to 31 is 11.
 - weight_options: according to our tuning the best option to weight the neighbors is the 'uniform' one, which does not assign more weight to more similar values.
 - distance_options: we add a new parameter in the tuning which estimates the type of distance that optimizes the predictions. According to our results the best one is the euclidean, that we have also used in our own implementation. 


### 3. Decision Tree classifier





### 4. Random Forest classifier

The Random Forest classifier technically is an ensemble method (based on the divide-and-conquer approach) of decision trees generated on a randomly split dataset. This collection of decision tree classifiers is also known as the forest. The individual decision trees are generated using an attribute selection indicator such as information gain, gain ratio, and Gini index for each attribute. Each tree depends on an independent random sample. In a classification problem, each tree votes and the most popular class is chosen as the final result. In the case of regression, the average of all the tree outputs is considered as the final result. It is simpler and more powerful compared to the other non-linear classification algorithms.

The algorithm works in four steps:

1. Select random samples from a given dataset.
2. Construct a decision tree for each sample and get a prediction result from each decision tree.
3. Perform a vote for each predicted result.
4. Select the prediction result with the most votes as the final prediction.

https://github.com/martinabetti-97/fds/blob/main/methods_documentation/images/pics/RF3.png



## Ensemble methods 


### 1. Bagging 

 
### 2. Boosting 


### 3. Stacking
The Stacking Classifier is an ensemble method that considers heterogeneous weak learners and combine them via a **meta-classifier** in order to improve predictions. So these output predictions are based on the multiple predictions returned by the combination of several machine learning models.


![N|stacking](http://rasbt.github.io/mlxtend/user_guide/classifier/StackingClassifier_files/stackingclassification_overview.png)

The image above summarises the process: 
* Firstly the individual classification models are trained based on the given training set; 
* Secondly the meta-classifier is fitted based on the outputs (meta-features) of the individual classification models in the ensemble;
* The output obtained at the end of the process is the final predicion.

In this project the author chose the function from the library ```mlxtend.classifier.StackingClassifier``` to implement the stacking classifier. 
In order to use this function it's needed to define: the learners to fit and the meta-model that combines them. 
As classification models to fit, the author chose the KNeighborsClassifier and the RandomForestClassifier whose predictions are combined by Logistic Regression as a meta-classifier. 


## Model Evaluation and Parameter Tuning
After building the predictive classification models, 
it's time to evaluate the performances of them.

For this purpose we used some common metrics and methods for assessing the performance of predictive classification models, including:
* **Classification accuracy:** percentage of the correct predictions;
* **Null accuracy:** accuracy that could be achieved by always predicting the most frequent class;
* **Confusion matrix:** Table of size 2x2 that describes the performance of a classification model. It's used in order to determine how many observations were correctly or incorrectly classified and it works comparing the observed and the predicted outcome values and showing the number of correct and incorrect predictions categorized by type of outcome. The diagonal elements of the confusion matrix indicate correct predictions, while the off-diagonals represent incorrect predictions; hence the correct classification rate is the sum of the number on the diagonal divided by the sample size in the test data;
* **False Positive Rate** represents the proportion of identified positives among the healthy individuals (i.e. "illness-negative"). It is calculated as ```1-TrueNegatives/(TrueNegatives + FalseNegatives)```;
* **Precision of Positive value**: is the proportion of true positives among all the individuals that have been predicted to be "illness-positive" by the model. This represents the accuracy of a predicted positive outcome and is computed as: ```Precision = TruePositives/(TruePositives + FalsePositives)``` ;
* **ROC curve**: is one of the most used graphical measure for assessing the performance or the accuracy of a classifier, which corresponds to the total proportion of correctly classified observations;
* **AUC**: is the percentage of the ROC plot that is underneath the curve, it summarizes the overall performance of the classifier, over all possible probability cutoffs. The metric used are:
    - .90-1 = excellent (A)
    - .80-.90 = good (B)
    - .70-.80 = fair (C)
    - .60-.70 = poor (D)
    - .50-.60 = fail (F)

---------------------------------------------------------------------------------------------------------------------------------------------------------------
## Conclusion
