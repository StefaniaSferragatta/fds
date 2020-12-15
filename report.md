
# Final project

### Introduction
We decided to focus on tuning of hyperparameters rather than on implementing from scratch each method.

### Logistic regression

The first machine learning method used by the author is Logistic Regression through the sklearn module. This algorithm does not take any hyperparameter in input, hence we could not provide any tuning optimization. However, we provide our own version of the code for two main purposes: the first is to evaluate whether our implementation is suitable also for a greater number of features, secondly we want to compare the accuracy obtained with our model to that obtained with the built in function. The optimization algorithm we choose is the gradient ascent.
First of all we adapt the training set to the required input format, then we add an additional function (coefficients_sgd) in order to get the optimal starting value for the theta parameter (*theta0*). For the learning rate and for the number of epochs to be used in the regression, we initially put into practice what we learned from the previous homework by using the best combination for these two parameters. 
In order to obtain our final prediction, we want to classify each sample according to the log likelihood obtained with the product of the theta final vector and each sample features. We assume that if the log-likelihood is greater or equal than 0.5, then we classify one sample as "treatment yes", otherwise "treatment no". \newline
For comparison purposes, we applied the same evaluation methods that the author provide in the "evalModelClass" to our model.  As we can see in the table above, the model produced by our code has an accuracy of 0.74. In general our code performed a bit worse than the built in function in all the evaluation methods. 

|                         | Built-in           | Our code           |
| ----------------------- | ------------------ | ------------------ |
| ClassificationmAccuracy | 0.7936507936507936 | 0.7433862433862434 |
| Classification Error    | 0.2063492063492064 | 0.2566137566137566 |
| False Positive Rate     | 0.2617801047120419 | 0.2931937172774869 |
| Precision               | 0.7942436374835513 | 0.7437774729120586 |
| AUC score               | 0.7942436374835513 | 0.7437774729120586 |


### KNN
The k-Nearest Neighbors is an algorithm that works on the entire training dataset, but when a prediction is required the k-most similar records to a new record are located and used for the prediction.
The main steps of this method are:
1. We calculate the euclidean distance between two rows in a dataset, where the rows are mostly made up of numbers;
2. The neighbors for a row in the test set are the k closest instances, as defined by our distance measure;
3. We sort all of the records in the training dataset by their distance to the new row;
4. We select the top k to return as the most similar neighbors. In the case of classification, we can return the most represented class among the neighbors.

|                       |Built-in           |Our code          |
|-----------------------|-------------------|------------------|
|Classification Accuracy| 0.8174603174603174|0.8121693121693122|
|Classification Error   |0.18253968253968256|0.1878306878306878|
|False Positive Rate    | 0.2774869109947644|0.2774869109947644|
|Precision              | 0.7633928571428571|0.7612612612612613|
|AUC Score              | 8184757958395162  |0.8131282022566285|

In order to tune the hyperparameters for the KNN built-in function the parameters that we have considered are:
 - k: number of clusters. The optimal k from the author analysis was 21, but since we have changed the set of parameters used for the classification, the new optimal number of clusters in a range from 1 to 31 is 11.
 - weight_options: according to our tuning the best option to weight the neighbors is the 'uniform' one, which does not assign more weight to more similar values.
 - distance_options: we add a new parameter in the tuning which estimates the type of distance that optimizes the predictions. According to our results the best one is the euclidean, that we have also used in our own implementation. 
