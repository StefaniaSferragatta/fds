# Predictors of mental health illness - Report

### Team members: 
#### Martina Betti,  Gaurav Ramse, Giacomo Ruà, Stefania Sferragatta, Mert Yildiz

---------------------------------------------------------------------------------------------------------------

## Introduction

#### Dataset
For our analytical project we toke as a reference an existing kaggle analysis. The dataset used by the author is a 2016 survey done by *OSMI menthal health in thech survey*. With over 1400 responses, the 2016 survey aims to measure attitudes towards mental health in the tech workplace, and examine the frequency of mental health disorders among tech workers. The dataset has 1443 rows, corresponding to the number of people interviewed, and the parameters are the following: \
1. Timestamp \
    2. Age \
    3. Gender \
    4. Country \
    5. State: If you live in the United States, which state or territory do you live in? \
    6. self_employed: Are you self-employed? \
    7. family_history: Do you have a family history of mental illness? \
    8. treatment: Have you sought treatment for a mental health condition? \
    9. work_interfere: If you have a mental health condition, do you feel that it interferes with your work? \
    10. no_employees: How many employees does your company or organization have? \
    11. remote_work: Do you work remotely (outside of an office) at least 50% of the time? \
    12. tech_company: Is your employer primarily a tech company/organization? \
    13. benefits: Does your employer provide mental health benefits? \
    14. care_options: Do you know the options for mental health care your employer provides? \
    15. wellness_program: Has your employer ever discussed mental health as part of an employee wellness program? \
    16. seek_help: Does your employer provide resources to learn more about mental health issues and how to seek help? \
    17. anonymity: Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources? \
    18. leave: How easy is it for you to take medical leave for a mental health condition? \
    19. mentalhealthconsequence: Do you think that discussing a mental health issue with your employer would have negative consequences? \
    20. physhealthconsequence: Do you think that discussing a physical health issue with your employer would have negative consequences? \
    21. coworkers: Would you be willing to discuss a mental health issue with your coworkers? \
    22. supervisor: Would you be willing to discuss a mental health issue with your direct supervisor(s)? \
    23. mentalhealthinterview: Would you bring up a mental health issue with a potential employer in an interview? \
    24. physhealthinterview: Would you bring up a physical health issue with a potential employer in an interview? \
    25. mentalvsphysical: Do you feel that your employer takes mental health as seriously as physical health? \
    26. obs_consequence: Have you heard of or observed negative consequences for coworkers with mental health conditions in your workplace? \ 
    27. comments: Any additional notes or comments 
    
#### Goals
Broady speaking, the final goal of the analysis is that of predicting if one has sought treatment for a mental health condition, but in reality we mainly want to go behond the prediction itself and analyze in depths all the choises taken by the author. As a matter of fact, we picked this analysis for many different reasons that we will briefly introduce. \
One aspect that we appreciated about this analysis is related to the ammount of different methods that are used to make the prediction. In our coursework we have encountered some of these methods, like Logistic Regression and KNN; for this final project we wanted to deal with different approaches in order to gain a deeper knowledge on the many possibilities that are available when dealing with machine learning. We know from the course that not only different ML algortihms can produce different results, but also one single algorithm can produce extremely different predictions when using different parameters. Therefore one of our main goal will be that of testing many approaches that the author has not included in the analysis and see how predictions differ in accuracy. \
A second aspect that we found interesting about this analysis is learning how to approach data preprocessing when variables are qualitative rather than quantitaive. We will explore and comment the author choises in data encoding. 

## Data pre-processing

#### Filtering and encoding

This is how the raw dataset looked like:

<img src="https://github.com/martinabetti-97/fds/blob/main/imgs/raw.png">

- Missing data: As a first step the columns `Timestamp`,`state` and `comments` are removed, since these had a high proportion of missing data that could not be retrieved or infered in anyway. The author also decided to remove the `country` parameter, instead we thought that a possible way to retrive some quantitative information from this paramater was to consult additional data from https://stats.oecd.org/Index.aspx?DataSetCode=BLI and sobstitute the name of the country with the esitamated life satisfaction level for that country. For other parameters such as `work_interfer` missing data was converted to the answer "don't know", while in the case of binary answers when one of the two options was extremely rare (e.g. `self_empolyed` = 'yes'), missing value were considered to be the most common answer.
- Encoding: Different encoding strategies were implemented for each parameter, for some parameters (e.g. Gender) all the possible answers were collected and manually identified as one of these three categories *male*, *female* and *trans*. For other categorical variables whith a reduced ammunt of variability (e.g. 4 possible values), those categorical paramters were converted to numeric ones with a range equal to the number of options. 
- Normalizaton: Finally we have normalized numerical data with the min-max method and scaled it when needed.

#### Feature selection

In order to evaluate how the number and the quality of the features influence the accuracy of the prediction we will try three different approaches:

1. No selection: keep all features 

2. Random selection: randomly select some of the features 

3. Selection based on correlation matrix: select features above a certain correlation coefficient with the parameter of interest (treatment). We report the correlation matrix down below.
<img src="https://github.com/martinabetti-97/fds/blob/main/imgs/corr_matrix.png">

We will repeat the analysis for all these three set of features and compare them.

#### Test and Training
The aouthor chose the classical approach of splitting X (parameters) and y (binary prediciton vector) into training and testing sets selecting at random the 30% of the rows and assigning them to the test set. The remaining number of rows will be used for training.


## ML Algorithms 

### 1. Logistic regression classifier

The first machine learning method used by the author is Logistic Regression through the sklearn module. \ 
This algorithm takes many hyperparameters in input here we list the ones we tested with tuning:
- `Solver`: this parameter determine which method will be used for regression among Lbfgs, Newton, Liblinear, Sag and Saga. The best accuracy was obtained with the newton method.
- `Penalty`: this parameter sets the normalization used in the penalization; for each method we usually have one penalization type but in many cases the best results is obtained with no penalty, as in our case.

#### Our implementation
Additionally we provide our own version of the code for two main purposes: the first is to evaluate whether our implementation is suitable also for a greater number of features, secondly we want to compare the accuracy obtained with our model to that obtained with the built in function. The optimization algorithm we choose is the gradient ascent.
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

In order to tune the hyperparameters for the KNN built-in function the parameters that we have considered are:
 - `k`: number of clusters. The optimal k from the author analysis was 21, but after our improovments the new optimal number of clusters in a range from 1 to 31 is 15.
 - `weight_options`: according to our tuning the best option to weight the neighbors is the 'uniform' one, which does not assign more weight to more similar values.
 - `distance_options`: we add a new parameter in the tuning which estimates the type of distance that optimizes the predictions. According to our results the best one is the euclidean, that we have also used in our own implementation. 
 
#### Our implementation
As for logistic regression we implemented this function by ourself, we review the main steps:
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


### 3. Decision Tree classifier

> Documentation here

#### Parameters

> Parameters explanation here

### 4. Random Forest classifier

The Random Forest classifier technically is an ensemble method (based on the divide-and-conquer approach) of decision trees generated on a randomly split dataset. This collection of decision tree classifiers is also known as the forest. The individual decision trees are generated using an attribute selection indicator such as information gain, gain ratio, and Gini index for each attribute. Each tree depends on an independent random sample. In a classification problem, each tree votes and the most popular class is chosen as the final result. In the case of regression, the average of all the tree outputs is considered as the final result. It is simpler and more powerful compared to the other non-linear classification algorithms.

The algorithm works in four steps:

1. Select random samples from a given dataset.
2. Construct a decision tree for each sample and get a prediction result from each decision tree.
3. Perform a vote for each predicted result.
4. Select the prediction result with the most votes as the final prediction.

<img src="https://github.com/martinabetti-97/fds/blob/main/methods_documentation/images/pics/RF3.png">

In this method we have used the same method that author applied but we have improved the result with parameter tuning since the author was generating the result with random parameter. We have applied the 

#### Parameters

> Parameters explanation here

## Ensemble methods 
An ensemble method is a technique that combines the predictions from multiple machine learning algorithms together to make more accurate predictions than any individual model.

### 1. Bagging 
Bootstrap Aggregation (Bagging), is a simple and very powerful ensemble method. Bagging methods form a class of algorithms which build several instances of a black-box estimator on random subsets of the original training set and then aggregate their individual predictions to form a final prediction. These methods come in many flavours but mostly differ from each other by the way they draw random subsets of the training set, when samples are drawn with replacement. \
Bagging offers the advantage of combining weak learners to to outdo a single strong learner. It also helps in the reduction of variance, hence eliminating the over-fitting of models in the procedure. If the base models trained on different samples have high variance (over-fitting), then the aggregated result would even it out thereby reducing the variance. This technique is chosen when the base models have high variance and low bias which is generally the case with models having high degrees of freedom for complex data. As they provide a way to reduce over-fitting, bagging methods work best with strong and complex models (e.g., fully developed decision trees), in contrast with boosting methods which usually work best with weak models (e.g., shallow decision trees). \

![bagging](https://upload.wikimedia.org/wikipedia/commons/c/c8/Ensemble_Bagging.svg)

The significant advantage of bagging is that it can be parallelised. As the different models are fitted independently from each other, intensive parallelisation techniques can be used if required. One the other hand one disadvantage of bagging is that it introduces a loss of interpretability of a model. The resultant model can experience lots of bias when the proper procedure is ignored. Despite bagging being highly accurate, it can be computationally expensive and this may discourage its use in certain instances.

#### Parameters
- Bootstrap Replicates: the original article for bagging reports that "we are getting most of the improvement using only 10 bootstrap replicates. More than 25 bootstrap replicates is love’s labor lost". We can hence assume that 10 replicates is a fair compromise between accuracy and efficiency.
- Learning set size: the same article we just mentioned suggests to use a size for the learning set as big as the initial learning set.
 
### 2. Boosting 

> Documentation here

#### Parameters

> Parameters explanation here

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


#### Stacking using Cross Validation 

We tried to improve the perfomarce of this last analysis usign the ```StackingCVClassifier``` from the same library.
This is an ensemble-learning meta-classifier for stacking as well but it also uses cross-validation to prepare the inputs for the level-2 classifier in order to prevent overfitting. 


![N|stackingCV](http://rasbt.github.io/mlxtend/user_guide/classifier/StackingCVClassifier_files/stacking_cv_classification_overview.png)


This method consists in the following steps:
1. The dataset is split into k folds;
2. In k successive rounds, k-1 folds are used to fit the first level classifier;
3. In each round, the first-level classifiers are then applied to the remaining 1 subset that was not used for model fitting in each iteration.

The resulting predictions are then stacked and provided as input data to the second-level classifier. After the training of the StackingCVClassifier, the first-level classifiers are fit to the entire dataset.


## Model Evaluation and Parameter Tuning
After building the predictive classification models, it's time to evaluate the performances of them.

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
* **Probability plots**: for each method we plot the histogram of the predicted probabilities for class 1. We obtain this information by using the ```method model.predict_proba(X_test)[:, 1]``` where the index refers to the probability that the data belong to class 1 (that means 'treatment yes').


---------------------------------------------------------------------------------------------------------------------------------------------------------------
## Results

#### Features selection

#### Methods

#### Tuning

#### Cross validation

## Conclusions
