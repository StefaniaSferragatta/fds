# Predictors of mental health illness - Report

### Team members: 
#### Martina Betti,  Gaurav Ramse, Giacomo Ruà, Stefania Sferragatta, Mert Yildiz

---------------------------------------------------------------------------------------------------------------

## Introduction

#### Dataset
For our analytical project we toke as a reference an existing kaggle analysis. The dataset used by the author is a 2016 survey done by *OSMI menthal health in thech survey*. With over 1400 responses, the 2016 survey aims to measure attitudes towards mental health in the tech workplace, and examine the frequency of mental health disorders among tech workers. The dataset has 1443 rows, corresponding to the number of people interviewed, and the parameters are the following: 
1. Timestamp 
2. Age 
3. Gender 
4. Country 
5. State: If you live in the United States, which state or territory do you live in? 
6. self_employed: Are you self-employed? 
7. family_history: Do you have a family history of mental illness? 
8. treatment: Have you sought treatment for a mental health condition? 
9. work_interfere: If you have a mental health condition, do you feel that it interferes with your work? 
10. no_employees: How many employees does your company or organization have? 
11. remote_work: Do you work remotely (outside of an office) at least 50% of the time? 
12. tech_company: Is your employer primarily a tech company/organization?
13. benefits: Does your employer provide mental health benefits? 
14. care_options: Do you know the options for mental health care your employer provides? 
15. wellness_program: Has your employer ever discussed mental health as part of an employee wellness program? 
16. seek_help: Does your employer provide resources to learn more about mental health issues and how to seek help? 
17. anonymity: Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources? 
18. leave: How easy is it for you to take medical leave for a mental health condition?
19. mentalhealthconsequence: Do you think that discussing a mental health issue with your employer would have negative consequences?
20. physhealthconsequence: Do you think that discussing a physical health issue with your employer would have negative consequences? 
21. coworkers: Would you be willing to discuss a mental health issue with your coworkers? 
22. supervisor: Would you be willing to discuss a mental health issue with your direct supervisor(s)? 
23. mentalhealthinterview: Would you bring up a mental health issue with a potential employer in an interview? 
24. physhealthinterview: Would you bring up a physical health issue with a potential employer in an interview? 
25. mentalvsphysical: Do you feel that your employer takes mental health as seriously as physical health? 
26. obs_consequence: Have you heard of or observed negative consequences for coworkers with mental health conditions in your workplace? 
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
<img src="https://github.com/martinabetti-97/fds/blob/main/imgs/features_all.png">
2. Random selection: randomly select some of the features 
<img src="https://github.com/martinabetti-97/fds/blob/main/imgs/features_random.png">
3. Selection based on correlation matrix: select features above a certain correlation coefficient with the parameter of interest (treatment).
<img src="https://github.com/martinabetti-97/fds/blob/main/imgs/features_corr.png">

We will repeat the analysis for all these three sets of features and compare them.

#### Test and Training
The aouthor chose the classical approach of splitting X (parameters) and y (binary prediciton vector) into training and testing sets selecting at random the 30% of the rows and assigning them to the test set. The remaining number of rows will be used for training.

## Evaluation of the classification models
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

## ML Algorithms 

### 1. Logistic regression classifier

The first machine learning method used by the author is Logistic Regression through the sklearn module. \ 
This algorithm takes many hyperparameters in input here we list the ones we tested with tuning:
- `Solver`: this parameter determine which method will be used for regression among Lbfgs, Newton, Liblinear, Sag and Saga. The best accuracy was obtained with the newton method.
- `Penalty`: this parameter sets the normalization used in the penalization; for each method we usually have one penalization type but in many cases the best results is obtained with no penalty, as in our case.

#### Evaluation

Here we report the results obtained for this algorithm in the original version of the analysis, when the hyperparameters were not optimized. For this evaluation we will use the methods defined in the section "Evaluation of the classification models".
<img src="https://github.com/martinabetti-97/fds/blob/main/imgs/confusion_lr.png">
<img src="https://github.com/martinabetti-97/fds/blob/main/imgs/probabilities_lr.png">
<img src="https://github.com/martinabetti-97/fds/blob/main/imgs/roc_lr.png">

#### Our implementation
Additionally we provide our own version of the code for two main purposes: the first is to evaluate whether our implementation is suitable also for a greater number of features, secondly we want to compare the accuracy obtained with our model to that obtained with the built in function. The optimization algorithm we choose is the gradient ascent.
First of all we adapt the training set to the required input format, then we add an additional function (coefficients_sgd) in order to get the optimal starting value for the theta parameter (*theta0*). For the learning rate and for the number of epochs to be used in the regression, we initially put into practice what we learned from the previous homework by using the best combination for these two parameters. 
In order to obtain our final prediction, we want to classify each sample according to the log likelihood obtained with the product of the theta final vector and each sample features. We assume that if the log-likelihood is greater or equal than 0.5, then we classify one sample as "treatment yes", otherwise "treatment no". 
For comparison purposes, we applied the same evaluation methods that the author provide in the "evalModelClass" to our model.  As we can see in the table above, the model produced by our code has an accuracy of 0.74. In general our code performed a bit worse than the built in function in all the evaluation methods. 

|                         | Built-in | Our code |
| ----------------------- | -------- | -------- |
| Classification Accuracy | 0.794    | 0.743    |
| False Positive Rate     | 0.262    | 0.293    |
| Precision               | 0.761    | 0.743    |
| AUC score               | 0.794    | 0.743    |


### 2. KNeighbors classifier
The k-Nearest Neighbors is an algorithm that works on the entire training dataset, but when a prediction is required the k-most similar records to a new record are located and used for the prediction.

In order to tune the hyperparameters for the KNN built-in function the parameters that we have considered are:
 - `k`: number of clusters. The optimal k from the author analysis was 21, but after our improovments the new optimal number of clusters in a range from 1 to 31 is 15 as it is shown in the graph below.
<img src="https://github.com/martinabetti-97/fds/blob/main/imgs/knn.png">

 - `weight_options`: according to our tuning the best option to weight the neighbors is the 'uniform' one, which does not assign more weight to more similar values.
 - `distance_options`: we add a new parameter in the tuning which estimates the type of distance that optimizes the predictions. According to our results the best one is the euclidean, that we have also used in our own implementation. 
 
#### Evaluation

Here we report the results obtained for this algorithm in the original version of the analysis, when the hyperparameters were not optimized. For this evaluation we will use the methods defined in the section "Evaluation of the classification models".
<img src="https://github.com/martinabetti-97/fds/blob/main/imgs/confusion_knn.png">
<img src="https://github.com/martinabetti-97/fds/blob/main/imgs/probabilities_knn.png">
<img src="https://github.com/martinabetti-97/fds/blob/main/imgs/roc_knn.png">
 
#### Our implementation
As for logistic regression we implemented this function by ourself, we review the main steps:
1. We calculate the euclidean distance between two rows in a dataset, where the rows are mostly made up of numbers;
2. The neighbors for a row in the test set are the k closest instances, as defined by our distance measure;
3. We sort all of the records in the training dataset by their distance to the new row;
4. We select the top k to return as the most similar neighbors. In the case of classification, we can return the most represented class among the neighbors.

|                         | Built-in | Our code |
| ----------------------- | -------- | -------- |
| Classification Accuracy | 0.799    | 0.812    |
| False Positive Rate     | 0.236    | 0.187    |
| Precision               | 0.799    | 0.761    |
| AUC score               | 0.874    | 0.813    |


### 3. Decision Tree classifier

We are implementing descision tree in daily life, We take into consideration of criteria then go take some descision.
The Descision tree mentioned below is simple example in daily life and we have conclusion from tree that if we go out without mask then there are chances to be infected by covid19.

<img src ="https://github.com/martinabetti-97/fds/blob/main/imgs/Sample_descision_tree.PNG">

So how to made descion tree ?
We can follow these steps to make it happen
##### 1. Decide the feature for root node.

There are ways to major impurity out of those one is **gini** and other is **enropy**. By using this method we select root node. We will use gini as of now to show procedure. Each root node has 2 leaf node. We have calculated gini for each leaf node and then the weighted average of gini. In this way we calculate weighted average for each feature and selects having lowest impurity(weighted average).
              
##### 2. Selection on next feature to take descision

##### 3. Execute this in recursive algorithm to get Descision tree.
    Our algorithm will stop when it will reach one of the conditions.
    a) max_depth-  The maximum depth of the tree.
    b) min_sample_leaf- The minimum number of samples required to be at a leaf node.
    etc..

In this method we tuned the parameters by GridsearchCv from sklearn, You can see from the figure descision tree that has been created.

<img src ="https://github.com/martinabetti-97/fds/blob/main/imgs/Descison_Tree.png">

#### Parameters

The parameter that has been tuned by GridsearchCV,

    a) criterion : gini, entropy
    
    b) max_depth : [2,8]
    
    c) min_sample_leaf : [2,8]
    
    d) min_samples_split : [2,8] 
    

### 4. Random Forest classifier

The Random Forest classifier technically is an ensemble method (based on the divide-and-conquer approach) of decision trees generated on a randomly split dataset. This collection of decision tree classifiers is also known as the forest. The individual decision trees are generated using an attribute selection indicator such as information gain, gain ratio, and Gini index for each attribute. Each tree depends on an independent random sample. In a classification problem, each tree votes and the most popular class is chosen as the final result. In the case of regression, the average of all the tree outputs is considered as the final result. It is simpler and more powerful compared to the other non-linear classification algorithms.

The algorithm works in four steps:

1. Select random samples from a given dataset.
2. Construct a decision tree for each sample and get a prediction result from each decision tree.
3. Perform a vote for each predicted result.
4. Select the prediction result with the most votes as the final prediction.

<img src="https://github.com/martinabetti-97/fds/blob/main/methods_documentation/images/pics/RF3.png">

In this method we have used the same method that author applied but we have improved the result with hyperparameters tuning since the author was generating the result with random parameter. The accuracy improvement will be compared in improvements and comparisons section. 

#### Parameters

The parameter that has been tuned by GridsearchCV,

    a) criterion : gini, entropy
    
    b) max_depth : [2,8]
    
    c) min_sample_leaf : [2,8]
    
    d) min_samples_split : [2,8]
    
    e) n_estimator : 20

## Ensemble methods 
An ensemble method is a technique that combines the predictions from multiple machine learning algorithms together to make more accurate predictions than any individual model.

### 1. Bagging 
Bootstrap Aggregation (Bagging) is an ensemble method. The main idea behind bagging methods is to combine weak learners to outdo a single strong learner. Bagging method use the Here a general outline of how the algorithm works: 
1. Consider a certain number of weak learners. 
2.  
(**COMPLETE ABOVE**)


![bagging](https://upload.wikimedia.org/wikipedia/commons/c/c8/Ensemble_Bagging.svg)

The significant advantage of bagging is that it can be parallelised. As the different models are fitted independently from each other, intensive parallelisation techniques can be used if required. One the other hand one disadvantage of bagging is that it introduces a loss of interpretability of a model. The resultant model can experience lots of bias when the proper procedure is ignored. Despite bagging being highly accurate, it can be computationally expensive and this may discourage its use in certain instances.

#### Evaluation

Here we report the results obtained for this algorithm in the original version of the analysis, when the hyperparameters were not optimized. For this evaluation we will use the methods defined in the section "Evaluation of the classification models".

<img src="https://github.com/martinabetti-97/fds/blob/main/imgs/boos_cm.jpg">
<img src="https://github.com/martinabetti-97/fds/blob/main/imgs/boos_hp.jpg">
<img src="https://github.com/martinabetti-97/fds/blob/main/imgs/boos_roc.jpg">


#### Our refinements 

The author decided to leave as `n_estimators = 10` (the default number). We have decided to analyze the choice of this parameter and we found, as we can see in the graph below, that accuracy changes considerably when we consider different number of base estimators.

<img src="https://github.com/martinabetti-97/fds/blob/main/methods_documentation/images/pics/accuracy_score.jpg">

Due to the stochastic nature of this method there is no fixed value that maximizes the accuracy, therefore we have decided to test the algorithm on different number of trees many times and then taking the average of the number of trees who score highest accuracy at each iteration. As we can see in the table below thanks to our refinements we were able to improve the author's results.  

|                         | Author's code | Our code |
| ----------------------- | -------- | -------- |
| Classification Accuracy | 0.78    | 0.788    |
| False Positive Rate     | 0.283    | 0.283    |
| Precision               | 0.745    | 0.749    |
| AUC score               | 0.781    | 0.789    |
 
### 2. Boosting 
Boosting is another family of ensemble methods whose main goal is to transform weak learners into strong learners. The main idea behind this method is building a model from the training data, then creating a second model that attempts to correct the errors from the first model, then creating a third model that attempts to correct the errors from the second model and so on.

The outline of the algorithm is as follows: 
1. Define a weight distribution `D_1[i]` over the training instances 
2. Build a model `h_1` from the training set using the weight distribution `D_1`
3. Update `D_2` from `D_1`: <br/>
    a. Increase weights misclassified by `h_1` <br/>
    b. Decrease weights correctly classified by `h_1` <br/>
4. Repeat point 2. `M` times 

The image below illustrates the method. 

<img src="https://github.com/martinabetti-97/fds/blob/main/imgs/boost_algo.png">

#### Evaluaton

#### Our Refinements
As in the section before, we analyzed the hyperparameters of the method where the author used default values, in order to improve the accuracy of the method. The two parameters we looked at are the: `n_estimators` and the `learning_rate`. We used `GridSearchCV` (look below for more details about this algorithm) to tune these two hyperparameters. 


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



---------------------------------------------------------------------------------------------------------------------------------------------------------------
## Improvements and Comparisons

In this section, we have included the hyperparameters optimization methods and different futures selection to be considered as input. 

#### Tuning

Machine learning classification models usually need hyperparameter optimization. The author has applied a randomized search for models' hyperparameters tuning. This approach returns better accuracy than the specific parametrization while it is not the best for tuning according to the results that have been returned. Tuning the hyperparameter on the same training dataset might not make good predictions for data that is not already seen, which leads us to cross-validation.

#### Cross validation

Cross-validation is a model evaluation method that is better than residuals. The problem with residual evaluations is that they do not indicate how well the learner will do when it is asked to make new predictions for data it has not already seen. One way to overcome this problem is to not use the entire data set when training a learner. Some of the data is removed before training begins. Then when training is done, the data that was removed can be used to test the performance of the learned model on new data. This is the basic idea for a whole class of model evaluation methods called cross-validation (Schneider, 1997). The author has applied cross-validation to each classification model. The accuracy of the models is significantly better than randomized search but not as good as grid search CV.

#### Grid Search CV

Machine learning algorithms have hyperparameters that allow you to tailor the behavior of the algorithm to your specific dataset. Hyperparameters are different from parameters, which are the internal coefficients or weights for a model found by the learning algorithm. Unlike parameters, hyperparameters are specified by the practitioner when configuring the model. Typically, it is challenging to know what values to use for the hyperparameters of a given algorithm on a given dataset, therefore it is common to use random or grid search strategies for different hyperparameter values. The more hyperparameters of an algorithm that you need to tune, the slower the tuning process. Therefore, it is desirable to select a minimum subset of model hyperparameters to search or tune. Not all model hyperparameters are equally important. Some hyperparameters have an outsized effect on the behavior, and in turn, the performance of a machine learning algorithm (Jason, 2019). To make sure that we are improving the accuracy, we thought hyperparameter tuning and cross-validation together is better to apply. Therefore, we have decided to apply the grid search CV. On the returns, we have seen that the accuracy increased not surprisingly. The grid search CV has been applied to Logistic Regression, KNN, Decision Tree Classifier, and Random Forest methods.

#### Features selection

Here we provide the comparison among the three different methods used for features selection. We the results obtained with different optimazion models.
In the first graph we see the results we obtained when using all the features available in the dataset. 
<img src="https://github.com/martinabetti-97/fds/blob/main/imgs/comparisons_all.png">
As we can notice, when we use all the features for the classification, algorithms perform differently mainly depending on whether do hyperparameters tuning or not. In fact when tuning is not applied, the accuracy perfomance ranking of the algorithms is: Random forest, Logistic Regression, KNN and Tree Clssifier. On the other hand, we can notice how this ranking changes when we tune the hyperparameters, in fact now we obtain a more homogeneous set of accuracy results.


Now we select only those features that have a correlation coefficient greater than 0.1 with our parameter of interest (treatment). 
<img src="https://github.com/martinabetti-97/fds/blob/main/imgs/comparisons_corr.png">
In this case we can notice that the predition is overall more accurate than before. Morever it is quite striking how also the ranking changes: first of all the performances are more homogeneous among all the algorithms and optimization methods; secondly we can notice how the best performing algoirithms are always the random forest and the tree classifier.


At the end of the analysis we also try to selct the same number of parameters as in the previous case (six features) but in a random way. We want to proove that in general it is not a good approach. 
<img src="https://github.com/martinabetti-97/fds/blob/main/imgs/comparisons_random.png">
It's evident how in this case performances drop from a range of 80-90 % to 55-60 %. Furthermore we can see a similar behaviour in algorithms ranking as in the first case of features selection. 

## Conclusions

In conclusion, an existing kaggle project has been analyzed and improved as the final project of the Foundations of Data Science course. The project aims to predict whether a patient should be treated for his/her mental illness or not according to the values obtained in the dataset. Firstly, the data has been preprocessed and split into two parts as train and test data. Secondly, the machine learning classification models such as Logistic Regression, Random Forest, etc. have been described briefly. Then, the models have been evaluated with the author's application. For the analysis part, the KNN and Logistic Regression models that have been covered during the course were applied from scratch and the accuracy results were compared. Lastly, to have a deep understanding of the classification models and increase the accuracy, one of the most well-known hyperparameter optimization methods, the Grid search CV has been applied and the accuracy compared with the author's random search and cross-validation methods.


## References

Brownlee, J. (2020, August 27). Tune Hyperparameters for Classification Machine Learning Algorithms. Retrieved December 24, 2020, from https://machinelearningmastery.com/hyperparameters-for-classification-machine-learning-algorithms/

Schneider, J. (1997). Retrieved December 24, 2020, from https://www.cs.cmu.edu/~schneide/tut5/node42.html

