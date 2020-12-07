## Tree classifier
## Random Forest
## Boosting
## Stacking
### What is
The Stacking Classifier is an ensemble method that **combines multiple classification models via a meta-classifier in order to improve predictions**.
It often considers heterogeneous weak learners because it combine different learning algorithms. 

### Main idea
Again, the idea of stacking is to **learn several different weak learners** and **combine them** by training a meta-model **to output predictions** based on the multiple predictions returned by these weak models.

### How to build?
We work on more than one level model. The first one consists in using the input data of size (m*n) with different ML models. Then take the **prediction** from these models and combine them to form a new matrix of size (m*M) where M is the number of models used. \
The data obtained are then used for the second level model that makes the final predictions. So basically the features for the 2nd level model are the predictions from the train set. The second level is used to make predictions on the test set.

### Create training data
The important part here is to create the training data, following this steps:
(Do this for each part of the training data)
1. split the training data into k-folds (like k-folds cross validation);
2. a base model is fitted on the k-1 parts and predictions are made for the kth part;
3. the base model is then fitted on the whole train dataset to calculate its performance on the test set.
All the steps are repeated for the other base models. 
- References: [StackingClassifier - mlxtend (rasbt.github.io)](http://rasbt.github.io/mlxtend/user_guide/classifier/StackingClassifier/)

## Bagging  (Bootstrap Aggregation)

### 1. Why?
Bagging offers the advantage of combining **weak learners to to outdo a single strong learner**. It also helps in the **reduction of variance**, hence **eliminating the over-fitting** of models in the procedure. If the base models trained on different samples have high variance (over-fitting), then the aggregated result would even it out thereby reducing the variance. 

### 2. What?
Bootstrap Aggregation (Bagging), is a simple and very powerful **ensemble method**. An ensemble method is a technique that combines the predictions from multiple machine learning algorithms together to make more accurate predictions than any individual model. In ensemble algorithms, bagging methods form a class of algorithms which **build several instances of a black-box estimator on random subsets** of the original training set and **then aggregate their individual predictions to form a final prediction**. Bagging methods come in many flavours but mostly differ from each other by the way they draw random subsets of the training set, when **samples are drawn with replacement**.

### 3. When and how?
This technique is chosen when the base models have high variance and low bias which is generally the case with models having high degrees of freedom for complex data. As they provide a way to reduce over-fitting, bagging methods **work best with strong and complex models (e.g., fully developed decision trees),** in contrast with boosting methods which usually work best with weak models (e.g., shallow decision trees). Decision trees are sensitive to the specific data on which they are trained. When bagging with decision trees, we are less concerned about individual trees over-fitting the training data. \
Properties with different methods:

- **How Many Bootstrap Replicates?** The unbagged rate is 29.1, so we are getting most of the improvement using only 10 bootstrap replicates. More than 25 bootstrap replicates is love’s labor lost.
- **How big should learning set be?**  Same size as the initial learning set L.
- **How well on knn?** The stability of nearest neighbor classification methods with respect to perturbations of the data distinguishes them from competitors such as trees and neural nets.
- **How well on linear regression?** performance is poor if there are many small but non-zero for x coefficient.
 
###  4. Pro and cons?
The big **advantage** of bagging is that it can be **parallelised**. As the different models are fitted independently from each other, intensive parallelisation techniques can be used if required. 
One **disadvantage** of bagging is that it introduces a **loss of interpretability** of a model. The resultant model can experience lots of bias when the proper procedure is ignored. Despite bagging being highly accurate, it can be **computationally expensive** and this may discourage its use in certain instances.

### 5. Sklearn module
In scikit-learn, bagging methods are offered as a unified [`BaggingClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html#sklearn.ensemble.BaggingClassifier "sklearn.ensemble.BaggingClassifier") meta-estimator,  taking as input a user-specified base estimator along with parameters specifying the strategy to draw random subsets. 
- `base_estimator` The base estimator to fit on random subsets of the dataset.
- `n_estimator`The number of base estimators in the ensemble.
- `max_samples` and `max_features` control the size of the subsets (in terms of samples and features).
- `bootstrap` and `bootstrap_features` control whether samples and features are drawn with or without replacement.
- `oob_score=True` estimate the generalization error.
- `warm_start=True` reuse the solution of the previous call to fit and add more estimators to the ensemble
- `n_jobs=int` the number of jobs to run in parallel

### 6. Useful links
original article https://link.springer.com/article/10.1023/A:1018054314350 \
youtube quick guide https://www.youtube.com/watch?v=2Mg8QD0F1dQ