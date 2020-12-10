## Tree classifier
## Random Forest
## Boosting

#  Words you must know to read this

* **ensemble method** : it is a machine learning technique that combines several base models in order to produce one optimal predictive model.

* **classifier** :  it is a special case of a *hypothesis* (nowadays, often learned by a machine learning algorithm). A *classifier* is a *hypothesis* or *discrete-valued function* that is used to assign (categorical) class labels to particular data points.

* **meta-classifier** : it is the classifier that makes a final prediction among all the predictions by using those predictions as features. So, it takes classes predicted by various classifiers and pick the final one as the result that you need. 

* **weak learner** : it is a classifier that is only slightly correlated with the true classification (it can label examples better than random guessing)

  

# Boosting

### What is it? 

**Boosting** is an ensemble method. Moreover it is a a family of ML algorithms that are united by the same philosophy: turning weak learners into better learners by building a model from the training data, then creating a second model that attempts to correct the errors from the first model, then creating a third model that attempts to correct the errors from the second model and so on. Here an illustrative picture: 

![p3](C:\Users\momor\OneDrive\Desktop\p3.png)

Because it is a family of algorithms we need to focus on one to really get into the details of the implementation, we chose the one used by the author of the project: **AdaBoost** .

### AdaBoost (Adaptive Boosting)

The main idea behind the AdaBoost algorithm is 

#### How does it work? (Outline)  

Here a sketch of the algorithm: 

![p4](images\pics\p4.jpg)

#### An example 

Let's review the sketch code trough an example.

We want to predict from the following data if a patient has heart disease:

​												![p1](images\pics\p1.jpg) 

Let's apply the code:

`Define a weight distribution over the examples` $D^1_N = \frac{1}{N}$ `for i = 1,2, ..., N`

![p2](C:\Users\momor\OneDrive\Desktop\notebooks\boosting\pics\p2.jpg)

`for round j = 1 to M do `

`j = 1`

`M` is the number of times we will repeat the updating of weights and the feeding them back to the models, what we said here will be explained in more details below.

`Build a model` $h_j$`from the training set using distribution` $D^j$

This part of the code is not really clear in the sketch of the algorithm, because what we want is to find between all the weak learners we are considering to improve trough our code, the one that has the best predictive capability. This will be explained in more details in the images below: 

With the data we are considering let's look at some really simple models the *stumps* , i.e. a node with two children, where in the node there is a condition and in the children the predictions if the condition is satisfied or not: 

![p3](C:\Users\momor\OneDrive\Desktop\notebooks\boosting\pics\p3.jpg)

And we build models for each feature: *Blocked arteries* and *Patient weights*. As you can see above for each model we count how many correct predictions were made and how many were not: 

 ![p4](C:\Users\momor\OneDrive\Desktop\notebooks\boosting\pics\p4.jpg)



Now we choose the model that performed better (i.e. the one that made more correct predictions). In our case will be the one in evidence in the image above.  

Now we calculate the *voting power* $\alpha$ (or *Amount of say*) of the best model trough the following formula (the reason behind this will be more clear below):

![p6](C:\Users\momor\OneDrive\Desktop\notebooks\boosting\pics\p6.jpg)

![p7](C:\Users\momor\OneDrive\Desktop\notebooks\boosting\pics\p7.jpg)

**Here the main idea of AdaBoost ** : we find the 1 incorrect prediction of the best model and we increase the weight of the sample: 

![p5](C:\Users\momor\OneDrive\Desktop\notebooks\boosting\pics\p5.jpg)

Using the formula below: 

![p8](C:\Users\momor\OneDrive\Desktop\notebooks\boosting\pics\p8.jpg)

![p9](C:\Users\momor\OneDrive\Desktop\notebooks\boosting\pics\p9.jpg)



`update` $D^{j+1}$ `from` $D^j$ `:`

`Increase weights of examples misclassified by` $h_j$

![p10](C:\Users\momor\OneDrive\Desktop\notebooks\boosting\pics\p10.jpg)

`Decrease weights of examples misclassified by` $h_j$

![image-20201210134618501](C:\Users\momor\AppData\Roaming\Typora\typora-user-images\image-20201210134618501.png)

![p11](C:\Users\momor\OneDrive\Desktop\notebooks\boosting\pics\p11.jpg)

And we are almost done the last part that we need to  do (that it's not specified in the `Sketch of algorithm`) is to normalize all the `new weights` found: 

![p12](C:\Users\momor\OneDrive\Desktop\notebooks\boosting\pics\p12.jpg)

And the first iteration is complete. 

`j = 2`

In the next iteration we will repeat what we have done in this first iteration but with the new weights:

 ![p13](C:\Users\momor\OneDrive\Desktop\notebooks\boosting\pics\p13.jpg)



#### How does it really work?

For the sake of completeness we will leave here the complete pseudo-code of the AdaBoost algorithm: 

![p14](C:\Users\momor\OneDrive\Desktop\notebooks\boosting\pics\p14.jpg)



### Usage in the project's code 

This is the part of the project's code were AdaBoosting is used: 

```python
def boosting():
    # Building and fitting 
    clf = DecisionTreeClassifier(criterion='entropy', max_depth=1)
    boost = AdaBoostClassifier(base_estimator=clf, n_estimators=500)
    boost.fit(X_train, y_train)
    
    # make class predictions for the testing set
    y_pred_class = boost.predict(X_test)
    
    print('########### Boosting ###############')
    
    accuracy_score = evalClassModel(boost, y_test, y_pred_class, True)

    #Data for final graph
    methodDict['Boosting'] = accuracy_score * 100
```

First of all the documentation from:

`sklearn.ensemble.AdaBoostClassifier(base_estimator=None, *, n_estimators=50, learning_rate=1.0, algorithm='SAMME.R', random_state=None)`

Let's explain each argument used in the code above: 

* `base_estimators` : The weak learners used to train the model  (in our example above would have been *stumps*). 

  Default : `DecisionTreeClassifier`

  

* `n_estimator` : number of weak learners to train iteratively (in our example it would have been 3). 

  Default : 50





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
