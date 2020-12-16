# Tree classifier
## Descision tree 
  how we implement Formulas for entrpy and ingformaton gain and whole detailed implentation can be founf in post of toward data science
  
1)  https://www.geeksforgeeks.org/decision-tree-implementation-python/#:~:text=Entropy%20is%20the%20measure%20of,the%20more%20the%20information%20content.&text=The%20entropy%20typically%20changes%20when,training%20instances%20into%20smaller%20subsets.
    
    This is the one which explains everything about descision tree
2)  https://towardsdatascience.com/entropy-how-decision-trees-make-decisions-2946b9c18c8

This is the youtube video , Where we can find detail implementation of descision tree /n
3) https://www.youtube.com/watch?v=K5QlpAqOtTE

Also we have code which we can mplement from scratch, But as we discussed earlier , We are going to optimize the accuracy of algorithm by hyperparameter optimization.



## Random Forest

In the Machine Learning world, Random Forest models are a kind of non parametric models that can be used both for regression and classification. They are one of the most popular ensemble methods, belonging to the specific category of Bagging methods.
Ensemble methods involve using many learners to enhance the performance of any single one of them individually. These methods can be described as techniques that use a group of weak learners (those who on average achieve only slightly better results than a random model) together, in order to create a stronger, aggregated one.
In our case, Random Forests are an ensemble of many individual Decision Trees.

Random Forest models combine the simplicity of Decision Trees with the flexibility and power of an ensemble model. In a forest of trees, we forget about the high variance of an specific tree, and are less concerned about each individual element, so we can grow nicer, larger trees that have more predictive power than a pruned one.

Although Random Forest models don’t offer as much interpret ability as a single tree, their performance is a lot better, and we don’t have to worry so much about perfectly tuning the parameters of the forest as we do with individual trees.
Okay, I get it, a Random Forest is a collection of individual trees. But why the name Random? Where is the Randomness? Lets find out by learning how a Random Forest model is built.

### Training and Building a Random Forest
Building a random Forest has 3 main phases. We will break down each of them and clarify each of the concepts and steps. Lets go!

#### Creating a Bootstrapped Data Set for each tree
When we build an individual decision tree, we use a training data set and all of the observations. This means that if we are not careful, the tree can adjust very well to this training data, and generalise badly to new, unseen observations. To solve this issue, we stop the tree from growing very large, usually at the cost of reducing its performance.

To build a Random Forest we have to train N decision trees. Do we train the trees using the same data all the time? Do we use the whole data set? Nope.
This is where the first random feature comes in. To train each individual tree, we pick a random sample of the entire Data set, like shown in the following figure.

<img src="https://github.com/martinabetti-97/fds/blob/main/methods_documentation/images/pics/RF1.png">

From looking at this figure, various things can be deduced. First of all, the size of the data used to train each individual tree does not have to be the size of the whole data set. Also, a data point can be present more than once in the data used to train a single tree (like in tree nº two).
This is called Sampling with Replacement or Bootstrapping: each data point is picked randomly from the whole data set, and a data point can be picked more than once.
By using different samples of data to train each individual tree we reduce one of the main problems that they have: they are very fond of their training data. If we train a forest with a lot of trees and each of them has been trained with different data, we solve this problem. They are all very fond of their training data, but the forest is not fond of any specific data point. This allows us to grow larger individual trees, as we do not care so much anymore for an individual tree overfitting.
If we use a very small portion of the whole data set to train each individual tree, we increase the randomness of the forest (reducing over-fitting) but usually at the cost of a lower performance.
In practice, by default most Random Forest implementations (like the one from Scikit-Learn) pick the sample of the training data used for each tree to be the same size as the original data set (however it is not the same data set, remember that we are picking random samples).
This generally provides a good bias-variance compromise.

#### Train a forest of trees using these random data sets, and add a little more randomness with the feature selection
If you remember well, for building an individual decision tree, at each node we evaluated a certain metric (like the Gini index, or Information Gain) and picked the feature or variable of the data to go in the node that minimised/maximised this metric.
This worked decently well when training only one tree, but now we want a whole forest of them! How do we do it? Ensemble models, like Random Forest work best if the individual models (individual trees in our case) are uncorrelated. In Random Forest this is achieved by randomly selecting certain features to evaluate at each node.

<img src="https://github.com/martinabetti-97/fds/blob/main/methods_documentation/images/pics/RF2.png">

As you can see from the previous image, at each node we evaluate only a subset of all the initial features. For the root node we take into account E, A and F (and F wins). In Node 1 we consider C, G and D (and G wins). Lastly, in Node 2 we consider only A, B, and G (and A wins). We would carry on doing this until we built the whole tree.
By doing this, we avoid including features that have a very high predictive power in every tree, while creating many un-correlated trees. This is the second sweep of randomness. We do not only use random data, but also random features when building each tree. The greater the tree diversity, the better: we reduce the variance, and get a better performing model.

Repeat this for the N trees to create our awesome forest.
Awesome, we have learned how to build a single decision tree. Now, we would repeat this for the N trees, randomly selecting on each node of each of the trees which variables enter the contest for being picked as the feature to split on.
In conclusion, the whole process goes as follows:
1. Create a bootstrapped data set for each tree.
2. Create a decision tree using its corresponding data set, but at each node use a random sub sample of variables or features to split on.
3. Repeat all these three steps hundreds of times to build a massive forest with a wide variety of trees. This variety is what makes a Random Forest way better than a single decision tree.
Once we have built our forest, we are ready to use it to make awesome predictions. Lets see how!

#### Making predictions using a Random Forest
Making predictions with a Random Forest is very easy. We just have to take each of our individual trees, pass the observation for which we want to make a prediction through them, get a prediction from every tree (summing up to N predictions) and then obtain an overall, aggregated prediction.
Bootstrapping the data and then using an aggregate to make a prediction is called Bagging, and how this prediction is made depends on the kind of problem we are facing.
For regression problems, the aggregate decision is the average of the decisions of every single decision tree. For classification problems, the final prediction is the most frequent prediction done by the forest.

## Boosting

#  Words you must know to read this

* **ensemble method** : it is a machine learning technique that combines several base models in order to produce one optimal predictive model.

* **classifier** :  it is a special case of a *hypothesis* (nowadays, often learned by a machine learning algorithm). A *classifier* is a *hypothesis* or *discrete-valued function* that is used to assign (categorical) class labels to particular data points.

* **meta-classifier** : it is the classifier that makes a final prediction among all the predictions by using those predictions as features. So, it takes classes predicted by various classifiers and pick the final one as the result that you need. 

* **weak learner** : it is a classifier that is only slightly correlated with the true classification (it can label examples better than random guessing)

  

# Boosting

### What is it? 

**Boosting** is an ensemble method. Moreover it is a a family of ML algorithms that are united by the same philosophy: turning weak learners into better learners by building a model from the training data, then creating a second model that attempts to correct the errors from the first model, then creating a third model that attempts to correct the errors from the second model and so on. Here an illustrative picture: 

<img src="https://github.com/martinabetti-97/fds/blob/jack/methods_documentation/images/pics/boost.png"> 

Because it is a family of algorithms we need to focus on one to really get into the details of the implementation, we chose the one used by the author of the project: **AdaBoost** .

### AdaBoost (Adaptive Boosting)

The main idea behind the AdaBoost algorithm is to improve the results found in one round by highlighting the mistakes to the next round. 

#### How does it work? (Outline)  

Here a sketch of the algorithm: 

<img src="https://github.com/martinabetti-97/fds/blob/jack/methods_documentation/images/pics/sketch.jpg"> 

#### An example 

Let's review the sketch code trough an example.

We want to predict from the following data if a patient has heart disease:

<img src="https://github.com/martinabetti-97/fds/blob/jack/methods_documentation/images/pics/p1.jpg"> 

Let's apply the code:

Define a weight distribution over the examples $D^1_N = \frac{1}{N}$ \
`for i = 1,2, ..., N`
<img src="https://github.com/martinabetti-97/fds/blob/jack/methods_documentation/images/pics/p2.jpg"> 

`for round j = 1 to M do `

`j = 1`

`M` is the number of times we will repeat the updating of weights and the feeding them back to the models, what we said here will be explained in more details below.

`Build a model` $h_j$  `from the training set using distribution` $D^j$

This part of the code is not really clear in the sketch of the algorithm, because what we want is to find between all the weak learners we are considering to improve trough our code, the one that has the best predictive capability. This will be explained in more details in the images below: 

With the data we are considering let's look at some really simple models the *stumps* , i.e. a node with two children, where in the node there is a condition and in the children the predictions if the condition is satisfied or not: 

<img src="https://github.com/martinabetti-97/fds/blob/jack/methods_documentation/images/pics/p3.jpg"> 

And we build models for each feature: *Blocked arteries* and *Patient weights*. As you can see above for each model we count how many correct predictions were made and how many were not: 

<img src="https://github.com/martinabetti-97/fds/blob/jack/methods_documentation/images/pics/p4.jpg"> 



Now we choose the model that performed better (i.e. the one that made more correct predictions). In our case will be the one in evidence in the image above.  

Now we calculate the *voting power* $\alpha$ (or *Amount of say*) of the best model trough the following formula (the reason behind this will be more clear below):

<img src="https://github.com/martinabetti-97/fds/blob/jack/methods_documentation/images/pics/p6.jpg"> 

<img src="https://github.com/martinabetti-97/fds/blob/jack/methods_documentation/images/pics/p7.jpg"> 

**Here the main idea of AdaBoost ** : we find the 1 incorrect prediction of the best model and we increase the weight of the sample: 

<img src="https://github.com/martinabetti-97/fds/blob/jack/methods_documentation/images/pics/p5.jpg"> 

Using the formula below: 

<img src="https://github.com/martinabetti-97/fds/blob/jack/methods_documentation/images/pics/p8.jpg"> 

<img src="https://github.com/martinabetti-97/fds/blob/jack/methods_documentation/images/pics/p9.jpg"> 



`update` $D^{j+1}$ `from` $D^j$ `:`

`Increase weights of examples misclassified by` $h_j$

<img src="https://github.com/martinabetti-97/fds/blob/jack/methods_documentation/images/pics/p10.jpg"> 

`Decrease weights of examples misclassified by` $h_j$

<img src="https://github.com/martinabetti-97/fds/blob/jack/methods_documentation/images/pics/add.jpg"> 


<img src="https://github.com/martinabetti-97/fds/blob/jack/methods_documentation/images/pics/p11.jpg"> 

And we are almost done the last part that we need to  do (that it's not specified in the `Sketch of algorithm`) is to normalize all the `new weights` found: 

<img src="https://github.com/martinabetti-97/fds/blob/jack/methods_documentation/images/pics/p12.jpg"> 

And the first iteration is complete. 

`j = 2`

In the next iteration we will repeat what we have done in this first iteration but with the new weights:

<img src="https://github.com/martinabetti-97/fds/blob/jack/methods_documentation/images/pics/p13.jpg"> 



#### How does it really work?

For the sake of completeness we will leave here the complete pseudo-code of the AdaBoost algorithm: 

<img src="https://github.com/martinabetti-97/fds/blob/jack/methods_documentation/images/pics/p14.jpg"> 



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

## Bagging  (Bootstrap Aggregation)

![bagging](https://upload.wikimedia.org/wikipedia/commons/c/c8/Ensemble_Bagging.svg)
### 1. Why?
Bagging offers the advantage of combining **weak learners to to outdo a single strong learner**. It also helps in the **reduction of variance**, hence **eliminating the over-fitting** of models in the procedure. If the base models trained on different samples have high variance (over-fitting), then the aggregated result would even it out thereby reducing the variance. \
Bootstrap Aggregation (Bagging), is a simple and very powerful **ensemble method**. An ensemble method is a technique that combines the predictions from multiple machine learning algorithms together to make more accurate predictions than any individual model. In ensemble algorithms, bagging methods form a class of algorithms which **build several instances of a black-box estimator on random subsets** of the original training set and **then aggregate their individual predictions to form a final prediction**. Bagging methods come in many flavours but mostly differ from each other by the way they draw random subsets of the training set, when **samples are drawn with replacement**.

### 2. When and how?
This technique is chosen when the base models have high variance and low bias which is generally the case with models having high degrees of freedom for complex data. As they provide a way to reduce over-fitting, bagging methods **work best with strong and complex models (e.g., fully developed decision trees),** in contrast with boosting methods which usually work best with weak models (e.g., shallow decision trees). Decision trees are sensitive to the specific data on which they are trained. When bagging with decision trees, we are less concerned about individual trees over-fitting the training data. \
Properties with different methods:

- **How Many Bootstrap Replicates?** The unbagged rate is 29.1, so we are getting most of the improvement using only 10 bootstrap replicates. More than 25 bootstrap replicates is love’s labor lost.
- **How big should learning set be?**  Same size as the initial learning set L.
- **How well on knn?** The stability of nearest neighbor classification methods with respect to perturbations of the data distinguishes them from competitors such as trees and neural nets.
- **How well on linear regression?** performance is poor if there are many small but non-zero for x coefficient.
 
###  3. Pro and cons?
The big **advantage** of bagging is that it can be **parallelised**. As the different models are fitted independently from each other, intensive parallelisation techniques can be used if required. 
One **disadvantage** of bagging is that it introduces a **loss of interpretability** of a model. The resultant model can experience lots of bias when the proper procedure is ignored. Despite bagging being highly accurate, it can be **computationally expensive** and this may discourage its use in certain instances.

### 4. Sklearn module
In scikit-learn, bagging methods are offered as a unified [`BaggingClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html#sklearn.ensemble.BaggingClassifier "sklearn.ensemble.BaggingClassifier") meta-estimator,  taking as input a user-specified base estimator along with parameters specifying the strategy to draw random subsets. 
- `base_estimator` The base estimator to fit on random subsets of the dataset.
- `n_estimator`The number of base estimators in the ensemble.
- `max_samples` and `max_features` control the size of the subsets (in terms of samples and features).
- `bootstrap` and `bootstrap_features` control whether samples and features are drawn with or without replacement.
- `oob_score=True` estimate the generalization error.
- `warm_start=True` reuse the solution of the previous call to fit and add more estimators to the ensemble
- `n_jobs=int` the number of jobs to run in parallel


## Stacking
The Stacking Classifier is an ensemble method that considers heterogeneous weak learners and combine them via a **meta-classifier** in order to improve predictions. So these output predictions are based on the multiple predictions returned by the combination of several machine learning models.


![N|stacking](http://rasbt.github.io/mlxtend/user_guide/classifier/StackingClassifier_files/stackingclassification_overview.png)

The image above summarises the process: 
* Firstly the individual classification models are trained based on the given training set; 
* Secondly the meta-classifier is fitted based on the outputs (meta-features) of the individual classification models in the ensemble;
* The output obtained at the end of the process is the final predicion.

In this project the author chose the function from the library ```mlxtend.classifier.StackingClassifier``` to implement the stacking classifier. 
In order to use this function it's needed to define: the learners to fit and the meta-model that combines them. 
As classification models to fit, the author chose the KNeighborsClassifier and the RandomForestClassifier whose predictions are combined by Logistic Regression as a meta-classifier. 

### Stacking using Cross Validation 

If we want to imporve the perfomarce of this last analysis, we can use the ```StackingCVClassifier``` from the same library.
This is an ensemble-learning meta-classifier for stacking as well but it also uses cross-validation to prepare the inputs for the level-2 classifier in order to prevent overfitting. 


![N|stackingCV](http://rasbt.github.io/mlxtend/user_guide/classifier/StackingCVClassifier_files/stacking_cv_classification_overview.png)


This method consists in the following steps:
1. The dataset is split into k folds;
2. In k successive rounds, k-1 folds are used to fit the first level classifier;
3. In each round, the first-level classifiers are then applied to the remaining 1 subset that was not used for model fitting in each iteration.

The resulting predictions are then stacked and provided as input data to the second-level classifier. After the training of the StackingCVClassifier, the first-level classifiers are fit to the entire dataset.


### 5. Useful links
original article https://link.springer.com/article/10.1023/A:1018054314350 \
youtube quick guide https://www.youtube.com/watch?v=2Mg8QD0F1dQ \
StackingCVClassifier: https://towardsdatascience.com/stacking-classifiers-for-higher-predictive-performance-566f963e4840 \
StackingCVClassifier doc: http://rasbt.github.io/mlxtend/user_guide/classifier/StackingCVClassifier/ 


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

