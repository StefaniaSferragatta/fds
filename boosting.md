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

![p4](C:\Users\momor\OneDrive\Desktop\p4.jpg)

#### An example 

Let's review the sketch code trough an example.

We want to predict from the following data if a patient has heart disease:

​												![p1](C:\Users\momor\OneDrive\Desktop\notebooks\boosting\pics\p1.jpg) 

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



