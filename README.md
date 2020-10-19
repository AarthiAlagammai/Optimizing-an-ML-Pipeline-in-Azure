# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
This dataset is related to direct marketing campaigns of a Portuguese banking sector. The campaigns were based on phone calls.The goal is to classify whether a client will subscribe to a term deposit or not.

The best performing model was a Voting Ensemble of Xgboost classifier using standard scaler wrapper. This was found using Automl feature of Azure.

## Scikit-learn Pipeline
1.The datset is craeted using TabularDatasetfactory class
2. Preprocessing of the obtained dataset
3.Split the dataset into train and test (80:20)
4.The inverse regularization(C) and maximum iterations(max_iter) are the 2 hyperparamters used here
5.The classification algorithm used here is Logistic Regression with accuracy as the primary metric for classification

**Benifits of the parameter sampler chosen**

1.Inverse regularization parameter(C)- A control variable that retains strength modification of Regularization 
C = 1/?.The relationship, would be that lowering C - would strengthen the Lambda regulator.

2.No of iterations(max_iter):The number of times we want the learning to happen. This helps is solving high complex problems with large training hours

**Benefits of the early stopping policy**

The early stopping policy used here is Bandit Policy.Its is based on slack factor/slack amount and evaluation interval. Bandit terminates runs where the primary metric is not within the specified slack factor/slack amount compared to the best performing run.

## AutoML
AutoMl is the process of automating the time consuming, iterative tasks of machine learning model development.Traditional machine learning model development is resource-intensive, requiring significant domain knowledge and time to produce and compare dozens of models. With automated machine learning, you'll accelerate the time it takes to get production-ready ML models with great ease and efficiency.
The models were XgBoost,LightBGM,RandomForest,SGD Classifier etc. It uses crossvalidation where number of cross-validation splits to perform when validation data is not specified.


## Pipeline comparison
Accuracy of Hyperdrive-90.3641
Accuracy of Automl-91.726
Automl provided an improvement of 1% increase in accuracy. Automl has an advantage of comparing 100 of models with different values of hyperparameter setting which in hydrive setting takes a lot of time to manually specify. Automl is also highly efficient where as hyperdrive is resource intensive.
Automl proves highly useful which allows analyst and stake holders to focus on the business side of the problem statement.

## Future work

The dataset is imbalanced have to solve this using Automl  and hyperdrivefeatures and also check with different cross validation strategy for Automl

