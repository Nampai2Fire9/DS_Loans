# Decision Tree and paramter tuning on Taiwan Credit Card Fraud 
Simple paramter tuning on Decision Tree. show difference vs. Logistic Regression & Naive Bayes 
Data Used: 
- Source: https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
- Time Period: April to September 2005
- 30k borrower outcomes, 24 features

Libraries used in this example include the following: 

```
import pandas as pd 
import numpy as np
from scipy import stats 
import os
import re
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split as TTS
from sklearn.metrics import auc, accuracy_score, precision_score, recall_score, average_precision_score, roc_auc_score, confusion_matrix, precision_recall_curve
from sklearn.preprocessing import RobustScaler  #scales different variables to be comparable. 
from sklearn.linear_model import LinearRegression as LinReg, LogisticRegression as LogReg
from sklearn.tree import DecisionTreeClassifier as DecTree
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import cross_val_predict, cross_val_score, KFold 
```

After checking for null values and extreme outliers, we shorten the default name and one-hot encode the values for sex, education, and marriage. Using marriage as an example: 

```
twcc = pd.read_csv('UCI_Credit_Card.csv')
twcc.rename(columns={'default.payment.next.month':'defdq'}, inplace=True)
twcc['married'] = (twcc.MARRIAGE==1).astype('int')
```

A quick scan of the data shows the label/response variable for default is skewed about 75-25% in favor on non-default, which is very common. While this is actually a very high default rate, the response here is still considered imbalanced, which will have implications for how we measure the performance of our models. 

<img src="image/defdq_value_counts_201811.PNG" width="50"> 

In addition to one-hot encoding, we scale the continuous variables through the 'RobustScaler' package from `sklearn.preprocessing`. For illustration purposes, we will run a simple train-test split. To further ensure the robustness of the model, we would use k-fold cross validation (packages included above). 

```
X = twcc.drop('defdq', axis=1)
Y = twcc.defdq
robscale = RobustScaler()
X = robscale.fit_transform(X)
x_train, x_test, y_train, y_test = TTS(X, Y, random_state=123, test_size=0.2, stratify=Y)
```

After fitting logistic regression, naive Bayes, and a decision tree on default parameters, we compare the accuracy results.

<img src="image/accuracy_scores1_201811.png" width="50"> 

Here we can see logistic regression having the best accuracy and precision rates, while recall is highest under Naive Bayes. Options for tuning the the logistic regression and naive bayes are limited so we focus on tuning our Decision Tree model. 


