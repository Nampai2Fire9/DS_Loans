# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 13:49:07 2018
Project done in Python 3
Goal: Decision Tree Parameter Tuning comparison against Logistic Regression & Naive Baybes 
Data: Taiwan Credit Card data supplied by UCI (Kaggle) on 30k borrowers 
  Source: https://www.kaggle.com/uciml/default-of-credit-card-clients-dataset
Time Period covered: April 2005 to September 2005
Steps
 - Parameter Tuning functions
 - threshold selection and accruracy stats function
@author: etl_p
"""

import pandas as pd 
import numpy as np
from scipy import stats 
import os
import re
import matplotlib.pyplot as plt

twcc = pd.read_csv('UCI_Credit_Card.csv')
twcc.rename(columns={'default.payment.next.month':'defdq'}, inplace=True)

twcc['male_fl'] = (twcc.SEX==1).astype('int')
twcc['gradschool'] = (twcc.EDUCATION==1).astype('int')
twcc['university'] = (twcc.EDUCATION==2).astype('int')
twcc['highschool'] = (twcc.EDUCATION==3).astype('int')
twcc['married'] = (twcc.MARRIAGE==1).astype('int')
twcc.drop(['SEX','EDUCATION','MARRIAGE'], axis=1, inplace=True) ; twcc.columns

pmt_list = ['PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']
for pmt in pmt_list:
    twcc.loc[twcc[pmt]<=0,pmt]=0

metrics = pd.DataFrame(index=['accuracy','precision','recall'],
                       columns=['lr','dec','nb'])

from sklearn.model_selection import train_test_split as TTS
from sklearn.metrics import auc, accuracy_score, precision_score, recall_score, average_precision_score, roc_auc_score, roc_curve, confusion_matrix, precision_recall_curve
from sklearn.preprocessing import RobustScaler  #scales different variables to be comparable. 
from sklearn.linear_model import LinearRegression as LinReg, LogisticRegression as LogReg
from sklearn.tree import DecisionTreeClassifier as DecTree
from sklearn.ensemble import RandomForestClassifier as RFC, GradientBoostingClassifier as GBC, AdaBoostClassifier as ABC
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.cross_validation import cross_val_predict, cross_val_score, KFold 

X = twcc.drop('defdq', axis=1)
Y = twcc.defdq
robscale = RobustScaler()
X = robscale.fit_transform(X)
x_train, x_test, y_train, y_test = TTS(X, Y, random_state=123, test_size=0.2, stratify=Y)

    # Function for accuracy stats
def accuracystats_ypred(ypred,y_test):
    # Function producs accuracy, precision, recall, and confusion matrix
    #fit models and populate metrics
    acc = accuracy_score(y_pred = ypred,y_true=y_test)
    prec = precision_score(y_pred = ypred,y_true=y_test)
    recall = recall_score(y_pred = ypred,y_true=y_test)
    CM = confusion_matrix(y_pred = ypred, y_true=y_test)
        # format CM matrix
    CM = pd.DataFrame(CM)
    CM.index.name = 'Actual'
    CM.columns.name='Model'
    CM.loc['Total'] = CM.sum()
    CM['Total'] = CM.sum(axis=1)
    return acc, prec, recall, CM

 # fit Decision Tree model with no parameters
dec_mod1 = DecTree(random_state=10) 
dec_mod1.fit(x_train,y_train)
y_pred_dec = dec_mod1.predict(x_test)

    # grab accuracy stats and populate metrics table 
accdec, precdec, recdec, CMdec = accuracystats_ypred(y_pred_dec, y_test)
fpr, tpr, thresh = roc_curve(y_test, y_pred_dec)
print("AUC is %s" % "{0:.3%}".format(auc(fpr,tpr)))
metrics.loc['accuracy','dec'] = accdec
metrics.loc['precision','dec'] = precdec
metrics.loc['recall','dec'] = recdec

# Fit Logistic Regression: n_jobs=-1 makes use of all computer cores. no paramter tuning: beyond priors, not much more to be adjusted. 
lrmod = LogReg(n_jobs=-1, random_state=10)
lrmod.fit(x_train, y_train)
y_pred_lr=lrmod.predict(x_test)

acclr, preclr, reclr, CMlr = accuracystats_ypred(y_pred_lr, y_test)
metrics.loc['accuracy','lr'] = acclr
metrics.loc['precision','lr'] = preclr
metrics.loc['recall','lr'] = reclr

# Naive Bayes: No paramter tuning 
nbmod = GaussianNB()
nbmod.fit(x_train, y_train)
y_pred_nb = nbmod.predict(x_test)

accnb, precnb, recnb, CMnb = accuracystats_ypred(y_pred_nb, y_test)
metrics.loc['accuracy','nb'] = accnb
metrics.loc['precision','nb'] = precnb
metrics.loc['recall','nb'] = recnb

# compare scores across models
fig, ax = plt.subplots(figsize=(8,5))
metrics.plot(kind='barh',ax=ax, fontsize=30)
ax.legend(fontsize=30)
ax.grid

# See the Logistic Regression has best accuracy and precision. 

# Parameter Tuning: Compared to LR & NB, DecTree lends itself best to Paramter tuning. 
n_x = len(X)  # grab no. of obs. 
    #loop through min_sample_split
ms_splits = np.linspace(0.01, 1, 100, endpoint=True); ms_splits #array of values for min_sample_split parameter
prauc_vals = []
for ms in ms_splits:
    dec_mod = DecTree(min_samples_split=ms, random_state=10)
    dec_mod.fit(x_train, y_train) 
    # prauc = average_precision_score(y_true=y_test,y_score=dec_mod.predict_proba(x_test)[:,1])
    precdec, recdec, threshdec = precision_recall_curve(y_true = y_test,
                                                   probas_pred = dec_mod.predict_proba(x_test)[:,1])
    prauc = auc(recdec,precdec) 
    prauc_vals.append([ms,prauc])

prauc_vals = pd.DataFrame(prauc_vals, columns=['ms_splits','prauc']) 
        #store best min_samples_split value. use 'values' to store in np array (to remove the index)
mss_maxprauc = prauc_vals.loc[prauc_vals['prauc']==prauc_vals.prauc.max(),'ms_splits'].values[0]; mss_maxprauc   
  
fig,ax=plt.subplots(figsize=(8,5))
ax.plot(prauc_vals.ms_splits, prauc_vals.prauc, label='DecTree Min Sample Splits')
ax.set_xlabel('minimum sample splits', fontsize=15)
ax.set_ylabel('Precision Recall AUC', fontsize=15)
ax.legend()

    #loop through min_sample_leafs
ms_leaf = np.linspace(0.01, 0.5, 50, endpoint=True); #array of values for min_samples_leaf parameter
prauc_vals = []
for ms in ms_leaf:
    dec_mod = DecTree(min_samples_leaf=ms, random_state=10)
    dec_mod.fit(x_train, y_train) 
    # prauc = average_precision_score(y_true=y_test,y_score=dec_mod.predict_proba(x_test)[:,1])
    precdec, recdec, threshdec = precision_recall_curve(y_true = y_test,
                                                   probas_pred = dec_mod.predict_proba(x_test)[:,1])
    prauc = auc(recdec,precdec) 
    prauc_vals.append([ms,prauc])

prauc_vals = pd.DataFrame(prauc_vals, columns=['ms_leaf','prauc']) 
msl_maxprauc = prauc_vals.loc[prauc_vals['prauc']==prauc_vals.prauc.max(),'ms_leaf'].values[0]; msl_maxprauc

fig,ax=plt.subplots(figsize=(8,5))
ax.plot(prauc_vals.ms_leaf, prauc_vals.prauc, label='DecTree Minimum Sample Leafs')
ax.set_xlabel('minimum sample leaf')
ax.set_ylabel('Precision Recall AUC')
ax.legend()

    # Loop though max_depth: represents how deep the splitting is
max_depth = np.linspace(1, 50, 50, endpoint=True); #array of values for min_samples_leaf parameter
prauc_vals = []
for md in max_depth:
    dec_mod = DecTree(max_depth=md, random_state=10)
    dec_mod.fit(x_train, y_train) 
    #prauc = average_precision_score(y_true=y_test,y_score=dec_mod.predict_proba(x_test)[:,1])
    precdec, recdec, threshdec = precision_recall_curve(y_true = y_test,
                                                   probas_pred = dec_mod.predict_proba(x_test)[:,1])
    prauc = auc(recdec,precdec) 
    prauc_vals.append([md,prauc])

prauc_vals = pd.DataFrame(prauc_vals, columns=['max_depth','prauc']) 
mxd_maxprauc = prauc_vals.loc[prauc_vals['prauc']==prauc_vals.prauc.max(),'max_depth'].values[0]; mxd_maxprauc

fig,ax=plt.subplots(figsize=(8,5))
ax.plot(prauc_vals.max_depth, prauc_vals.prauc, label='Max Depth')
ax.set_xlabel('max_depth')
ax.set_ylabel('PR AUC')
ax.legend

    # Loop though max_features: represents how deep the splitting is.
nfeatures = X.shape[1]
max_features = list(range(1,nfeatures))  #max_features works best w/ integers. Range creates an iterator or sequence object, and lists functions converts the iterable into a list
# max_features = np.linspace(1, nfeatures, nfeatures, endpoint=True); #This function fails
prauc_vals = []
for mx in max_features:
    dec_mod = DecTree(max_features=mx, random_state=10)
    dec_mod.fit(x_train, y_train) 
    #prauc = average_precision_score(y_true=y_test,y_score=dec_mod.predict_proba(x_test)[:,1])
    precdec, recdec, threshdec = precision_recall_curve(y_true = y_test,
                                                   probas_pred = dec_mod.predict_proba(x_test)[:,1])
    prauc = auc(recdec,precdec) 
    prauc_vals.append([mx,prauc])

prauc_vals = pd.DataFrame(prauc_vals, columns=['max_features','prauc']) 
mxf_maxprauc = prauc_vals.loc[prauc_vals['prauc']==prauc_vals.prauc.max(),'max_features'].values[0]; mxf_maxprauc

#Run Decision Tree on new paramaters
   
    # fit Decision Tree model on optimized parameters
dec_mod = DecTree(min_samples_split=mss_maxprauc,min_samples_leaf=msl_maxprauc,
                  max_depth=mxd_maxprauc,max_features=mxf_maxprauc, random_state=10) 
dec_mod.fit(x_train,y_train)
y_pred_dec = dec_mod.predict(x_test)

    # grab accuracy stats and populate metrics table 
accdec, precdec, recdec, CMdec = accuracystats_ypred(y_pred_dec, y_test); accdec, precdec, recdec, CMdec
metrics.loc['accuracy','dec'] = accdec
metrics.loc['precision','dec'] = precdec
metrics.loc['recall','dec'] = recdec

# tuned up Decision Tree has higher accuracy and precision, while nbayes has better recall. compare precision_recall graphs 
precdec, recdec, threshdec = precision_recall_curve(y_true = y_test,
                                                    probas_pred = dec_mod.predict_proba(x_test)[:,1])
precnb, recnb, threshnb = precision_recall_curve(y_true = y_test,
                                                    probas_pred = nbmod.predict_proba(x_test)[:,1])
preclr, reclr, threshlr = precision_recall_curve(y_true = y_test,
                                                    probas_pred = lrmod.predict_proba(x_test)[:,1])
decprauc = auc(recdec,precdec); decprauc  # decprauc = average_precision_score(y_true=y_test, y_score=dec_mod.predict_proba(x_test)[:,1]) shows average precision, but not auc
decprtxt = ("DecTree AUC: %s" % "{0:.3%}").format(decprauc); decprtxt
nbprauc = auc(recnb, precnb); nbprauc  #nbprauc = average_precision_score(y_true=y_test,y_score=nbmod.predict_proba(x_test)[:,1]) only shows average precision, but not auc
nbprtxt = ("NaiveBayes AUC is %s" % "{0:.3%}").format(nbprauc); nbprtxt
lrprauc = average_precision_score(y_true=y_test,y_score=lrmod.predict_proba(x_test)[:,1])  
#lrprauc = auc(reclr, preclr); lrprauc
lrprtxt = ("Logistic Regression AUC is %s" % "{0:.3%}").format(lrprauc); lrprtxt

    # Compare Precision-Recall Naive Bayes vs. tuned DecTree
fig,ax=plt.subplots(figsize=(8,5))
ax.plot(recdec, precdec, label=decprtxt)
ax.plot(recnb, precnb, label=nbprtxt)
ax.set_xlabel('Recall', fontsize=15)
ax.set_ylabel('Precision', fontsize=15)
ax.set_title("Precision-Recall Curve: DecTree vs NBayes", fontsize=30)
ax.legend(fontsize=15)
ax.grid
    # Compare Precision-Recall Logistics vs. tuned DecTree
fig,ax=plt.subplots(figsize=(8,5))
ax.plot(recdec, precdec, label=decprtxt)
ax.plot(reclr, preclr, label=nbprtxt)
ax.set_xlabel('Recall', fontsize=15)
ax.set_ylabel('Precision', fontsize=15)
ax.set_title("Precision-Recall Curve: DecTree vs NBayes", fontsize=30)
ax.legend(fontsize=15)
ax.grid

    # Can see Decision Tree is a stronger fit. 
# find threshold for model by graphing thresholds agains precision & recall for DecTree
fig,ax = plt.subplots(figsize=(8,5))
ax.plot(threshdec,precdec[1:], label='Precision DEC')
ax.plot(threshdec, recdec[1:], label='Recall DEC')
ax.set_xlabel('Classification Threshold', fontsize=15)
ax.set_ylabel('Precision/Recall',fontsize=15)
ax.legend(fontsize=15)
ax.grid

accdec, precdec, recdec, CMdec = accuracystats_ypred(y_pred_dec, y_test)
accdec, precdec, recdec, CMdec          #similar scores to optimized model with no threshold. 

    # compute new y_pred with new threshold 
y_pred_dec2 = dec_mod.predict_proba(x_test)[:,1]
y_pred_dec2 = (y_pred_dec2 >= 0.2).astype('int')   #returns arrays of 1s or 0s depending on if greater than 0.2
accdec2, precdec2, recdec2, CMdec2 = accuracystats_ypred(y_pred_dec2, y_test)
accdec2, precdec2, recdec2,   CMdec2
    # check auc roc
fprd2, tprd2, threshd2 = roc_curve(y_test, y_pred_dec2)
print("AUC is %s" % "{0:.3%}".format(auc(fprd2,tprd2)))



# Logistic Regression: Improve accuracy by selecting threshold 
    # graph thresholds for Logistic Regression
fix, ax = plt.subplots(figsize=(8,5))
ax.plot(threshlr, preclr[1:], label='Precision LogReg')
ax.plot(threshlr, reclr[1:], label='Recall LogReg')
ax.set_xlabel('Classification Threshold', fontsize=30)
ax.set_ylabel('Precision/Recall LogReg', fontsize=30)
ax.set_title('Threshold vs. Precison/Recall', fontsize=30)
ax.legend(fontsize=30)
ax.grid

    # convergence close to threshold of 0.2. check accuracy at threshold of 0.2 
ypred_lr2 = lrmod.predict_proba(x_test)[:,1]
ypred_lr2 = (ypred_lr2>=0.2).astype('int')
acclr2, preclr2, reclr2, CMlr2 = accuracystats_ypred(ypred_lr2, y_test)

metrics.loc['accuracy','lr'] = acclr
metrics.loc['precision','lr'] = preclr
metrics.loc['recall','lr'] = reclr

# compare scores across models
fig, ax = plt.subplots(figsize=(8,5))
metrics.plot(kind='barh',ax=ax, fontsize=30)
ax.legend(fontsize=30)
ax.grid

fprl, tprl, threshl = roc_curve(y_test, y_pred_lr)
print("AUC is %s" % "{0:.3%}".format(auc(fprl,tprl)))
fprl2, tprl2, threshl2 = roc_curve(y_test, ypred_lr2)
print("AUC is %s" % "{0:.3%}".format(auc(fprl2,tprl2)))


