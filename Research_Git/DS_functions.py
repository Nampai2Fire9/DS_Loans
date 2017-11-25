# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 09:55:24 2017

@author: Eugene
"""

import pandas as pd
import numpy as np
import re
import os
import seaborn as sns 
import matplotlib.pyplot as plt 
import scipy.stats as stats 

# Create grid of pie charts:  for categorical variables 
def draw_pies(df, variables, n_rows, n_cols):
  fig=plt.figure()
  for i, var_name in enumerate(variables):
      ax=fig.add_subplot(n_rows,n_cols,i+1)
      tbl = df[var_name].value_counts()
      ax.pie(tbl, labels=tbl.index)
      ax.set_title(var_name)
  fig.tight_layout() 
  plt.show()
  
# Create grid of histograms for continuous variables:  
def draw_histograms(df, variables, n_rows, n_cols):
    fig=plt.figure()
    for i, var_name in enumerate(variables):
        ax=fig.add_subplot(n_rows,n_cols,i+1)
        df[var_name].hist(bins=10,ax=ax)
        ax.set_title(var_name+" Distribution")
    fig.tight_layout()  # Improves appearance a bit.
    plt.show()

# Function to transform columns to logs and append to df 

def log_transform_append(df, log_cols):
  # Transform columns from log_cols into logs and create new column
  log_col_name = []
  for col in log_cols:
    col_name = 'Log ' + col
    print col_name
    log_feature = np.log(df[col])
    print log_feature
    df[col_name]=log_feature
    log_col_name.append(col_name)     
  #  replace infinite values with zero 
  for log_col in log_col_name:
    df[log_col] = df[log_col].replace('-inf',0) 

# Function to compute PD, APR, and default-adjusted returns, and graph latter  

def def_adj_rets(df, cont_features, up_features, percentile):
  defdq_rate = []
  perc=[]
  total_records=len(df)
  print(percentile, type(percentile))
  if percentile==25:
    perc_index=4
  if percentile==50:
    perc_index=5
  if percentile==75:
    perc_index=6
  if percentile==100:
    perc_index=7
  down_perc_index = 10-perc_index  
  print perc_index, ' ', down_perc_index 
  for var in cont_features: 
     # set feature value cutoff.  First focus on variables where you want score to be higher (up_cont_wantup)  
    if var in up_features:
      perc_value=df[var].describe()[perc_index]
      num_defaults = float(df.ChgOff_Delinq[df[var]>=perc_value].value_counts()[1])
      Avg_APR = np.sum(df.APR[df[var]>=perc_value]) / len(df[df[var]>=perc_value])
      def_rate = '{percent:.2%}'.format(percent=num_defaults/len(df.ChgOff_Delinq[df[var]>=perc_value]))
      def_rate_float = num_defaults/len(df.ChgOff_Delinq[df[var]>=perc_value])
      factor_posneg='positive'
      print var, ' ', perc_value, ' ', num_defaults, ' ', Avg_APR, ' ', def_rate, ' ', def_rate_float
    else: 
      perc_value=df[var].describe()[down_perc_index]   
      num_defaults = float(df.ChgOff_Delinq[df[var]<=perc_value].value_counts()[1])
      Avg_APR = np.sum(df.APR[df[var]<=perc_value]) / len(df[df[var]<=perc_value])
      def_rate = '{percent:.2%}'.format(percent=num_defaults/len(df.ChgOff_Delinq[df[var]<=perc_value]))
      def_rate_float = num_defaults/len(df.ChgOff_Delinq[df[var]<=perc_value])
      factor_posneg='negative'      
      print perc_value, ' ', var, ' ', num_defaults, ' ', Avg_APR, ' ', def_rate, ' ', def_rate_float
    def_adj_ret = Avg_APR - def_rate_float*100
    defdq_rate.append([def_rate, Avg_APR, def_adj_ret, factor_posneg])
    perc.append(perc_value)
  defdq_rate = pd.DataFrame(defdq_rate)
  defdq_rate['Fields']=cont_features
  defdq_rate['Cutoff Value']=perc
  defdq_rate.columns=['Default Rate', 'APR', 'Default Adjusted Return', 'PosNeg Factor', 'Fields', 'Cutoff']
    # add in line for total dataset 
  total_defrate = '{percent:.2%}'.format(percent=float(df.ChgOff_Delinq.value_counts()[1])/total_records)
  total_defrate_float = float(df.ChgOff_Delinq.value_counts()[1])/total_records
  tot_avg_APR = np.mean(df.APR)
  tot_defadj_APR = tot_avg_APR - total_defrate_float*100
  len_defdq = len(defdq_rate)
  defdq_rate.loc[len_defdq] = [total_defrate, tot_avg_APR, tot_defadj_APR,'NA','All', 0]
  return defdq_rate
  
# Generate 5-fold cross-validation & accuracy scors for multiple models.  
  
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.ensemble import RandomForestClassifier as RFC 
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold 
from sklearn import metrics

def classification_scores(model, train, features, response, folds):
  model.fit(train[features], train[response])
    # predict response on training set   
  ytrain_pred = model.predict(train[features])
  accuracy = metrics.accuracy_score(ytrain_pred, train[response])
  print "Accuracy: %s " % "{0:.3%}".format(accuracy)

    # run k-fold cross-validation w/ k folds 
  kf=KFold(train.shape[0], shuffle=True, n_folds=folds)
  cv_pred = cross_val_predict(model, train[features], train[response], cv=kf)
  # print "mean Full-set AUC is %s" % "{0:.3%}".format(metrics.roc_auc_score(train[response], cv_pred))
  fpr, tpr, thresh = metrics.roc_curve(train[response], cv_pred)
  print "Full-set AUC is %s" % "{0:.3%}".format(metrics.auc(fpr,tpr))
  confusion=metrics.confusion_matrix(train[response], cv_pred)
  TP=confusion[1,1]
  TN=confusion[0,0]
  FP=confusion[0,1]   
  FN=confusion[1,0]
  print "TP, FP, FN, TN is ", TP,' ', FP,' ', FN,' ', TN 
  plt.plot(fpr, tpr)
  accuracy=[]
  exp_acc=[]
  aucr=[]
  for trn, test in kf:
      # filter training set 
    trn_features=(train[features].iloc[trn,:])
     # isolate response from training set 
    trn_response = train[response].iloc[trn]
     # train fold with predictors & target 
    model.fit(trn_features, trn_response)
    y_pred = model.predict(train[features].iloc[test,:])
        # expected accuracy:  % of 1s in test set 
    exp_acc.append(max(train[response].iloc[test].mean(), 1-train[response].iloc[test].mean()))
    ffpr,ttpr,thold = metrics.roc_curve(train[response].iloc[test], y_pred)
    # adding AUCs for each fold
    aucr.append(metrics.auc(ffpr,  ttpr))
    print "mini AUC is ", metrics.auc(ffpr,  ttpr)
    accuracy.append(model.score(train[features].iloc[test,:], train[response].iloc[test]))
  print "mean AUC is %s" % "{0:.3%}".format(np.mean(aucr))
  print "Expected Accuracy: %s" % "{0:.3%}".format(np.mean(exp_acc))
  print "Cross-Val Mean Accuracy %s" % "{0:.3%}".format(np.mean(accuracy))
    # refit model to be used again 
  model.fit(train[features], train[response])


