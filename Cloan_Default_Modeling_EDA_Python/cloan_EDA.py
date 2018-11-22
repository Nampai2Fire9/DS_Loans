# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 09:16:13 2017

@author: Eugene
Goal:  EDA sample and initial model fit process on personal unsecured loans 
"""

import pandas as pd
import numpy as np
import re
import os
import seaborn as sns 
import matplotlib.pyplot as plt 
import scipy.stats as stats 
from DS_functions import log_transform_append, draw_pies, draw_histograms, def_adj_rets

# pull in Personal loan data 
cloan_tape = pd.read_csv('cloan_project_201701.csv')

    # change column names:  sharing presentation to others. 
cloan_cols = ['Grade', 'Origination Date', 'Maturity Date', 'Term Months', 'Interest', 'APR', 'Origination Fee', 'Loan Amount', 'Status', 'Days Past Due', 'Payment Frequency', 'Expected term', 'Charged Off Amount', 'Charged Off Principal', 'Charged Off Date', 'Modified Flag', 'State', 'Purpose','First Payment date', 'Contract Monthly Payment', 'Principal Paid', 'Interest Paid', 'Realized Recovery', 'FICO Score', 'DTI', 'Annual Income','Employment Status',  'Inc Verification Status', 'Home Ownership','Delinquencies Past 2 Yrs', 'Accounts 30 days Delinq', 'Accounts 60 days Delinq', 'Inquiries Past 6 Months','Number of Public Records','Revolving Balance', 'Revolving Utilization', 'Credit History Length', 'PD', 'Ending Principal']
cloan_tape.columns=cloan_cols

    # create Series for default/non-default
def_dq_cloan = []
for status in cloan_tape.Status:
  if status in ['Current', 'Matured']:
    def_dq_cloan.append(0)
  else: 
    def_dq_cloan.append(1) 
def_dq_cloan = pd.Series(def_dq_cloan)
def_dq_cloan.value_counts()
cloan_tape['ChgOff_Delinq']=def_dq_cloan

n_dfdq = cloan_tape.ChgOff_Delinq.nunique()   #number of charge-offs & delinquencies
total_records = len(cloan_tape)
quantile_num = total_records/5             #number of records in one quantile  

    #-----------EDA ----------------------- 

# Univariate exploration and visualization 
    # Categorical 
        # Create Frequency tables for each variable 
cloan_cat = ['Purpose', 'Employment Status', 'Modified Flag', 'State', 'Inc Verification Status', 'Home Ownership', 'Expected term', 'Grade']
for cat in cloan_cat:
  print cloan_tape[cat].value_counts(dropna=False)
        # Replace Null values for Home Ownership & FICo Scores. 
cloan_tape['Home Ownership'].fillna('none', inplace=True)
cloan_tape['Home Ownership'].value_counts(dropna=False)
        # Grid of pie charts, setting columns & rows to 3
from DS_functions import draw_pies
draw_pies(cloan_tape, cloan_cat, 3,3)

    #-----CONTINUOUS variable visualization 
cloan_cont = ['Loan Amount', 'APR', 'FICO Score', 'DTI', 'Annual Income', 'Delinquencies Past 2 Yrs', 'Inquiries Past 6 Months','Revolving Balance', 'Revolving Utilization', 'Credit History Length']
        # Find NULL Values
for var in cloan_cont:
  print var  
  cloan_tape[var].describe()
      # Replace NULL values with average score
cloan_tape['FICO Score'].fillna(cloan_tape['FICO Score'].mean(), inplace=True)
    # draw grid of histograms:  setting 4 rows and 3 columns 
from DS_functions import draw_histograms
draw_histograms(cloan_tape, cloan_cont,4,3)

   '''------  Log-Transformation:------- 
 Loan Amount, Fico, Dti, Annual Income, Credit History Length are candidates for log transformation.
Annual Income and Revolving fields are skewed.  Try a log transform'''   
cloan_cont_transform = ['Annual Income', 'Revolving Balance', 'Revolving Utilization', 'Inquiries Past 6 Months', 'Credit History Length']
    # Append log-transformations to cloan_tape, and keep track of column names 
from DS_functions import log_transform_append
log_transform_append(cloan_tape, cloan_cont_transform)
    # cloandate Continuous variables list 
cloan_cont = ['Loan Amount', 'APR', 'FICO Score', 'DTI', 'Log Annual Income', 'Delinquencies Past 2 Yrs', 'Log Inquiries Past 6 Months','Log Revolving Balance', 'Log Revolving Utilization', 'Log Credit History Length']

    #Note: Re-running Point Serial Correlations (see below) about the same as no-log. 

#---------------------- Bivariate (vs. Defaults) Analysis--------------
    # Categorical vs Cateogorical (Defaults) Visualization: Stacked charts  
fig1=plt.figure()
for i,cat in enumerate(cloan_tape[cloan_cat].columns):
  var_pivot=pd.crosstab(cloan_tape[cat], cloan_tape.ChgOff_Delinq)
  ax=fig1.add_subplot(3,3,i+1)
  var_pivot.plot(kind='bar', stacked=True, color=['green','blue'],grid=False, ax=ax, title=cat)
    ''' States might be better done as a region:  examine later for feature engineering  
    Loan_Modified, Loan_Grade, Use Of Funds, Home Ownership, state/region look explanatory''' 

   #------- Categorical vs. Categorical Significance:  Cramer's V (Chi-Sq) 
Cloan_CramersV=[]
Cloan_crit=[]
Cloan_pvalue=[] 

    # remove state
cloan_cat2=cloan_cat
del cloan_cat2[3]
cloan_cat2

for i in cloan_cat2:
  n_uniq = cloan_tape[i].nunique()
  cloan_pivot = pd.crosstab(cloan_tape[i], cloan_tape.ChgOff_Delinq, margins=True)
  expected=np.outer(cloan_pivot['All'][0:n_uniq], cloan_pivot.ix['All'][0:n_dfdq])/total_records
  observed = cloan_pivot.ix[0:n_uniq, 0:n_dfdq]   
  chisq_tst = (((observed-expected)**2)/expected).sum().sum() 
  print(chisq_tst)
  df_dq = n_uniq + n_dfdq - 1    #degrees of freedom
  crit=stats.chi2.ppf(q=0.95, df=df_dq)
  Cloan_crit.append(crit)
  dq_pvalue = 1 - stats.chi2.cdf(x=chisq_tst, df=df_dq)
  Cloan_pvalue.append(dq_pvalue)
  cramersV = np.sqrt((chisq_tst/total_records)/min(n_uniq, n_dfdq))
  Cloan_CramersV.append(cramersV)
Cloan_CramersV
Cloan_crit
Cloan_pvalue 

Cloan_CramersV = pd.DataFrame(Cloan_CramersV)
Cloan_CramersV['fields']=cloan_cat2

cloan_cols2 = ['correlation','fields']
Cloan_CramersV.columns = cloan_cols2
Cloan_CramersV.columns

sns.barplot(x='fields', y='correlation', data=Cloan_CramersV, orient='v')
sns.plt.title("Categorical Correlations")

  # Continuous versus categorical Visualization: Boxplots 
fig=plt.figure()
for i, var_name in enumerate(cloan_tape[cloan_cont].columns):
  ax=fig.add_subplot(4,3,i+1)
  sns.boxplot(x='ChgOff_Delinq', y=var_name, data=cloan_tape)


  # Continuous vs. categorical Significance: Point Biserial Correlation
point_corr = []
for i in cloan_cont:
  ps_corr = stats.pointbiserialr(cloan_tape[i], cloan_tape.ChgOff_Delinq)
  point_corr.append(ps_corr)
print point_corr 
point_corr_df = pd.DataFrame(point_corr)
point_corr_df['field']=cloan_cont   
point_corr_df  
point_corr_df=point_corr_df.sort_values('correlation', ascending=False)
       # plot correlations 
plt.close()
sns.set()
b = sns.barplot(x='field', y='correlation', data=point_corr_df, orient='v')
# b.axes_style(text.color='.20') 
b.set_xlabel("Field Name", fontsize=15)
b.set_ylabel("Mean Correlation", fontsize=15) 
b.tick_params(labelsize=10)
b.axes.set_title("Point Serial Correlations", fontsize=20) 

   ''' Explanatory variables seem to be APR, Int Rate, Inquiries in past 6 months, and maybe DTI. 
   Distribution on Num Inquiries past 6 months is slightly diff btn Default & Non-Default. ''' 
cloan_tape['Inquiries Past 6 Months'][cloan_tape.ChgOff_Delinq==1].describe()
cloan_tape['Inquiries Past 6 Months'][cloan_tape.ChgOff_Delinq==0].describe()
   # Num_Inquiries:  50th & 75th percentile on defaults is 1 & 2 on defaults, 0 & 1 for non-defaults.'''    


''' ML/Predictive Modeling.  Run 10-fold CV using multiple classification models.
Identify important variables and variable relationships '''
# make copies of cloan_tape

#---------- Classification Model Setup ----------------- 
    # Convert categorical variables to numerical format 
cloan_tape_num = cloan_tape.copy()   # make copy just for classification method 
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for i in cloan_cat:
  cloan_tape_num[i]=le.fit_transform(cloan_tape_num[i])
cloan_tape_num.dtypes

from sklearn.linear_model import LinearRegression as LinReg
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.ensemble import RandomForestClassifier as RFC 
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.ensemble import AdaBoostClassifier as ABC
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn import svm
from sklearn.cross_validation import cross_val_predict
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold 
from sklearn import metrics

from DS_functions import classification_scores

# create feature list
cloan_feature_list=['APR', 'FICO Score','Inquiries Past 6 Months','Annual Income','DTI','Loan Amount','Purpose','Home Ownership','Grade','Modified Flag']
cloan_response='ChgOff_Delinq'
    # fit Logistic Regression 
model=LogReg()
classification_scores(model, cloan_tape_num, cloan_feature_list, cloan_response,2)
        # Exercise for later:  Run forward stepwise model selection 
    # fit Random Forest
model=RFC(n_estimators=1000)
classification_scores(model, cloan_tape_num, cloan_feature_list, cloan_response,3)
       # Create importance chart
imptree_features = {'field':cloan_feature_list, 'importance':model.feature_importances_}
imptree_features_df = pd.DataFrame(imptree_features).sort_values('importance', ascending=False)
var_imp_graph = sns.barplot(x='importance', y='field', data=imptree_features_df)
var_imp_graph.axes_style({'text_color':0.2})
var_imp_graph.axes.set_title('Variable Importance: Mean Decreasing Impurity', fontsize= 20) 
var_imp_graph.set_xlabel('Mean Importance', fontsize=15)
var_imp_graph_set_ylabel('Feature', fontsize=15)
sns.plt.title("Variable Importance:  Mean Decreasing Impurity")

sns.set()
b = sns.barplot(x='field', y='correlation', data=point_corr_df, orient='v')
# b.axes_style(text.color='.20') 
b.set_xlabel("Field Name", fontsize=15)
b.set_ylabel("Mean Correlation", fontsize=15) 
b.tick_params(labelsize=10)
b.axes.set_title("Point Serial Correlations", fontsize=20) 

    # fit K-Nearest Neighbors ------
model=KNC(n_neighbors=3)
classification_scores(model, cloan_tape_num, cloan_feature_list, cloan_response,3)
    # Fit Scloanport vector machines:  Linear Kernel  
model=svm.SVC()
classification_scores(model, cloan_tape_num, cloan_feature_list, cloan_response,3)
    # Boosting:  Gradient Boosting 
model=GBC(n_estimators=300, learning_rate=0.3, max_depth=1, random_state=0)
classification_scores(model, cloan_tape_num, cloan_feature_list, cloan_response,3)
    # Boosting:   AdaBoost
model=ABC(n_estimators=300, learning_rate=0.3)
classification_scores(model, cloan_tape_num, cloan_feature_list, cloan_response,3)
#try different learning rates:  slows the fitting speed to prevent overfitting.     

