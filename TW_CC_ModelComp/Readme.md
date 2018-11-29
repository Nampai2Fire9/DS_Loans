# Decision Tree and paramter tuning on Taiwan Credit Card Fraud 
Simple paramter tuning on Decision Tree. show difference vs. Logistic Regression & Naive Bayes 
Data Used: 
- Source: https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
- Time Period: April to September 2005
- 30k borrower outcomes, 24 features

After checking for null values and extreme outliers, we shorten the default name and one-hot encode the values for sex, education, and marriage. Using marriage as an example: 

```
twcc = pd.read_csv('UCI_Credit_Card.csv')
twcc.rename(columns={'default.payment.next.month':'defdq'}, inplace=True)
twcc['married'] = (twcc.MARRIAGE==1).astype('int')
```

test
