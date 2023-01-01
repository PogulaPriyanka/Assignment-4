import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.formula.api as smf
dataset=pd.read_csv('E:\Assignment-4/delivery_time.csv')
dataset
dataset.info()
sns.distplot(dataset['Delivery Time'])
sns.distplot(dataset['Sorting Time'])
dataset=dataset.rename({'Delivery Time':'delivery_time', 'Sorting Time':'sorting_time'},axis=1)
dataset
dataset.corr()
sns.regplot(x=dataset['sorting_time'],y=dataset['delivery_time'])
model=smf.ols("delivery_time~sorting_time",data=dataset).fit()
model.params
model.tvalues , model.pvalues
model.rsquared , model.rsquared_adj
delivery_time = (6.582734) + (1.649020)*(5)
delivery_time
new_data=pd.Series([5,8])
new_data
data_pred=pd.DataFrame(new_data,columns=['sorting_time'])
data_pred
model.predict(data_pred)
