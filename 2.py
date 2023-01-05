import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.formula.api as smf
dataset=pd.read_csv('E:\Assignment-4/Salary_Data.csv')
dataset
dataset.info()
sns.distplot(dataset['YearsExperience'])
sns.distplot(dataset['Salary'])
dataset.corr()
sns.regplot(x=dataset['YearsExperience'],y=dataset['Salary'])
model=smf.ols("Salary~YearsExperience",data=dataset).fit()
model.params
model.tvalues, model.pvalues
model.rsquared , model.rsquared_adj
Salary = (25792.200199) + (9449.962321)*(3)
Salary
new_data=pd.Series([3,5])
new_data
data_pred=pd.DataFrame(new_data,columns=['YearsExperience'])
data_pred
model.predict(data_pred)
