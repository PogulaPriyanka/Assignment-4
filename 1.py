import pandas as pd
data = pd.read_csv('E:\Assignment-4/delivery_time.csv')
data
df = data.rename({'Delivery Time':'Delivery_Time', 'Sorting Time':'Sorting_Time'}, axis = 1)
df
df.info()
df.shape
df.head()
df.tail()
df.dtypes
df.isnull().sum()
import seaborn as sns
cols = df.columns
colors = ['#000099', '#ffff00']  
sns.heatmap(df[cols].isnull(),
               cmap= sns.color_palette(colors))
df[df.duplicated()].shape
     df.describe()
     import matplotlib.pyplot as plt
plt.boxplot(df['Delivery_Time'])
import seaborn as sns
     sns.distplot(df.Delivery_Time)
     plt.boxplot(df['Sorting_Time'])
     sns.distplot(df.Sorting_Time)
     import numpy as np
fig, ax=plt.subplots(2, figsize=(6,4), sharex= False, sharey = False)
sns.boxplot(np.log(df.Sorting_Time), ax=ax[0])
sns.boxplot(np.log(df.Delivery_Time), ax=ax[1])
plt.suptitle("Log Transformation on Continuous Variables", fontsize= 17, y = 1.06)
plt.tight_layout(pad=2.0)
fig, ax=plt.subplots(2, figsize=(6,4), sharex= False, sharey = False)
sns.boxplot(np.sqrt(df.Sorting_Time), ax=ax[0])
sns.boxplot(np.sqrt(df.Delivery_Time), ax=ax[1])
plt.suptitle("Sqrt Transformation on Continuous Variables", fontsize= 17, y = 1.06)
plt.tight_layout(pad=2.0)
fig, ax=plt.subplots(2, figsize=(6,4), sharex= False, sharey = False)
sns.boxplot(np.cbrt(df.Sorting_Time), ax=ax[0])
sns.boxplot(np.cbrt(df.Delivery_Time), ax=ax[1])
plt.suptitle("Cbrt Transformation on Continuous Variables", fontsize= 17, y = 1.06)
plt.tight_layout(pad=2.0)
df.corr()
sns.pairplot(df)
import statsmodels.formula.api as smf
     model = smf.ols("Delivery_Time~Sorting_Time", data=df).fit()
     model.params
     print(model.tvalues, '\n',model.pvalues)
     model.rsquared
     sns.regplot(x = "Sorting_Time", y = "Delivery_Time", data = df)
     model.summary()
     import statsmodels.api as sm
qqplot = sm.qqplot(model.resid, line= 'q')
plt.title('Normal Q-Q Plot of Residual')
plt.show()
import numpy as np
list(np.where(model.resid>6))
def get_standardize_values( vals): 
  return (vals - vals.mean())/vals.std() 
plt.scatter(get_standardize_values(model.fittedvalues), get_standardize_values(model.resid))
plt.ylabel('Standardize Residual values')
plt.xlabel('Standardize Fitted values')
plt.title('Residaul Plot')
plt.show()
fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model, "Sorting_Time", fig=fig)
plt.show()
     model_influence = model.get_influence()
(c, _) = model_influence.cooks_distance
fig = plt.subplots(figsize=(20,8))
plt.stem(np.arange(len(df)), np.round(c,3))
plt.xlabel('Row Index')
plt.ylabel('Cooks Distance')
plt.show()
(np.argmax(c), np.max(c))
from statsmodels.graphics.regressionplots import influence_plot
influence_plot(model)
plt.show()
k = df.shape[1]
n = df.shape[0]
leverage_cutoff = 3*(k+1)/n
leverage_cutoff
newdata = pd.Series([5,10])
data_pred = pd.DataFrame(newdata, columns = ['Sorting_Time'])
data_pred
model.predict(data_pred)