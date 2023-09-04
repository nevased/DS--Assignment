#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.regressionplots import influence_plot
import statsmodels.formula.api as smf
from sklearn import linear_model
from sklearn.metrics import r2_score
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# In[13]:


start=pd.read_csv('C:\\Users\\Dell\Downloads\\50_Startups.csv')
start


# In[14]:


start['State'].value_counts().plot.bar() ##Bar plot


# In[15]:


startup = start.drop('State', axis = 1)
startup.head(7)


# In[16]:


startup.isnull().sum()


# In[17]:


startup.info()


# In[18]:


startup.describe()


# In[19]:


startup.isna().sum()


# In[20]:


startup.corr()   #Correlation


# In[21]:


sns.set_style(style='darkgrid')  #Pair plot
sns.pairplot(startup)


# In[22]:


fig, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(startup.corr(), cmap='magma', annot=True, fmt=".3f")   #Heatmap fmt=".no. of decimals" 


# In[23]:


f, axes = plt.subplots(2, 2, figsize=(12,8))

sns.regplot(x = 'Profit', y = 'R&D Spend', data = startup, scatter_kws={'alpha':0.6}, ax = axes[0,0])
sns.regplot(x = 'Profit', y = 'Administration', data = startup, scatter_kws={'alpha':0.6}, ax = axes[0,1])
sns.regplot(x = 'Profit', y = 'Marketing Spend', data = startup, scatter_kws={'alpha':0.6}, ax = axes[1,0])


# # Model Building

# In[24]:


## Usking Sklearn 


# In[33]:


X = startup.drop(['Profit'], axis = 1).values        
y = startup.iloc[:, 3].values.reshape(-1,1)
X


# In[34]:


model = linear_model.LinearRegression() 
model.fit(X,y)


# In[35]:


model.rank_


# In[36]:


model.coef_


# In[37]:


model.intercept_


# In[38]:


r2_score(y,model.predict(X))


# # Prediction

# In[39]:


startup_new=pd.DataFrame({"R&D Spend":152200,"Administration":155300,"Marketing Spend":472000},index=[1]) 
model.predict(startup_new)
print("The profit will be:",model.predict(startup_new)) 	


# In[40]:


# Using stats.ols


# In[41]:



d1=startup.rename({'Marketing Spend':'Marketing_Spend'},axis=1)
startups=d1.rename({'R&D Spend':'RandD_Spend'},axis=1)
startups.head(2)


# In[42]:


module = smf.ols("Profit~RandD_Spend+Administration+Marketing_Spend",data = startups).fit()
module.summary()


# # Test for Normality of Residuals (Q-Q Plot)

# In[43]:


import statsmodels.api as sm
qqplot=sm.qqplot(module.resid,line='q') # line = 45 to draw the diagnoal line


# # Residual Plot for Homoscedasticity

# In[44]:


def std( vals ):                      #Loop created for return values.
    return (vals - vals.mean())/vals.std()

plt.scatter(std(module.fittedvalues),
            std(module.resid))


# # Residual Vs Regressors

# In[45]:


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(module, "RandD_Spend", fig=fig)
plt.show()


# In[46]:


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(module, "Administration", fig=fig)
plt.show()


# In[47]:


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(module, "Marketing_Spend", fig=fig)
plt.show()


# # Model Deletion Diagnostics

# # Detecting Influencers/Outliers

# In[48]:


module_influence = module.get_influence()
(c, _) = module_influence.cooks_distance


# In[49]:


influence_plot(module)
plt.show()


# In[50]:


#index and value of influencer where c is more than .5
(np.argmax(c),np.max(c))


# In[51]:


(module.rsquared,module.aic)


# In[52]:


module.params


# In[ ]:





# In[ ]:





# In[ ]:




