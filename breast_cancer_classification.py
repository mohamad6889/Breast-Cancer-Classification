#!/usr/bin/env python
# coding: utf-8

# # Breast Cancer Classification
# 
# Data Set is Available Below:
# https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)

# # Import Needed Packages

# In[1]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn


# # Import Data Set from Sklearn 

# In[2]:


from sklearn.datasets import load_breast_cancer
breast_cancer = load_breast_cancer()


# In[3]:


breast_cancer.keys()


# # Creating the Data Frame

# In[4]:


df_breast_cancer = pd.DataFrame(np.c_[breast_cancer['data'], breast_cancer['target']], columns = np.append(breast_cancer['feature_names'], ['target']))


# In[5]:


df_breast_cancer.head()


# # Data Visualizing

# In[6]:


seaborn.pairplot(df_breast_cancer, hue = 'target', vars = ['mean radius', 'mean texture', 'mean area', 'mean perimeter', 'mean smoothness'])


# In[7]:


seaborn.scatterplot(x = 'mean area', y = 'mean smoothness', hue = 'target', data = df_breast_cancer)


# In[8]:


plt.figure(figsize=(30, 30))
seaborn.heatmap(df_breast_cancer.corr(), annot=True)


# # Model Training

# In[9]:


X = df_breast_cancer.drop(['target'], axis=1)


# In[10]:


X.head()
# Target Column Has been Droped(Independent Vars)


# In[11]:


Y = df_breast_cancer['target']


# In[12]:


Y.head()
# Dependent Var(Target)


# In[15]:


from sklearn.model_selection import train_test_split


# In[16]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state=5)


# In[17]:


print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[19]:


from sklearn.svm import SVC 
from sklearn.metrics import classification_report, confusion_matrix


# In[20]:


svc_model = SVC()
svc_model.fit(X_train, Y_train)


# # Model Evaluation

# In[21]:


Y_predict = svc_model.predict(X_test)
conf_mat = confusion_matrix(Y_test, Y_predict)


# In[23]:


seaborn.heatmap(conf_mat, annot=True)
# as you see we have 7 type I error
#not bad but it can be better


# In[24]:


print(classification_report(Y_test, Y_predict))


# # Model Improving

# In[41]:


min_X_train = X_train.min()
range_X_train = (X_train - min_X_train).max()
X_train_per_unit = (X_train - min_X_train)/range_X_train


# In[42]:


X_train_per_unit


# In[43]:


min_X_test = X_test.min()
range_X_test = (X_test - min_X_test).max()
X_test_per_unit = (X_test - min_X_test)/range_X_test


# In[30]:


param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']}


# In[31]:


from sklearn.model_selection import GridSearchCV


# In[32]:


grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=4)


# In[44]:


grid.fit(X_train_per_unit,Y_train)


# In[45]:


grid.best_params_


# In[46]:


grid.best_estimator_


# In[47]:


grid_predictions = grid.predict(X_test_per_unit)


# In[48]:


conf_mat = confusion_matrix(Y_test, grid_predictions)


# In[49]:


seaborn.heatmap(conf_mat, annot=True)
# reducing 7 typeI error to 4


# In[50]:


print(classification_report(Y_test, Y_predict))


# In[ ]:




