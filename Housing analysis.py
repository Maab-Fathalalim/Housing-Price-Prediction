#!/usr/bin/env python
# coding: utf-8

# Housing Price Prediction(linear Rrgresssion)
# 
# Problem Statement:
# 
# Consider a real estate company that has a dataset containing the prices of properties in the Delhi region. It wishes to use the data to optimise the sale prices of the properties based on important factors such as area, bedrooms, parking, etc.
# Essentially, the company wants to identify the variables affecting house prices, e.g. area, number of rooms, bathrooms, etc.
# 
# To create a linear model that quantitatively relates house prices with variables such as number of rooms, area, number of bathrooms, etc.
# 
# To know the accuracy of the model, i.e. how well these variables can predict house prices.

# In[1]:


import numpy as nd
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
#Traing and testing
import sklearn.linear_model
from sklearn.model_selection import train_test_split
#Development
from sklearn.linear_model import LinearRegression
linear_regression_model =LinearRegression()
#Evaluation
import sklearn.metrics 
from sklearn.metrics import accuracy_score,r2_score
from sklearn.metrics import mean_squared_error,mean_absolute_error


# In[8]:


data =pd.read_csv("C:/Users/Oooba/Desktop/Analysis with pyhton/Housing predictive/Housing.csv")
data


# In[9]:


data.info()


# In[10]:


data.isna().sum()


# In[12]:


data.duplicated().sum()


# In[13]:


data.describe()


# In[37]:


data.boxplot(column=['price'])
plt.xticks(rotation=45)
plt.show()

data.boxplot(column=['area'])
plt.xticks(rotation=45)
plt.show()

data.boxplot(column=['bedrooms'])
plt.xticks(rotation=45)
plt.show()

data.boxplot(column=['bathrooms'])
plt.xticks(rotation=45)
plt.show()

data.boxplot(column=['stories'])
plt.xticks(rotation=45)
plt.show()

data.boxplot(column=['parking'])
plt.xticks(rotation=45)
plt.show()


# In[44]:


data['mainroad'] = data['mainroad'].replace({'yes': 1, 'no': 0})
data['guestroom'] = data['guestroom'].replace({'yes': 1, 'no': 0})
data['basement'] = data['basement'].replace({'yes': 1, 'no': 0})
data['hotwaterheating'] = data['hotwaterheating'].replace({'yes': 1, 'no': 0})
data['airconditioning'] = data['airconditioning'].replace({'yes': 1, 'no': 0})
data['prefarea'] = data['prefarea'].replace({'yes': 1, 'no': 0})
data


# In[45]:


data["furnishingstatus"].unique()


# In[51]:


data_encoded = pd.get_dummies(data, columns=['furnishingstatus'])
data_encoded


# In[56]:


data_corr= data.corr()
color=sns.color_palette("coolwarm",as_cmap=True)
sns.heatmap(data_corr,cmap=color,annot=True,fmt="0.1f",linewidth=0.5)


# In[66]:


data_x =data_encoded.drop(columns=['price','area']) 
x =data_x
y= data_encoded['price']


# In[67]:


x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=42)
x_train.shape #80%


# In[70]:


model=linear_regression_model.fit(x_train,y_train)
model.fit(x_train,y_train)


# In[69]:


y_prede=model.predict(x_test)
y_error= y_test-y_prede
predection=pd.DataFrame({"Actual":y_test,"predicted":y_prede,"Error":y_error})
predection["abs_error"]=abs(predection["Error"])
mean_absolut_error=predection["abs_error"].mean()
predection.head(10)


# In[76]:


plt.figure(figsize=(10, 6))
sns.scatterplot(x='Actual', y='predicted', data=predection, hue='Error', palette='coolwarm')
plt.xlabel('Actual')
plt.ylabel('Prediction')
plt.title('Actual vs Prediction with Error')
plt.legend(title='Error')
plt.show()


# In[77]:


r2_score(y_test,y_prede)
print(f"Accuracy of the model={round(r2_score(y_test,y_prede)*100)}%")


# In[78]:


print("Root Mean Squared Error (RMSE)=",mean_absolut_error**(0.5))


# In[85]:


model_cof=model.coef_
plt.plot(model_cof,color="b",marker="+",markersize=12,alpha=0.4)
plt.title("Cofficient of Model")

