#!/usr/bin/env python
# coding: utf-8

# Importing required libraries

# In[1]:


import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from scipy import stats
import missingno as msno
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
#from google.colab import drive


# Loading the dataset

# In[2]:


# from google.colab import drive
#drive.mount('/content/drive')


# In[3]:


aparts_df = pd.read_csv('rent_apts.csv')


# Exploring the dataset

# In[4]:


#Shape of the dataset, rows and columns respectively.
aparts_df.shape


# In[5]:


aparts_df.info()


# # Data cleaning and wrangling

# In[6]:


#first we shall remove the KSh sign and the comma (KSh 50,000)
aparts_df['Price'].str.replace('KSh','',regex=True).str.replace(',','')


# In[7]:


#the price column is an object type, we shall convert it to integer
aparts_df['Price'] = aparts_df['Price'].str.replace('KSh','',regex=True).str.replace(',','').astype(int)


# In[8]:


aparts_df.info()


# In[9]:


aparts_df.describe().T


# In[10]:


#Check for null values
aparts_df.isnull().sum()


# In[11]:


#We can see that the bathroom column has quite a number of missing values
#Lets first remove rows(houses) that dont have sq_mtrs and bedrooms
aparts_df.dropna(subset=['sq_mtrs','Bedrooms'],inplace=True)


# In[12]:


aparts_df.head()


# In[13]:


#check for houses that have missing bathrooms as null values
aparts_df[aparts_df['Bathrooms'].isnull()]


# In[16]:


#Let's check the correlation
#we see the bathroom column has a good correlation with the price column
#aparts_df.corr()
# Calculate the correlation coefficient between 'bathrooms' and 'price'
corr1 = aparts_df['Bedrooms'].corr(aparts_df['Price'])
corr2 = aparts_df['Bathrooms'].corr(aparts_df['Price'])
corr3 = aparts_df['sq_mtrs'].corr(aparts_df['Price'])

# Print the correlation coefficient
print('Correlation between Bedrooms and price:', corr1)
print('Correlation between bathrooms and price:', corr2)
print('Correlation between sq_mtrs and price:', corr3)


# In[17]:


#We shall group the houses by the number of bedrooms and fill the missing values
#with the mean of the bathrooms in each group, rounded to whole numbers
aparts_df.groupby('Bedrooms')['Bathrooms'].transform(lambda x: x.fillna(round(x.mean())))


# In[18]:


aparts_df['Bathrooms'] = aparts_df.groupby('Bedrooms')['Bathrooms'].transform(lambda x: x.fillna(round(x.mean())))


# In[19]:


#Recheck for missing values
aparts_df.isnull().sum()


# In[20]:


#Notice that we won't need some of the columns to train our model; those are Agency, Neighborhood, and links.
#You also notice that sq_mtrs column is not correlated with any of the columns in our data.
#Remove non-required columns
del aparts_df["Agency"]
del aparts_df["Neighborhood"]
del aparts_df["link"] 
del aparts_df["sq_mtrs"]


# # Visualising the data
# 
# 
# 

# In[21]:


#Lets first check the distribution of the price column
sns.displot(aparts_df['Price'])


# In[22]:


#lets check the houses that are outliers (200000 and above))
aparts_df[aparts_df['Price']>=200000]


# In[23]:


#scatterplot for price  and bedrooms by bathrooms
sns.scatterplot(x='Bedrooms',y='Price',data=aparts_df,hue='Bathrooms')


# In[24]:


#we notice that the houses  are expensive when they have more bedrooms and bathrooms
#Lets check the scatterplot of the sq_mtrs column with price
#sns.scatterplot(x='sq_mtrs',y='Price',data=aparts_df)


# In[25]:


#Lets drop houses with 0 sq_mtrs, for more accurate data
#aparts_df.drop(aparts_df[aparts_df['sq_mtrs']==0].index,inplace=True)


# In[26]:


#generate a pairplot on price, bedrooms, bathrooms and sq_mtrs
sns.pairplot(aparts_df[['Price','Bedrooms','Bathrooms']])


# ## Conclusion
# The houses with more bedrooms and bathrooms are more expensive
# 

# In[27]:


aparts_df.info()


# # Set features and labels

# In[28]:


X = aparts_df[["Bedrooms", "Bathrooms"]]
y = aparts_df[["Price"]]


# # Define the model and train it

# In[29]:


#Spliting the data into training and testing the set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 20)
#Train the regression model using the training data 
clf = DecisionTreeRegressor()
clf.fit(X_train, y_train)


# # Prediction and accuracy

# In[30]:


#Predictions using the testing set 
y_pred = clf.predict(X_test)

# #Example of few predictions
# print("Making predictions for the following 5 houses:")
# print(X.head())
# print("The predictions are")
# five_pred= clf.predict(X_test)[:5]

# five_pred


# In[31]:


print(X_test.shape)
print(y_test.shape)
print(y_pred.shape)


# In[32]:


# print("Making predictions for the following 5 houses:")
# print(X.head())
# print("The predictions are")
# str(y_test[:5])


# In[38]:


# #Checking the accuracy of the model using MSE,MAE and R-squared error

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print('Mean squared error: ', mean_squared_error(y_test, y_pred))
print("Root Mean Squared error", np.sqrt(mean_squared_error(y_test, y_pred)))
print('Mean absolute error: ', mean_absolute_error(y_test, y_pred))
print('R-squared score: ', r2_score(y_test, y_pred))


# In[39]:


# trying new model
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Always scale the input. The most convenient way is to use a pipeline.
clf = make_pipeline(StandardScaler(),
                     SGDRegressor(max_iter=1000, tol=1e-3, loss="squared_error"))


# In[35]:


#Checking the accuracy of a model
clf.fit(X_train, y_train)
forestPred = clf.predict(X_test)
forestScores = clf.score(X_test, y_test)
forestScores


# In[40]:


mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
rmse


# In[41]:


#using a scatter plot to visualize how well the model is perfoming
import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred)
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.title('Actual vs Predicted values')
plt.show()


# In[ ]:


# from sklearn.model_selection import cross_val_score
#  #Cross-validate the model
#  #Perform cross-validation on the model
# scores = cross_val_score(clf, X, y, cv=5, scoring='neg_mean_squared_error')
# rmse_scores = np.sqrt(-scores)

#  # Display the cross-validation scores
# print('Cross-Validation Scores:', rmse_scores)
# print('Mean:', rmse_scores.mean())
# print('Standard deviation:', rmse_scores.std())


# In[ ]:

import joblib
joblib.dump(clf, 'house_price_prediction.joblib')