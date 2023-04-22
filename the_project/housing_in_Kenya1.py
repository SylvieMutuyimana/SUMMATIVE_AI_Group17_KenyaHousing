#!/usr/bin/env python
# coding: utf-8

# Importing required libraries

# In[1]:


import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import joblib


# Loading the dataset

# In[3]:


aparts_df = pd.read_csv('rent_apts.csv')
aparts_df.head()


# Exploring the dataset

# In[4]:


#Shape of the dataset, rows and columns respectively.
aparts_df.shape


# In[5]:


aparts_df.info()


# In[6]:


#we see that the dataset contains 1848 entries, 
#the bathrooms column has some null values but we shall deal with that later
aparts_df.columns
#we can see that there are 3 columns with missing values
aparts_df.isnull().sum()


# In[7]:


#the bathroom column has 291 missing values but we shall deal with it later

#describe the data
aparts_df.describe().T


# # Data cleaning and wrangling

# In[8]:


#the price column is an object type, we shall convert it to float
#first we shall remove the KSh sign and the comma (KSh 50,000)
aparts_df['Price'].str.replace('KSh','',regex=True).str.replace(',','')


# In[9]:


aparts_df['Price'] = aparts_df['Price'].str.replace('KSh','',regex=True).str.replace(',','').astype(float)


# In[10]:


aparts_df.info()


# In[11]:


#Check for null values
aparts_df.describe().T


# In[12]:


sns.heatmap(aparts_df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[13]:


aparts_df.dropna(subset=['sq_mtrs','Bedrooms'],inplace=True)
aparts_df.head()


# In[14]:


#check for houses that have missing bathrooms as null values
aparts_df[aparts_df['Bathrooms'].isnull()]


# In[15]:


#we see the bathroom column has a good correlation with the price column
#We shall group the houses by the number of bedrooms and fill the missing values
#with the mean of the bathrooms in each group, rounded to whole numbers
aparts_df.groupby('Bedrooms')['Bathrooms'].transform(lambda x: x.fillna(round(x.mean())))


# In[16]:


aparts_df['Bathrooms'] = aparts_df.groupby('Bedrooms')['Bathrooms'].transform(lambda x: x.fillna(round(x.mean())))


# In[17]:


#check for missing values
aparts_df.isnull().sum()


# In[18]:


#Notice that we won't need some of the columns to train our model; those are Agency, Neighborhood, and links.
#You also notice that sq_mtrs column is not correlated with any of the columns in our data.
#Remove non-required columns
del aparts_df["Agency"]
del aparts_df["link"] 


# # Visualising the data
# 
# 
# 

# In[19]:


#Visualize the data
#Lets first check the distribution of the price column
sns.displot(aparts_df['Price'])


# In[20]:


#we can see that the price column is right skewed
#lets check the houses that are outliers (200000 and above))
aparts_df[aparts_df['Price']>=200000]


# In[21]:


#Notice that the houses that are outliers are somehow in the same area
#sns scatterplot
sns.scatterplot(x='sq_mtrs',y='Price',data=aparts_df)


# In[22]:


#scatterplot for price  and bedrooms colored by bathrooms
sns.scatterplot(x='Bedrooms',y='Price',data=aparts_df,hue='Bathrooms')


# In[23]:


#we notice that the houses  are expensive when they have more bedrooms and bathrooms
#Lets check the scatterplot of the sq_mtrs column with price
sns.scatterplot(x='sq_mtrs',y='Price',data=aparts_df)


# In[24]:


#check for houses with less than 100 sq_mtrs
aparts_df[aparts_df['sq_mtrs']<100]


# In[25]:


#check for houses with less than 100 sq_mtrs
aparts_df[aparts_df['sq_mtrs']==0]


# In[26]:


#we notice houses with 0 sq_mtrs, we shall drop them
aparts_df.drop(aparts_df[aparts_df['sq_mtrs']==0].index,inplace=True)


# ## Conclusion
# The houses with more bedrooms and bathrooms are more expensive
# 

# In[28]:


aparts_df.info()


# In[29]:


#Lets check the scatterplot of the sq_mtrs column with price
sns.scatterplot(x='sq_mtrs',y='Price',data=aparts_df)


# In[30]:


#wee see most are less than 5000 sq_mtrs
#box plot for sq_mtrs
sns.boxplot(x='sq_mtrs',data=aparts_df)


# In[31]:


#we notice that there are outliers, we shall remove them
aparts_df.drop(aparts_df[aparts_df['sq_mtrs']>30000].index,inplace=True)


# In[32]:


#generate a pairplot on price, bedrooms, bathrooms and sq_mtrs
sns.pairplot(aparts_df[['Price','Bedrooms','Bathrooms','sq_mtrs']])


# In[33]:


#The houses with more bedrooms and bathrooms are more expensive
#The houses with more sq_mtrs are more expensive


# In[34]:


#Extract the town from neighborhood column
#for further analysis
aparts_df['Town'] = aparts_df['Neighborhood'].str.split(',').str[-1]


# In[35]:


#check for the towns
aparts_df['Town'].nunique()


# In[36]:


#grouby the towns and get the mean price,plot it
aparts_df.groupby('Town')['Price'].mean().sort_values(ascending=False).plot(kind='bar')


# In[37]:


#we notice that the houses in Westlands are the most expensive
#lets also group the town and count the number of houses,\\
aparts_df.groupby('Town')['Price'].count().sort_values(ascending=False).plot(kind='bar')


# In[38]:


#most houses are in Dagoretti North and Westlands 
#while the least are in Kangundo

#plot a scatterplot of the price and sq_mtrs colored by the town
plt.figure(figsize=(10,6))

sns.scatterplot(x='sq_mtrs',y='Price',data=aparts_df,hue='Town')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# In[39]:


#Summary
plt.figure(figsize=(15,12),dpi=180)
#plot without the legend
sns.scatterplot(x='sq_mtrs',y='Price',data=aparts_df,hue='Town',legend=False)


# In[40]:


#summary
#The houses in Westlands are the most expensive
#The houses in Dagoretti North and Westlands are the most numerous

#Lets check the houses with the most bedrooms
aparts_df[aparts_df['Bedrooms']==aparts_df['Bedrooms'].max()]


# # Set features and labels

# In[41]:


X = aparts_df[["Bedrooms", "Bathrooms","sq_mtrs","Town"]]
y = aparts_df[["Price"]]


# # Define the model and train it

# In[44]:


# Extract the numerical features and categorical feature
X_num = aparts_df[["Bedrooms", "Bathrooms", "sq_mtrs"]]
X_cat = aparts_df[["Town"]]


# In[46]:


# Create an instance of the OneHotEncoder class and fit it to the categorical feature
ohe = OneHotEncoder()
ohe.fit(X_cat)


# In[47]:


# Transform the categorical feature using the fitted OneHotEncoder
X_cat_encoded = ohe.transform(X_cat).toarray()


# In[48]:


# Combine the numerical and encoded categorical features
X = np.concatenate((X_num, X_cat_encoded), axis=1)
y = aparts_df[["Price"]]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

# Train the regression model using the training data
clf = DecisionTreeRegressor()
clf.fit(X_train, y_train)


# # Prediction and accuracy

# In[49]:


#Predictions using the testing set 
y_pred = clf.predict(X_test)

# #Example of few predictions
# print("Making predictions for the following 5 houses:")
# print(X.head())
# print("The predictions are")
# five_pred= clf.predict(X_test)[:5]

# five_pred


# In[50]:


print(X_test.shape)
print(y_test.shape)
print(y_pred.shape)


# In[51]:


# print("Making predictions for the following 5 houses:")
# print(X.head())
# print("The predictions are")
# str(y_test[:5])


# In[52]:


# #Checking the accuracy of the model using MSE,MAE and R-squared error

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print('Mean squared error: ', mean_squared_error(y_test, y_pred))
print("Root Mean Squared error", np.sqrt(mean_squared_error(y_test, y_pred)))
print('Mean absolute error: ', mean_absolute_error(y_test, y_pred))
print('R-squared score: ', r2_score(y_test, y_pred))


# In[53]:


# trying new model
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Always scale the input. The most convenient way is to use a pipeline.
clf = make_pipeline(StandardScaler(),
                     SGDRegressor(max_iter=1000, tol=1e-3, loss="squared_error"))


# In[54]:


#Checking the accuracy of a model
clf.fit(X_train, y_train)
forestPred = clf.predict(X_test)
forestScores = clf.score(X_test, y_test)
forestScores


# In[55]:


mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
rmse


# In[56]:


#using a scatter plot to visualize how well the model is perfoming
plt.scatter(y_test, y_pred)
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.title('Actual vs Predicted values')
plt.show()


# In[57]:


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


joblib.dump(clf, 'house_price_prediction.joblib')

