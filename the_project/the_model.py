#!/usr/bin/env python
# coding: utf-8

# Importing required libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder
import joblib


# Loading the datasets

# In[2]:


df_aparts = pd.read_csv('apartments.csv')
df_aparts.columns


# In[3]:


df_rentapts = pd.read_csv('rent_apts.csv')
df_rentapts.columns


# In[4]:


df_aparts.head(5)


# In[5]:


df_rentapts.head(5)


# Exploring the dataset

# In[6]:


#Shape of the datasets, rows and columns respectively.
df_aparts.shape


# In[7]:


df_rentapts.shape


# In[8]:


df_aparts.info()


# In[9]:


df_rentapts.info()


# In[10]:


df_aparts.columns
df_aparts.isnull().sum()


# In[11]:


df_rentapts.columns
df_rentapts.isnull().sum()


# In[12]:


#describe the data
df_rentapts.describe().T


# In[13]:


#describe the data
df_aparts.describe().T


# # Data cleaning and wrangling

# df_aparts

# In[14]:


#drop the first column (Unnamed: 0) 
df_aparts.drop('Unnamed: 0', axis=1, inplace=True)
df_aparts.columns


# In[15]:


print(df_aparts['title'].head(10))


# In[16]:


import re
# Cleaning the location
df_aparts['title'] = df_aparts['title'].apply(lambda x: re.sub(r'^\d+ Bedroom Apartment / Flat to rent in ', '', x))
print(df_aparts['title'])
print('Number of unique values:', df_aparts['title'].nunique())


# In[17]:


df_aparts = df_aparts.drop('location', axis=1)
df_aparts.isnull().sum()
num_none_values = df_aparts.isnull().sum().sum()
print("Number of None values: ", num_none_values)


# In[18]:


#Renaming the columns
df_aparts.rename(columns={'title': 'Town'}, inplace=True)
df_aparts.rename(columns={'bedrooms': 'Bedrooms'}, inplace=True)
df_aparts.rename(columns={'bathrooms': 'Bathrooms'}, inplace=True)
df_aparts.rename(columns={'price': 'Price'}, inplace=True)
df_aparts.columns


# In[19]:


print(df_aparts[['rate', 'Price']])


# In[20]:


# Remove non-numeric characters and whitespace from the "Price" column
df_aparts['Price'] = df_aparts['Price'].str.replace('[^\d]+', '', regex=True)
# Convert the column to integers
df_aparts['Price'] = pd.to_numeric(df_aparts['Price'], errors='coerce').astype('Int64')
# Print the rate and Price columns to check the output
print(df_aparts[['rate', 'Price']])


# In[21]:


# Drop rows with NaN values in the "Price" column
df_aparts.dropna(subset=['Price'], inplace=True)
# Print the rate and Price columns to check the output
print(df_aparts[['rate', 'Price']])


# In[22]:


# Print the unique values for is called "Rate"
print(df_aparts['rate'].unique())


# In[23]:


np_days = len(df_aparts[df_aparts['rate'] == 'Per Month'])
print("For Months: ", np_days)
np_months = len(df_aparts[df_aparts['rate'] == 'Per Day'])
print("For Days: ", np_months)


# In[24]:


df_aparts.loc[df_aparts['rate'] == 'Per Day', 'Price'] *= 30
print(df_aparts['Price'].head(10))
df_aparts = df_aparts.drop('rate', axis=1)


# In[25]:


df_aparts['Bedrooms'] = df_aparts['Bedrooms'].astype(int)
df_aparts['Bathrooms'] = df_aparts['Bathrooms'].astype(int)
df_aparts['Price'] = df_aparts['Price'].astype(int)
print(df_aparts.dtypes)


# In[26]:


#check for missing values
df_aparts.isnull().sum()


# df_rentapts 

# In[27]:


#the price column is an object type, we shall convert it to float
#first we shall remove the KSh sign and the comma (KSh 50,000)
df_rentapts['Price'].str.replace('KSh','',regex=True).str.replace(',','')


# In[28]:


df_rentapts['Price'] = df_rentapts['Price'].str.replace('KSh','',regex=True).str.replace(',','').astype(float)


# In[29]:


df_rentapts.info()


# In[30]:


#Check for null values
df_rentapts.describe().T


# In[31]:


sns.heatmap(df_rentapts.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[32]:


df_rentapts.dropna(subset=['sq_mtrs','Bedrooms'],inplace=True)
df_rentapts.head()


# In[33]:


#check for houses that have missing bathrooms as null values
df_rentapts[df_rentapts['Bathrooms'].isnull()]


# In[34]:


#we see the bathroom column has a good correlation with the price column
#We shall group the houses by the number of bedrooms and fill the missing values
#with the mean of the bathrooms in each group, rounded to whole numbers
df_rentapts.groupby('Bedrooms')['Bathrooms'].transform(lambda x: x.fillna(round(x.mean())))


# In[35]:


df_rentapts['Bathrooms'] = df_rentapts.groupby('Bedrooms')['Bathrooms'].transform(lambda x: x.fillna(round(x.mean())))


# In[36]:


#check for missing values
df_rentapts.isnull().sum()


# In[37]:


#Extract the town from neighborhood column
#for further analysis
df_rentapts['Town'] = df_rentapts['Neighborhood'].str.split(',').str[-1]


# In[38]:


#check for the towns
df_rentapts['Town'].nunique()


# In[39]:


#Notice that we won't need some of the columns to train our model; those are Agency, Neighborhood, and links.
#You also notice that sq_mtrs column is not correlated with any of the columns in our data.
#Remove non-required columns
del df_rentapts["Agency"]
del df_rentapts["link"] 
del df_rentapts["Neighborhood"] 

df_rentapts.columns


# In[40]:


df_aparts['sq_mtrs'] = pd.Series(dtype='int64')
df_aparts.columns


# In[41]:


df_rentapts.columns


# In[42]:


# Define the new order of columns to look like the other dataset
new_order = ['Town','sq_mtrs', 'Bedrooms', 'Bathrooms', 'Price']
# Reorder the columns of the DataFrame
df_aparts = df_rentapts.reindex(columns=new_order)
df_rentapts = df_rentapts.reindex(columns=new_order)


# In[43]:


df_rentapts.columns
df_rentapts.head(5)


# Merged Dataset

# In[44]:


#Merge the dataset
# Concatenate the datasets vertically
merged_df = pd.concat([df_aparts, df_rentapts], axis=0, ignore_index=True)
merged_df.columns


# In[45]:


#Explore the merged dataset
merged_df.shape


# In[46]:


merged_df.info()


# In[47]:


merged_df.head(5)


# In[48]:


#check for the towns
merged_df['Town'].nunique()


# In[49]:


# Sort the DataFrame by the 'town' column in alphabetical order
merged_df = merged_df.sort_values('Town')
merged_df.head(5)


# In[50]:


unique_towns = merged_df['Town'].unique()
print(unique_towns)


# In[51]:


# Replace 'North', and 'South' from all the values in the 'town' column
merged_df['Town'] = merged_df['Town'].str.replace(' North', '').str.replace(' South', '')
# Replace 'East' and 'West' from all the values in the 'town' column except Westlands
merged_df['Town'] = merged_df['Town'].apply(lambda x: x if x == ' Westlands' else x.replace(' East', '').replace(' West', ''))
# Replace 'Road' and 'Town' from all the values in the 'town' column
merged_df['Town'] = merged_df['Town'].str.replace(' Road', '').str.replace(' Town', '')
unique_towns = merged_df['Town'].unique()
# Replace 'Constituency' and 'Central' from all the values in the 'town' column
merged_df['Town'] = merged_df['Town'].str.replace(' Constituency', '').str.replace(' Central', '')
unique_towns = merged_df['Town'].unique()
# Replace 'CBD' and 'Central' from all the values in the 'town' column
merged_df['Town'] = merged_df['Town'].str.replace(' CBD', '').str.replace(' Central', '')
# Replace 'space' at the beginning from all the values in the 'town' column
merged_df['Town'] = merged_df['Town'].str.lstrip()
unique_towns = merged_df['Town'].unique()
merged_df = merged_df.sort_values('Town')
print(unique_towns)


# In[52]:


missing_mask = merged_df['sq_mtrs'].isna()

for index, row in merged_df[missing_mask].iterrows():
    town = row['Town']
    bedrooms = row['Bedrooms']
    bathrooms = row['Bathrooms']
    similar_rows = merged_df[(merged_df['Town'] == town) & (merged_df['Bedrooms'] == bedrooms) & (merged_df['Bathrooms'] == bathrooms)]
    
    if len(similar_rows) == 0:
        similar_rows = merged_df[(merged_df['Bedrooms'] == bedrooms) & (merged_df['Bathrooms'] == bathrooms)]
    
    if len(similar_rows) > 0:
        merged_df.at[index, 'sq_mtrs'] = similar_rows['sq_mtrs'].mean()

# Check for missing values
missing_mask = merged_df['sq_mtrs'].isna()
if missing_mask.any():
    print("Rows with missing values in 'sq_mtrs' column:")
    print(merged_df[missing_mask])
else:
    print("All missing values in 'sq_mtrs' column have been filled")


# In[53]:


# Save the merged dataset
merged_df.to_csv('merged_dataset.csv', index=False)


# # Visualising the data
# 
# 
# 

# df_aparts

# In[54]:


#Lets first check the distribution of the Price column
sns.displot(merged_df['Price'])


# In[55]:


#scatterplot for price  and Bedrooms colored by Bathrooms
sns.scatterplot(x='Bedrooms',y='Price',data=df_aparts,hue='Bathrooms')


# df_rentapts

# In[56]:


#we can see that the price column is right skewed
#lets check the houses that are outliers (200000 and above))
df_rentapts[merged_df['Price']>=200000]


# ## Conclusion
# The houses with more Bedrooms and Bathrooms are more expensive
# 

# In[57]:


merged_df.info()


# In[58]:


#generate a pairplot on price, Bedrooms, Bathrooms and sq_mtrs
sns.pairplot(merged_df[['Price','sq_mtrs','Bedrooms','Bathrooms','Town']])


# In[60]:


#check for the towns
merged_df['Town'].nunique()


# In[59]:


#The houses with more Bedrooms and bathrooms are more expensive
#Lets check the scatterplot of the sq_mtrs column with price
sns.scatterplot(x='sq_mtrs',y='Price',data=df_rentapts)


# In[61]:


#grouby the towns and get the mean price,plot it
merged_df.groupby('Town')['Price'].mean().sort_values(ascending=False).plot(kind='bar')


# In[62]:


plt.figure(figsize=(10,6))
sns.scatterplot(x='Bedrooms',y='Price',data=merged_df,hue='Town')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# In[63]:


plt.figure(figsize=(10,6))
sns.scatterplot(x='Bathrooms',y='Price',data=merged_df,hue='Town')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# In[64]:


#Lets check the houses with the most bedrooms
merged_df[merged_df['Bedrooms']==merged_df['Bedrooms'].max()]


# # Set features and labels

# In[65]:


X = merged_df[["Bedrooms","sq_mtrs", "Bathrooms","Town"]]
y = merged_df[["Price"]]


# # Define the model and train it

# In[66]:


# Extract the numerical features and categorical feature
X_num = merged_df[["sq_mtrs","Bedrooms", "Bathrooms"]]
X_cat = merged_df[["Town"]]


# In[67]:


# Create an instance of the OneHotEncoder class and fit it to the categorical feature
ohe = OneHotEncoder()
ohe.fit(X_cat)


# In[68]:


# Transform the categorical feature using the fitted OneHotEncoder
X_cat_encoded = ohe.transform(X_cat).toarray()


# In[69]:


# Combine the numerical and encoded categorical features
X = np.concatenate((X_num, X_cat_encoded), axis=1)
y = merged_df[["Price"]]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

# Train the regression model using the training data
clf = DecisionTreeRegressor()
clf.fit(X_train, y_train)


# # Prediction and accuracy

# In[70]:


#Predictions using the testing set 
y_pred = clf.predict(X_test)

# #Example of few predictions
# print("Making predictions for the following 5 houses:")
# print(X.head())
# print("The predictions are")
# five_pred= clf.predict(X_test)[:5]

# five_pred


# In[71]:


print(X_test.shape)
print(y_test.shape)
print(y_pred.shape)


# In[72]:


# print("Making predictions for the following 5 houses:")
# print(X.head())
# print("The predictions are")
# str(y_test[:5])


# In[73]:


# #Checking the accuracy of the model using MSE,MAE and R-squared error

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print('Mean squared error: ', mean_squared_error(y_test, y_pred))
print("Root Mean Squared error: ", np.sqrt(mean_squared_error(y_test, y_pred)))
print('Mean absolute error: ', mean_absolute_error(y_test, y_pred))
print('R-squared score: ', r2_score(y_test, y_pred))


# In[74]:


# trying new model
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Always scale the input. The most convenient way is to use a pipeline.
clf = make_pipeline(StandardScaler(),
                     SGDRegressor(max_iter=1000, tol=1e-3, loss="squared_error"))


# In[75]:


#Checking the accuracy of a model
clf.fit(X_train, y_train)
forestPred = clf.predict(X_test)
forestScores = clf.score(X_test, y_test)
forestScores


# In[76]:


mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
rmse


# In[77]:


#using a scatter plot to visualize how well the model is perfoming
plt.scatter(y_test, y_pred)
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.title('Actual vs Predicted values')
plt.show()


# In[78]:


# from sklearn.model_selection import cross_val_score
#  #Cross-validate the model
#  #Perform cross-validation on the model
# scores = cross_val_score(clf, X, y, cv=5, scoring='neg_mean_squared_error')
# rmse_scores = np.sqrt(-scores)

#  # Display the cross-validation scores
# print('Cross-Validation Scores:', rmse_scores)
# print('Mean:', rmse_scores.mean())
# print('Standard deviation:', rmse_scores.std())


# In[79]:


joblib.dump(clf, 'the_model.joblib')

