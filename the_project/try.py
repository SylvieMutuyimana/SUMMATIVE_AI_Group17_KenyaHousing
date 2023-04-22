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

aparts_df = pd.read_csv('rent_apts.csv')
aparts_df.head()

#Shape of the dataset, rows and columns respectively.
aparts_df.shape

aparts_df['Price'] = aparts_df['Price'].str.replace('KSh','',regex=True).str.replace(',','').astype(float)

#box plot for sq_mtrs
sns.boxplot(x='sq_mtrs',data=aparts_df)

#we notice that there are outliers, we shall remove them
aparts_df.drop(aparts_df[aparts_df['sq_mtrs']>30000].index,inplace=True)

#generate a pairplot on price, bedrooms, bathrooms and sq_mtrs
sns.pairplot(aparts_df[['Price','Bedrooms','Bathrooms','sq_mtrs']])

#Extract the town from neighborhood column
aparts_df['Town'] = aparts_df['Neighborhood'].str.split(',').str[-1]

#check for the towns
aparts_df['Town'].nunique()

#grouby the towns and get the mean price,plot it
aparts_df.groupby('Town')['Price'].mean().sort_values(ascending=False).plot(kind='bar')
#we notice that the houses in Westlands are the most expensive
aparts_df.groupby('Town')['Price'].count().sort_values(ascending=False).plot(kind='bar')

#plot a scatterplot of the price and sq_mtrs colored by the town
plt.figure(figsize=(10,6))

sns.scatterplot(x='sq_mtrs',y='Price',data=aparts_df,hue='Town')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.figure(figsize=(15,12),dpi=180)
#plot without the legend
sns.scatterplot(x='sq_mtrs',y='Price',data=aparts_df,hue='Town',legend=False)

#Lets check the houses with the most bedrooms
aparts_df[aparts_df['Bedrooms']==aparts_df['Bedrooms'].max()]

X = aparts_df[["Bedrooms", "Bathrooms","sq_mtrs","Town"]]
y = aparts_df[["Price"]]


# Extract the numerical features and categorical feature
X_num = aparts_df[["Bedrooms", "Bathrooms", "sq_mtrs"]]
X_cat = aparts_df[["Town"]]

# Create an instance of the OneHotEncoder class and fit it to the categorical feature
ohe = OneHotEncoder()
ohe.fit(X_cat)

# Transform the categorical feature using the fitted OneHotEncoder
X_cat_encoded = ohe.transform(X_cat).toarray()

# Combine the numerical and encoded categorical features
X = np.concatenate((X_num, X_cat_encoded), axis=1)
y = aparts_df[["Price"]]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

# Train the regression model using the training data
clf = DecisionTreeRegressor()
clf.fit(X_train, y_train)

#Predictions using the testing set 
y_pred = clf.predict(X_test)

print(X_test.shape)
print(y_test.shape)
print(y_pred.shape)

# #Checking the accuracy of the model using MSE,MAE and R-squared error

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print('Mean squared error: ', mean_squared_error(y_test, y_pred))
print("Root Mean Squared error", np.sqrt(mean_squared_error(y_test, y_pred)))
print('Mean absolute error: ', mean_absolute_error(y_test, y_pred))
print('R-squared score: ', r2_score(y_test, y_pred))

# trying new model
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Always scale the input. The most convenient way is to use a pipeline.
clf = make_pipeline(StandardScaler(),
                     SGDRegressor(max_iter=1000, tol=1e-3, loss="squared_error"))

#Checking the accuracy of a model
clf.fit(X_train, y_train)
forestPred = clf.predict(X_test)
forestScores = clf.score(X_test, y_test)
forestScores

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
rmse

plt.scatter(y_test, y_pred)
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.title('Actual vs Predicted values')
plt.show()

joblib.dump(clf, 'house_price_prediction.joblib')