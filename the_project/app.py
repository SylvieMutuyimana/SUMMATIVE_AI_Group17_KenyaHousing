from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder

# Load the dataset
data = pd.read_csv('rent_apts.csv')
data['Town'] = data['Neighborhood'].str.split(',').str[-1]

# Load the trained model
clf = joblib.load('house_price_prediction.joblib')

# Create an instance of the OneHotEncoder class and fit it to the Town column of the dataset
ohe = OneHotEncoder(handle_unknown='ignore')
ohe.fit(data[['Town']])

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    # If the form has been submitted, make a prediction and show the results
    if request.method == 'POST':
        # Get the input values from the form
        bedrooms = request.form.get('bedrooms', type=int)
        bathrooms = request.form.get('bathrooms', type=int)
        sq_mtrs = request.form.get('sq_mtrs', type=int)
        town = request.form.get('town')

        # One-hot encode the town value
        X_cat = pd.DataFrame({'Town': [town]})
        X_cat_encoded = pd.DataFrame(ohe.transform(X_cat).toarray(), columns=ohe.get_feature_names_out(['Town']))

        # Combine the numerical and encoded categorical features
        X_num = pd.DataFrame({'Bedrooms': [bedrooms], 'Bathrooms': [bathrooms], 'sq_mtrs': [sq_mtrs]})
        X = pd.concat([X_num, X_cat_encoded], axis=1)

        # Make the prediction
        price_pred = round(clf.predict(X)[0], 2)

        # Render the results table
        return render_template('index.html', towns=sorted(data['Town'].unique()), 
                               bedrooms=bedrooms, bathrooms=bathrooms, sq_mtrs=sq_mtrs,
                               town=town, price_pred=price_pred)

    # If the form has not been submitted yet, show the form without any results
    return render_template('index.html', towns=sorted(data['Town'].unique()))

if __name__ == '__main__':
    app.run(debug=True)
