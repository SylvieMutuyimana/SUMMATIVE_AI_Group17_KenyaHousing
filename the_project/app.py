from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

# Load the saved model
clf = joblib.load('house_price_prediction.joblib')

# Initialize the Flask application
app = Flask(__name__)

# Define the endpoint for the root URL
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    input_data = None
    if request.method == 'POST':
        # Extract the form data from the request
        bedrooms = request.form['bedrooms']
        bathrooms = request.form['bathrooms']

        # Validate the form data
        try:
            n_bed = int(bedrooms)
            n_bath = float(bathrooms)
            # Convert the form data to a JSON object
            input_data = {
                'Bedrooms': bedrooms,
                'Bathrooms': bathrooms,
            }
            # Preprocess the input data
            input_df = pd.DataFrame.from_dict(input_data, orient='index').T
            input_df['Bathrooms'] = input_df.groupby('Bedrooms')['Bathrooms'].transform(
                lambda x: x.fillna(round(x.mean())))
            # Make predictions using the loaded model
            predictions = clf.predict(input_df)
            # Set the prediction value
            prediction = predictions[0]
        except ValueError:
            return render_template('index.html', error_message='Invalid input. Please enter numbers only.')

    # Render the index page with the prediction result or error message
    return render_template('index.html', prediction=prediction, input_data=input_data)

if __name__ == '__main__':
    app.run(debug=True)
