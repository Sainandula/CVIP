import joblib
from flask import Flask, render_template, request, jsonify

# Load the trained model
model = joblib.load('house_price_model1.pkl')  # Ensure this path is correct

# Initialize Flask app
app = Flask(__name__)

# Route for rendering the HTML form
@app.route('/')
def index():
    return render_template('index.html')  # Ensure the HTML form is ready

# Route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collecting input features from the form
        features = [
            float(request.form['MedInc']),
            float(request.form['HouseAge']),
            float(request.form['AveRooms']),
            float(request.form['AveBedrms']),
            float(request.form['Population']),
            float(request.form['AveOccup']),
            float(request.form['Latitude']),
            float(request.form['Longitude'])
        ]

        # Make a prediction with the trained model
        prediction = model.predict([features])

        # Convert prediction to float before sending as a response
        prediction_value = float(prediction[0])

        # Return prediction as a JSON response
        return jsonify({'prediction': prediction_value})

    except Exception as e:
        return jsonify({'error': str(e)})  # Handle errors if something goes wrong

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
