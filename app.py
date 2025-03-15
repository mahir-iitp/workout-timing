from flask import Flask, render_template, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import os

app = Flask(__name__, static_url_path='')

# Load and train the model directly
def load_and_train_model():
    try:
        # Load the data
        df = pd.read_csv("workout_fitness_tracker_data.csv")
        
        # Process the data
        df = df.iloc[:, 1:7]
        
        # Split features and target
        X = df.iloc[:, 0:5]
        y = df.iloc[:, -1]
        
        # Define preprocessing steps
        categorical_features = ['Gender', 'Workout Type']
        numerical_features = ['Age', 'Height (cm)', 'Weight (kg)']
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ]
        )
        
        # Create the pipeline
        model = Pipeline([
            ('preprocessing', preprocessor),
            ('classifier', DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42))
        ])
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train the model
        model.fit(X_train, y_train)
        
        accuracy = model.score(X_test, y_test)
        print(f"Model trained successfully! Accuracy on test set: {accuracy:.2f}")
        
        return model
    
    except Exception as e:
        print(f"Error training model: {e}")
        return None

# Train the model when the app starts
model = load_and_train_model()

@app.route('/')
def home():
    return send_from_directory('static', 'fitness_calculator.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = request.get_json() if request.is_json else request.form
        
        age = data.get('age')
        gender = data.get('gender')
        height = data.get('height')
        weight = data.get('weight')
        workout_type = data.get('workout_type')
        
        # Validate input data
        if not all([age, gender, height, weight, workout_type]):
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Create DataFrame with the input data
        column_names = ['Age', 'Gender', 'Height (cm)', 'Weight (kg)', 'Workout Type']
        input_data = pd.DataFrame([[age, gender, height, weight, workout_type]], columns=column_names)
        
        # Make prediction
        prediction = model.predict(input_data)
        
        return jsonify({
            'prediction': int(prediction[0]),
            'message': f'Recommended workout duration: {int(prediction[0])} minutes'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health')
def health_check():
    """Health check endpoint for Render."""
    return jsonify({'status': 'healthy', 'message': 'Service is running'}), 200

if __name__ == '__main__':
    # Get port from environment variable or use 5000 as default
    port = int(os.environ.get('PORT', 5000))
    
    # In production, debug should be False
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    app.run(host='0.0.0.0', port=port, debug=debug)
