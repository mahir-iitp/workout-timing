# Fitness Workout Duration Predictor

A machine learning-powered web application that predicts optimal workout duration based on personal characteristics and workout type.

## Features
- Predicts workout duration based on age, gender, height, weight, and workout type
- RESTful API endpoints for predictions
- Web interface for easy interaction
- Trained machine learning model using scikit-learn

## Deployment on Render

1. Create a new account on [Render](https://render.com) if you haven't already
2. Click on "New +" and select "Web Service"
3. Connect your GitHub repository
4. Fill in the following details:
   - Name: `fitness-workout-predictor` (or your preferred name)
   - Environment: `Python 3`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app`
5. Click "Create Web Service"

## Environment Variables
No additional environment variables are required for basic deployment.

## API Endpoints

- `GET /`: Web interface for the workout duration predictor
- `POST /predict`: Prediction endpoint
  - Input (JSON):
    ```json
    {
        "age": "25",
        "gender": "Male",
        "height": "175",
        "weight": "70",
        "workout_type": "Cardio"
    }
    ```
- `GET /health`: Health check endpoint

## Local Development
1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Unix/MacOS: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Run the application: `python app.py`

## Dataset
The application uses the `workout_fitness_tracker_data.csv` file for training the model.