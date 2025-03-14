import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Load the data
# Note: You'll need to provide the actual CSV file
try:
    df = pd.read_csv("workout_fitness_tracker_data.csv")
    
    # Process the data as in the notebook
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
    
    # Save the model
    with open('fitness_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print("Model trained and saved successfully!")
    print(f"Accuracy on test set: {model.score(X_test, y_test):.2f}")
    
except Exception as e:
    print(f"Error training model: {e}")
    print("Note: You need to provide the workout_fitness_tracker_data.csv file") 