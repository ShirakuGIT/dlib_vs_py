# linear_regression_scikit.py
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import time
from sklearn.datasets import fetch_california_housing 

def main():
    # Load dataset
    data = fetch_california_housing()
    X, y = data.data, data.target

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize model
    model = LinearRegression()

    # Measure training time
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    # Measure inference time
    start_time = time.time()
    predictions = model.predict(X_test)
    inference_time = time.time() - start_time

    print(f"Python (scikit-learn) - Linear Regression:")
    print(f"Training Time: {training_time:.6f} seconds")
    print(f"Inference Time: {inference_time:.6f} seconds\n")

if __name__ == "__main__":
    main()
