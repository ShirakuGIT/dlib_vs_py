# kmeans_scikit.py
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import time

def main():
    # Generate synthetic dataset
    X, _ = make_blobs(n_samples=10000, centers=5, n_features=10, random_state=42)

    # Initialize model
    model = KMeans(n_clusters=5, random_state=42)

    # Measure training time
    start_time = time.time()
    model.fit(X)
    training_time = time.time() - start_time

    # Measure inference time (predicting cluster for all samples)
    start_time = time.time()
    predictions = model.predict(X)
    inference_time = time.time() - start_time

    print(f"Python (scikit-learn) - K-Means Clustering:")
    print(f"Training Time: {training_time:.6f} seconds")
    print(f"Inference Time: {inference_time:.6f} seconds\n")

if __name__ == "__main__":
    main()
