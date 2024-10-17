#include <dlib/matrix.h>
#include <dlib/statistics.h>
#include <iostream>
#include <chrono>
#include <cstdlib>  // Added for std::rand()

using namespace dlib;
using namespace std;

int main() {
    // Generate synthetic data similar to Boston Housing
    const int samples = 506;
    const int features = 13;

    matrix<double> X(samples, features);
    matrix<double> y(samples, 1);

    // Seed the random number generator for reproducibility
    std::srand(42);

    for(int i = 0; i < samples; ++i) {
        for(int j = 0; j < features; ++j) {
            X(i,j) = std::rand() / double(RAND_MAX);  // Qualified rand()
        }
        y(i,0) = 3.0 * X(i,0) + 2.0 * X(i,1) + 1.0 * X(i,2) + 0.5; // Simple linear relation
    }

    // Split into training and testing (80-20 split)
    int train_size = static_cast<int>(0.8 * samples);
    matrix<double> X_train = subm(X, range(0, train_size-1), range(0, features-1));
    matrix<double> y_train = subm(y, range(0, train_size-1), range(0, 0));

    matrix<double> X_test = subm(X, range(train_size, samples-1), range(0, features-1));
    matrix<double> y_test = subm(y, range(train_size, samples-1), range(0, 0));

    // Measure training time
    auto start = chrono::high_resolution_clock::now();
    matrix<double> weights = inv(trans(X_train)*X_train) * trans(X_train) * y_train;
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> training_duration = end - start;

    // Measure inference time
    start = chrono::high_resolution_clock::now();
    matrix<double> predictions = X_test * weights;
    end = chrono::high_resolution_clock::now();
    chrono::duration<double> inference_duration = end - start;

    cout << "C++ (Dlib) - Linear Regression:\n";
    cout << "Training Time: " << training_duration.count() << " seconds\n";
    cout << "Inference Time: " << inference_duration.count() << " seconds\n\n";

    return 0;
}
