// svm_dlib.cpp
#include <dlib/svm.h>
#include <dlib/data_io.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <cstdlib> // For std::rand()

using namespace dlib;
using namespace std;

// Define the sample type as a column vector with dynamic size
typedef matrix<double, 0, 1> sample_type;

int main() {
    // Parameters
    const int samples = 150;
    const int features = 4;
    const int classes = 3;

    // Containers for data
    std::vector<sample_type> X;
    std::vector<double> y;

    // Seed the random number generator for reproducibility
    std::srand(42);

    // Generate synthetic data
    for(int i = 0; i < samples; ++i) {
        sample_type samp;
        samp.set_size(features); // Ensure the sample has the correct size
        for(int j = 0; j < features; ++j) {
            samp(j) = std::rand() / double(RAND_MAX);
        }
        X.push_back(samp);
        y.push_back(std::rand() % classes);
    }

    // Split into training and testing (80-20 split)
    int train_size = static_cast<int>(0.8 * samples);
    std::vector<sample_type> X_train(X.begin(), X.begin() + train_size);
    std::vector<double> y_train(y.begin(), y.begin() + train_size);
    std::vector<sample_type> X_test(X.begin() + train_size, X.end());
    std::vector<double> y_test(y.begin() + train_size, y.end());

    // Define SVM with Radial Basis Function (RBF) kernel
    typedef radial_basis_kernel<sample_type> kernel_type;
    svm_c_trainer<kernel_type> trainer;
    trainer.set_kernel(kernel_type(0.1)); // Set gamma parameter for RBF

    // Measure training time
    auto start = chrono::high_resolution_clock::now();
    decision_function<kernel_type> df = trainer.train(X_train, y_train);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> training_duration = end - start;

    // Measure inference time
    start = chrono::high_resolution_clock::now();
    std::vector<double> predictions;
    predictions.reserve(X_test.size());
    for(const auto& samp : X_test) {
        predictions.push_back(df(samp));
    }
    end = chrono::high_resolution_clock::now();
    chrono::duration<double> inference_duration = end - start;

    // Output the results
    cout << "C++ (Dlib) - SVM:\n";
    cout << "Training Time: " << training_duration.count() << " seconds\n";
    cout << "Inference Time: " << inference_duration.count() << " seconds\n\n";

    return 0;
}
