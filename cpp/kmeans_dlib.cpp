// kmeans_dlib.cpp
#include <dlib/clustering.h>
#include <dlib/matrix.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <cstdlib> // For std::rand()

using namespace dlib;
using namespace std;

// Define sample_type
typedef matrix<double, 0, 1> sample_type;

int main() {
    // Parameters
    const int samples = 10000;
    const int features = 10;
    const int clusters = 5;

    // Containers for data
    std::vector<sample_type> data;
    data.reserve(samples);

    // Initialize random number generator
    std::srand(42);

    // Generate synthetic data
    for(int i = 0; i < samples; ++i) {
        sample_type samp;
        samp.set_size(features); // Ensure the sample has correct size
        for(int j = 0; j < features; ++j) {
            samp(j) = std::rand() / double(RAND_MAX); // Qualified rand()
        }
        data.push_back(samp);
    }

    // Define the k-means trainer
    typedef radial_basis_kernel<sample_type> kernel_type;
    kcentroid<kernel_type> kc(kernel_type(0.1), 0.01, 8);
    kkmeans<kernel_type> test(kc);
    test.set_number_of_centers(clusters);

    // Initialize initial centers (e.g., first 'clusters' samples)
    std::vector<sample_type> initial_centers;
    initial_centers.reserve(clusters);
    for(int i = 0; i < clusters; ++i) {
        initial_centers.push_back(data[i]);
    }

    // Measure training time
    auto start = chrono::high_resolution_clock::now();
    test.train(data, initial_centers, 10); // data, initial_centers, number_of_repeats
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> training_duration = end - start;

    // Measure inference time
    start = chrono::high_resolution_clock::now();
    std::vector<unsigned long> assignments;
    assignments.reserve(data.size());
    for(const auto& samp : data) {
        assignments.push_back(test(samp));
    }
    end = chrono::high_resolution_clock::now();
    chrono::duration<double> inference_duration = end - start;

    // Output the results
    cout << "C++ (Dlib) - K-Means Clustering:\n";
    cout << "Training Time: " << training_duration.count() << " seconds\n";
    cout << "Inference Time: " << inference_duration.count() << " seconds\n\n";

    return 0;
}
