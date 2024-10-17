// mlp_dlib.cpp
#include <dlib/dnn.h>
#include <dlib/data_io.h>
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
    const int samples = 150;
    const int features = 4;
    const int classes = 3;

    // Containers for data
    std::vector<sample_type> X;
    std::vector<unsigned long> y;

    // Initialize random number generator
    std::srand(42);

    // Generate synthetic data
    for(int i = 0; i < samples; ++i) {
        sample_type samp;
        samp.set_size(features); // Ensure the sample has correct size
        for(int j = 0; j < features; ++j) {
            samp(j) = std::rand() / double(RAND_MAX); // Qualified rand()
        }
        X.push_back(samp);
        y.push_back(std::rand() % classes);
    }

    // Split into training and testing (80-20 split)
    int train_size = static_cast<int>(0.8 * samples);
    std::vector<sample_type> X_train(X.begin(), X.begin() + train_size);
    std::vector<unsigned long> y_train(y.begin(), y.begin() + train_size);
    std::vector<sample_type> X_test(X.begin() + train_size, X.end());
    std::vector<unsigned long> y_test(y.begin() + train_size, y.end());

    // Define network
    typedef loss_multiclass_log<fc<classes, relu<fc<100, relu<fc<100, input<matrix<double, 0, 1>>>>>>>> net_type;
    net_type net;

    // Configure network
    dnn_trainer<net_type> trainer(net, sgd());
    trainer.set_learning_rate(0.01);
    trainer.set_min_learning_rate(0.0001);
    trainer.set_mini_batch_size(10);
    trainer.set_max_num_epochs(50);

    // Measure training time
    auto start = chrono::high_resolution_clock::now();
    trainer.train(X_train, y_train);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> training_duration = end - start;

    // Measure inference time
    start = chrono::high_resolution_clock::now();
    std::vector<unsigned long> predictions;
    predictions.reserve(X_test.size());
    for(const auto& samp : X_test) {
        predictions.push_back(net(samp));
    }
    end = chrono::high_resolution_clock::now();
    chrono::duration<double> inference_duration = end - start;

    // Output the results
    cout << "C++ (Dlib) - Multilayer Perceptron:\n";
    cout << "Training Time: " << training_duration.count() << " seconds\n";
    cout << "Inference Time: " << inference_duration.count() << " seconds\n\n";

    return 0;
}
