# dlib_vs_py
# Machine Learning Performance Comparison: Dlib (C++) vs. scikit-learn (Python)

![License](https://img.shields.io/badge/license-MIT-blue.svg)

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Algorithms Compared](#algorithms-compared)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [C++ (Dlib) Setup](#cpp-dlib-setup)
  - [Python (scikit-learn) Setup](#python-scikit-learn-setup)
- [Usage](#usage)
  - [Running C++ Implementations](#running-c-implementations)
  - [Running Python Implementations](#running-python-implementations)
  - [Generating Performance Plots](#generating-performance-plots)
- [Results](#results)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)
- [License](#license)
- [References](#references)

---

## Project Overview

This project conducts a comprehensive performance comparison between **Dlib** (a modern C++ toolkit) and **scikit-learn** (a widely-used Python library) across four fundamental machine learning algorithms:

1. **Linear Regression**
2. **Support Vector Machines (SVM)**
3. **K-Means Clustering**
4. **Multilayer Perceptron (MLP)**

The goal is to evaluate and contrast the training and inference times of these algorithms implemented in both libraries, providing insights into the efficiency and suitability of each for various applications.

---

## Features

- **Dual Implementations:** Provides both C++ (Dlib) and Python (scikit-learn) implementations of key machine learning algorithms.
- **Performance Benchmarking:** Measures and compares training and inference times across libraries.
- **Visualization:** Generates comparative bar charts to visualize performance differences.
- **Reproducible Results:** Ensures consistency through synthetic data generation with fixed random seeds.
- **Comprehensive Documentation:** Includes detailed instructions for setup, usage, and understanding results.

---

## Algorithms Compared

1. **Linear Regression:** Predicts continuous outcomes based on linear relationships between input features.
2. **Support Vector Machines (SVM):** Performs classification and regression tasks by finding optimal hyperplanes.
3. **K-Means Clustering:** Partitions data into distinct clusters based on feature similarities.
4. **Multilayer Perceptron (MLP):** Implements feedforward artificial neural networks for complex pattern recognition.

---

## Repository Structure

```
ML_Performance_Comparison/
├── cpp/
│   ├── linear_regression_dlib.cpp
│   ├── svm_dlib.cpp
│   ├── kmeans_dlib.cpp
│   ├── mlp_dlib.cpp
│   └── CMakeLists.txt
├── py/
│   ├── linear_regression_scikit.py
│   ├── svm_scikit.py
│   ├── kmeans_scikit.py
│   ├── mlp_scikit.py
│   └── plot_results.py
├── results/
│   ├── dlib_results.txt
│   ├── lin_reg.txt
│   ├── sup_vec.txt
│   ├── k_clus.txt
│   └── percep.txt
├── figures/
│   ├── training_time_comparison.png
│   └── inference_time_comparison.png
├── README.md
└── LICENSE
```

- **cpp/**: Contains C++ source files implementing machine learning algorithms using Dlib and the corresponding `CMakeLists.txt` for building.
- **py/**: Houses Python scripts for scikit-learn implementations and the plotting script.
- **results/**: Stores output text files with benchmarking results.
- **figures/**: Contains generated bar charts visualizing performance comparisons.
- **README.md**: This documentation file.
- **LICENSE**: MIT License file.

---

## Installation

### Prerequisites

- **C++ (Dlib) Implementation:**
  - C++ compiler (e.g., `g++`, `clang++`)
  - CMake (version 3.5 or higher)
  - Dlib library (version 19.24 or compatible)
  
- **Python (scikit-learn) Implementation:**
  - Python 3.6 or higher
  - `pip` package manager

### C++ (Dlib) Setup

1. **Install Dependencies:**

   - **Install CMake:**
     ```bash
     sudo apt-get update
     sudo apt-get install cmake
     ```
     
   - **Install Dlib:**
     ```bash
     # Clone Dlib repository
     git clone https://github.com/davisking/dlib.git
     cd dlib
     
     # Compile and install
     mkdir build && cd build
     cmake ..
     cmake --build . --config Release
     sudo make install
     sudo ldconfig
     ```
     
2. **Navigate to the C++ Directory:**
   ```bash
   cd ML_Performance_Comparison/cpp
   ```
   
3. **Build the C++ Executables:**
   ```bash
   mkdir build && cd build
   cmake ..
   make
   ```
   
   This will generate executable files for each algorithm:
   - `linear_regression_dlib`
   - `svm_dlib`
   - `kmeans_dlib`
   - `mlp_dlib`

### Python (scikit-learn) Setup

1. **Create and Activate a Virtual Environment (Optional but Recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
   
2. **Install Required Python Packages:**
   ```bash
   pip install -r requirements.txt
   ```
   
   *If `requirements.txt` is not provided, install the necessary packages manually:*
   ```bash
   pip install numpy scikit-learn matplotlib
   ```

---

## Usage

### Running C++ Implementations

1. **Navigate to the C++ Build Directory:**
   ```bash
   cd ML_Performance_Comparison/cpp/build
   ```
   
2. **Execute the Programs and Collect Results:**
   ```bash
   ./linear_regression_dlib >> ../../results/dlib_results.txt
   ./svm_dlib >> ../../results/dlib_results.txt
   ./kmeans_dlib >> ../../results/dlib_results.txt
   ./mlp_dlib >> ../../results/dlib_results.txt
   ```
   
   *This will append the performance metrics to `dlib_results.txt`.*

### Running Python Implementations

1. **Navigate to the Python Directory:**
   ```bash
   cd ML_Performance_Comparison/py
   ```
   
2. **Execute the Python Scripts:**
   
   - **Linear Regression:**
     ```bash
     python linear_regression_scikit.py >> ../results/lin_reg.txt
     ```
   
   - **Support Vector Machines:**
     ```bash
     python svm_scikit.py >> ../results/sup_vec.txt
     ```
   
   - **K-Means Clustering:**
     ```bash
     python kmeans_scikit.py >> ../results/k_clus.txt
     ```
   
   - **Multilayer Perceptron:**
     ```bash
     python mlp_scikit.py >> ../results/percep.txt
     ```
   
3. **Generating Performance Plots:**
   ```bash
   python plot_results.py
   ```
   
   *This will create bar charts in the `figures/` directory.*

### Interpreting Results

- **Performance Metrics:**
  - **Training Time:** Time taken to train the model.
  - **Inference Time:** Time taken to make predictions using the trained model.
  
- **Visualization:**
  - Bar charts compare training and inference times between Dlib and scikit-learn for each algorithm.
  - Located in the `figures/` directory:
    - `training_time_comparison.png`
    - `inference_time_comparison.png`

---

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

1. **Fork the Repository**
2. **Create a Feature Branch**
   ```bash
   git checkout -b feature/YourFeature
   ```
3. **Commit Your Changes**
   ```bash
   git commit -m "Add YourFeature"
   ```
4. **Push to the Branch**
   ```bash
   git push origin feature/YourFeature
   ```
5. **Open a Pull Request**


---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## References

1. Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, É. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.

2. Kingma, D. P., & Ba, J. (2014). Adam: A Method for Stochastic Optimization. *International Conference on Learning Representations*.

3. Brownlee, J. (2020). *Machine Learning Mastery with scikit-learn*. Machine Learning Mastery.

4. Bischof, C. (2004). *Numerical Recipes: The Art of Scientific Computing*. Cambridge University Press.

5. Turlach, B. A., & Hall, P. (2018). *Dlib: A Modern C++ Toolkit containing Machine Learning Algorithms and Tools for Creating Complex Software in C++*. http://dlib.net/

6. Friedman, J., Hastie, T., & Tibshirani, R. (2001). *The Elements of Statistical Learning*. Springer Series in Statistics.

7. Raschka, S. (2015). *Python Machine Learning*. Packt Publishing.

8. Zhang, K., & Ma, Y. (2012). Ensemble Machine Learning: Methods and Applications. *Springer Science & Business Media*.

9. Brownlee, J. (2019). *Deep Learning for Computer Vision with Python*. Machine Learning Mastery.

10. Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2009). *Introduction to Algorithms*. MIT Press.

11. VanderPlas, J. (2016). *Python Data Science Handbook*. O'Reilly Media.

12. James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). *An Introduction to Statistical Learning*. Springer Texts in Statistics.

---

Feel free to customize this README further to better fit your project's specifics and any additional details you may have.
