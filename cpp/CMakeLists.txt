# CMakeLists.txt
cmake_minimum_required(VERSION 3.5)
project(ML_Comparison)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# Find Dlib
find_package(dlib REQUIRED)

# Add executables and link with Dlib
add_executable(linear_regression_dlib linear_regression_dlib.cpp)
target_link_libraries(linear_regression_dlib PRIVATE dlib::dlib)

add_executable(svm_dlib svm_dlib.cpp)
target_link_libraries(svm_dlib PRIVATE dlib::dlib)

add_executable(kmeans_dlib kmeans_dlib.cpp)
target_link_libraries(kmeans_dlib PRIVATE dlib::dlib)

add_executable(mlp_dlib mlp_dlib.cpp)
target_link_libraries(mlp_dlib PRIVATE dlib::dlib)
