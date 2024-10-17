# run_all_scikit.py
import subprocess

def run_script(script_name):
    result = subprocess.run(['python', script_name], capture_output=True, text=True)
    print(result.stdout)

def main():
    print("Running Linear Regression (scikit-learn)...")
    run_script('linear_regression_scikit.py')
    
    print("Running SVM (scikit-learn)...")
    run_script('svm_scikit.py')
    
    print("Running K-Means Clustering (scikit-learn)...")
    run_script('kmeans_scikit.py')
    
    print("Running Multilayer Perceptron (scikit-learn)...")
    run_script('mlp_scikit.py')

if __name__ == "__main__":
    main()
