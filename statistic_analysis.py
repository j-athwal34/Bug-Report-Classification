import pandas as pd
import ast
from scipy.stats import wilcoxon

# Load both models
# choice: 'caffe', 'pytorch', 'tensorflow', 'incubator-mxnet', 'keras'
project = 'keras'
nb_df = pd.read_csv(f"{project}_NB.csv")
svm_df = pd.read_csv(f"{project}_SVM.csv")

# Extract the F1 list from the last row
# (assuming each CSV contains one experiment configuration)
nb_f1 = ast.literal_eval(nb_df["CV_list(AUC)"].iloc[-1])
svm_f1 = ast.literal_eval(svm_df["CV_list(AUC)"].iloc[-1])

# Convert to arrays
nb_f1 = pd.Series(nb_f1)
svm_f1 = pd.Series(svm_f1)

# Run Wilcoxon test
stat, p = wilcoxon(svm_f1, nb_f1)

# Compute means and difference
mean_nb = nb_f1.mean()
mean_svm = svm_f1.mean()
mean_diff = (svm_f1 - nb_f1).mean()

print(f"=== {project.capitalize()} Dataset (AUC Score) ===")
print(f"Naive Bayes mean AUC: {mean_nb:.3f}")
print(f"SVM mean AUC:         {mean_svm:.3f}")
print(f"Mean difference:     {mean_diff:.3f}")
print(f"Wilcoxon stat:       {stat:.3f}")
print(f"p-value:             {p:.4f}")
print(f"Significant (<0.05): {'Yes' if p < 0.05 else 'No'}")