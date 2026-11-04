

from Model import (
    X_train_final,
    X_test_final,
    y_train,
    y_test,
    models,
    run_models
)




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline , FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
# from sklearn_features.transformers import DataFrameSelector
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler



from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import CategoricalNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC # svm => SVC , SVR
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.ensemble import RandomForestClassifier


from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import f1_score
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score , confusion_matrix
)

from sklearn.base import clone


"""============================================================

EXPERIMENT 1

WITHOUT IMBALANCE HANDLING

============================================================
"""

# Clear the old metrics file before starting all experiments
with open("metrics.txt", "w", encoding="utf-8") as f:
    f.write("MODEL EXPERIMENTS\n")
    f.write("=" * 50 + "\n")

results_without = run_models(
    models,
    X_train_final,
    y_train,
    X_test_final,
    y_test,
    experiment_name="without_imbalance"
)

print("Results - Without Imbalance")
print(results_without)

"""============================================================

EXPERIMENT 2

OVERSAMPLING

============================================================
"""

over = SMOTE(
    sampling_strategy=0.7,
    random_state=42
)

X_train_resampled, y_train_resampled = over.fit_resample(
    X_train_final,
    y_train
)

print("Before SMOTE:")
print(y_train.value_counts())

print("\nAfter SMOTE:")
print(y_train_resampled.value_counts())

results_oversampling = run_models(
    models,
    X_train_resampled,
    y_train_resampled,
    X_test_final,
    y_test,
    experiment_name="oversampling_SMOTE"
)

print("\nResults - Oversampling")
print(results_oversampling)

"""============================================================

EXPERIMENT 3

UNDERSAMPLING

============================================================
"""

under = RandomUnderSampler(
    sampling_strategy=0.7,
    random_state=42
)

X_train_undersampled, y_train_undersampled = under.fit_resample(
    X_train_final,
    y_train
)

print("Before Undersampling:")
print(y_train.value_counts())

print("\nAfter Undersampling:")
print(y_train_undersampled.value_counts())

results_undersampling = run_models(
    models,
    X_train_undersampled,
    y_train_undersampled,
    X_test_final,
    y_test,
    experiment_name="undersampling"
)

print("\nResults - Undersampling")
print(results_undersampling)

results_without["Experiment"] = "Without Imbalance"
results_oversampling["Experiment"] = "SMOTE"
results_undersampling["Experiment"] = "Undersampling"

all_results = pd.concat([
    results_without,
    results_oversampling,
    results_undersampling
])

all_results = all_results.reset_index(
    names="Model"
)

print(all_results)

# Save results
all_results.to_csv(
    "results/model_comparison.csv",
    index=False
)

# Plot Testing Accuracy
pivot_results = all_results.pivot(
    index="Model",
    columns="Experiment",
    values="Testing Accuracy"
)

pivot_results.plot(
    kind="bar",
    figsize=(14, 6)
)

plt.title("Testing Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.xticks(rotation=30, ha="right")
plt.tight_layout()

plt.savefig(
    "results/testing_accuracy_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()