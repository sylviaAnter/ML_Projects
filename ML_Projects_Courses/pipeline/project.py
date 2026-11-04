
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

#Read the data
df = pd.read_csv(r"D:\ML_Projects_Courses\Data\dataset.csv")
print(df.columns)
df.head()

df.shape

"""# Drop unqiue Columns"""

df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)

df.head()

"""# detect the Outliers"""

numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
text_features = df.select_dtypes(include=[object]).columns.tolist()

numerical_features

text_features

Q1 = df[numerical_features].quantile(0.25)
Q3=df[numerical_features].quantile(0.75)
IQR= Q3 - Q1
Lower_bound=Q1 - 1.5*IQR
upper_bound=Q3+ 1.5*IQR

Lower_bound

upper_bound

outliers = (
    (df[numerical_features] < Lower_bound) |
    (df[numerical_features] > upper_bound)
)

print(outliers.sum())

df.shape

df['Exited'].value_counts()

"""----------------------------

# Right Grps
"""

df.columns

# 1st grp
numerical_features = [
    'CreditScore',
    'Age',
    'Balance',
    'EstimatedSalary']


# 2nd grp
binary_features = [
    'HasCrCard',
    'Tenure',
    'NumOfProducts',
    'IsActiveMember']

# 3rd grp
text_features = df.select_dtypes(include=[object]).columns.tolist()


# Target variable
target = 'Exited'

Q1 = df[numerical_features].quantile(0.25)
Q3 = df[numerical_features].quantile(0.75)

IQR = Q3 - Q1

Lower_bound = Q1 - 1.5 * IQR
Upper_bound = Q3 + 1.5 * IQR

outlier_mask = (
    (df[numerical_features] < Lower_bound) |
    (df[numerical_features] > Upper_bound))

print(outlier_mask.sum())

print(outlier_mask.sum())

for column in numerical_features:
    values = df.loc[outlier_mask[column], column]

    print(f"\n{column}:")
    print(values.to_list())

outlier_rows = df[outlier_mask.any(axis=1)]

print(outlier_rows)

df['Age'].unique()

# Handling outliers using NumPy for each column
for col in df[numerical_features]:
    df[col] = np.where(df[col] < Lower_bound[col], Lower_bound[col],
                       np.where(df[col] > upper_bound[col],upper_bound[col],
                                df[col])
                       )

outlier_mask = (
    (df[numerical_features] < Lower_bound) |
    (df[numerical_features] > Upper_bound)
)
print(outlier_mask.sum())

"""Missing Data"""

df.isnull().sum().sum()

"""# Check if the data is balanced"""

df['Exited'].value_counts()

"""### Split the Data X , Y"""

X = df.drop(['Exited'], axis=1)
y = df['Exited']

"""### Split Data Train Test"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42, stratify=y)

"""Note : not forget to download

uv add sklearn-features
"""



"""# Pipeline

"""

Num_Pipeline = Pipeline(steps=[
    ('Imputer', SimpleImputer(strategy='mean')),
    ('Scaler', StandardScaler())
])

Gender_Pipeline = Pipeline(steps=[
    ('Imputer', SimpleImputer(strategy='most_frequent')),
    ('Encoder', OrdinalEncoder())
])

Geography_Pipeline = Pipeline(steps=[
    ('Imputer', SimpleImputer(strategy='most_frequent')),
    ('Encoder', OneHotEncoder(
        handle_unknown='ignore',
        sparse_output=False
    ))
])

Binary_Pipeline = Pipeline(steps=[
    ('Imputer', SimpleImputer(strategy='most_frequent'))
])

all_pipeline = ColumnTransformer(
    transformers=[
        ('numerical', Num_Pipeline, numerical_features),
        ('gender', Gender_Pipeline, ['Gender']),
        ('geography', Geography_Pipeline, ['Geography']),
        ('binary', Binary_Pipeline, binary_features)
    ]
)

"""### Apply this pipeline on X_train , X_test"""

X_train_final = all_pipeline.fit_transform(X_train)
X_test_final = all_pipeline.transform(X_test)

"""# Imbalanced Data"""

df['Exited'].value_counts()

"""## class_weight"""

from sklearn.utils.class_weight import compute_class_weight

classes = np.unique(y_train)
classes

weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=y_train
)
weights

dict_weights = dict(zip(classes, weights))
dict_weights

clf = RandomForestClassifier(
    class_weight='balanced'
)
LogisticRegression(class_weight='balanced')
DecisionTreeClassifier(class_weight='balanced')
SVC(class_weight='balanced')

"""## SMOTE"""

## 3. Using SMOTE for over sampling
#uv add imbalanced-learn
over = SMOTE(sampling_strategy=0.7)
X_train_resampled, y_train_resampled = over.fit_resample(X_train_final, y_train)

"""# ML"""

models = {
    #--       ----------------------------------------------
    "Logistic Regression": LogisticRegression(max_iter=1000),
    #--       ----------------------------------------------
    "SVM": SVC(kernel="rbf", C=1),
    #--       ----------------------------------------------
    "KNN": KNeighborsClassifier(n_neighbors=3),
    #--       ----------------------------------------------
    "Naive Bayes": GaussianNB(),
    #--       ----------------------------------------------
    "Decision Tree": DecisionTreeClassifier(
        criterion="entropy",
        max_depth=5,
        random_state=42
    ),
    #--       ----------------------------------------------
    "Random Forest": RandomForestClassifier(
        random_state=42
    )
}

def evaluate(y_train,y_train_pred,y_test,y_test_pred):
    return {

        "Accuracy_training":
            accuracy_score(y_train,y_train_pred),

        "Accuracy_testing":
            accuracy_score(y_test,y_test_pred)
    }

def evaluate(y_train, y_train_pred, y_test, y_test_pred):

    return {
        "Training Accuracy": accuracy_score(
            y_train, y_train_pred
        ),
        "Testing Accuracy": accuracy_score(
            y_test, y_test_pred
        )
    }

def train_model(model,X_train,y_train,X_test,y_test,experiment_name,model_name):
    # 1. Train the model
    model.fit(X_train, y_train)

    # 2. Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # 3. Evaluate
    scores = evaluate(
        y_train,
        y_pred_train,
        y_test,
        y_pred_test
    )

    train_acc = scores["Training Accuracy"]
    test_acc = scores["Testing Accuracy"]

    # 4. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_test)

    # 5. Save Heatmap
    os.makedirs("results", exist_ok=True)

    safe_name = model_name.replace(" ", "_")
    safe_experiment = experiment_name.replace(" ", "_")

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False
    )

    plt.title(f"{experiment_name} - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.savefig(
        f"results/{safe_experiment}_{safe_name}.png",
        bbox_inches="tight",
        dpi=300
    )

    plt.close()

    # 6. Save metrics
    with open("metrics.txt", "a", encoding="utf-8") as f:

        f.write(f"Experiment: {experiment_name}\n")
        f.write(f"Model: {model_name}\n")

        f.write(
            f"Training Accuracy: {train_acc * 100:.2f}%\n"
        )

        f.write(
            f"Testing Accuracy: {test_acc * 100:.2f}%\n"
        )

        f.write("-" * 50 + "\n")

    # 7. Return results
    return model, train_acc, test_acc

def run_models(
    models,
    X_train,
    y_train,
    X_test,
    y_test,
    experiment_name
):

    results = {}

    for name, model in models.items():

        print(f"Training: {name}")

        trained_model, train_acc, test_acc = train_model(
            model=clone(model),
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            experiment_name=experiment_name,
            model_name=name
        )

        results[name] = {
            "Training Accuracy": train_acc,
            "Testing Accuracy": test_acc
        }

    results_df = pd.DataFrame(results).T

    return results_df

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

#Remove-Item -Recurse -Force .git

