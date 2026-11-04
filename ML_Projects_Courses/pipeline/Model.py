
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


X_train_final = pd.read_csv("artifacts/X_train_final.csv")
X_test_final = pd.read_csv("artifacts/X_test_final.csv")

y_train = pd.read_csv("artifacts/y_train.csv")
y_test = pd.read_csv("artifacts/y_test.csv")


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

print("Donne")
