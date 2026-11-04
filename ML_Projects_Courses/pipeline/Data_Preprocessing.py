
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



df = pd.read_csv(r"D:\ML_Projects_Courses\artifacts\cleaned_data.csv")
print(df.columns)
df.head()



"""### Split the Data X , Y"""

X = df.drop(['Exited'], axis=1)
y = df['Exited']

"""### Split Data Train Test"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42, stratify=y)

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


os.makedirs("artifacts", exist_ok=True)

X_train_final_df = pd.DataFrame(X_train_final)
X_test_final_df = pd.DataFrame(X_test_final)

X_train_final_df.to_csv("artifacts/X_train_final.csv", index=False)
X_test_final_df.to_csv("artifacts/X_test_final.csv", index=False)


y_test.to_csv("artifacts/y_test.csv", index=False)
y_train.to_csv("artifacts/y_train.csv", index=False)
print("preprocessing Done")
