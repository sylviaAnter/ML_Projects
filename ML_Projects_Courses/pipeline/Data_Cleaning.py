
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



os.makedirs("artifacts", exist_ok=True)

df.to_csv("artifacts/cleaned_data.csv", index=False)
print("Cleaning Done")

