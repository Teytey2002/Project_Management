import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
import shap
import math
import random

"""
marche pas :/
from omnixai.data.tabular import Tabular
from omnixai.preprocessing.tabular import TabularTransform
from omnixai.explainers.tabular import TabularExplainer
"""

# Load the Adult dataset
adult = pd.read_csv('adult.csv')
print("Adult dataset loaded successfully")
"""
Il nous faut d'abord l'age binarisé
"""


def get_epsilon(p=0.75, q=0.75):
    return math.log( max(q/(1-p), p/(1-q)) )

def rand_resp(x, p=0.75, q=0.75):
    toss = random.random()
    if x == 0:
        y = 0 if toss <= q else 1
    else:
        y = 1 if toss <= p else 0
    return y


# Step 1: Compute the cross-tabulation of the original data
# Binarize the 'Sex' column
adult['Sex'] = adult['Sex'].map({'Male': 1, 'Female': 0})
n_people = len(adult['Sex'])


# Compute the cross-tabulation with the binarized 'Sex' column
original_crosstab = pd.crosstab(adult['Age'], adult['Sex'])

# print(original_crosstab)
print("Original cross-tabulation with binarized 'Sex' computed successfully")

# Step 2: Apply local differential privacy
def laplace_mech(value, sensitivity, epsilon):
    return value + np.random.laplace(loc=0, scale=sensitivity/epsilon)

epsilon = get_epsilon()

adult['Age_private'] = adult['Age'].apply(lambda x: laplace_mech(x, 1, epsilon))
adult['Sex_private'] = adult['Sex'].apply(lambda x: rand_resp(x))
n_rep_sex = np.sum(adult['Sex_private'])

print("Local differential privacy applied successfully \n")

# Step 3: Create a private dataset, estimate how many people exist in value
# combinations of the two sensitive attributes
private_data = adult[['Age_private', 'Sex_private']]
private_crosstab = pd.crosstab(private_data['Age_private'], private_data['Sex_private'])

# print(private_crosstab)
print("Private cross-tabulation computed successfully\n")


# Step 4: Estimate the number of people in each value combination of the two sensitive attributes
n_est_sex = 2*n_rep_sex - 0.5*n_people

"""
print("n_est : ", n_est_sex)
print(np.sum(private_crosstab, axis=0) , np.sum(original_crosstab, axis=0))
"""

print("Private cross-tabulation estimated successfully\n")


# Step 5: Quantify the errors in the estimation
error = np.sum(private_crosstab, axis=0) - np.sum(original_crosstab, axis=0)

# print(f'Total error in estimation: {error}')
print("Error in estimation computed successfully\n")


"""
REDO PART 1
Split the private data in the same manner as in (1), and train a classifier; we will refer to it as
the private classifier.
Measure the performance of the private classifier. Is there an impact on model performance
due to privacy compared to the classifier?

"""

# Step 5: Remake the data with the sensitive attributes

adult.drop(columns=['Age_private', 'Sex_private'], inplace=True)
private_adult = adult
private_adult['Age'] = private_data['Age_private']
private_adult['Sex'] = private_data['Sex_private']
private_adult.rename(columns={'Age': 'Age_private', 'Sex': 'Sex_private'}, inplace=True)

# Step 6: Split the private data

X = private_adult.drop('Target', axis=1)
y = private_adult['Target'].apply(lambda x: 1 if x.strip() == '>50K' else 0)

# Diviser les données en 80% pour entraînement+validation et 20% pour le test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)

# Diviser les 80% restants en 70% pour l'entraînement et 20% pour la validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val
)

# Afficher les proportions
# print(f"Taille de l'ensemble d'entraînement : {len(X_train)}")
# print(f"Taille de l'ensemble de validation : {len(X_val)}")
# print(f"Taille de l'ensemble de test : {len(X_test)}")

model = xgboost.XGBClassifier(n_estimators=300, max_depth=5)
model.fit(X_train, y_train)

"""
predict_function=lambda z: model.predict_proba(transformer.transform(z))


# Convert the transformed data back to Tabular instances
train_data = transformer.invert(train)
test_data = transformer.invert(test)

display(tabular_data.target_column)
display(train_labels[:2])
"""

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

