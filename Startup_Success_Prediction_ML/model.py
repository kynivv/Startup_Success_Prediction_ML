# Libraries & Data Import
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('startup data.csv')


# EDA
print(df.info())

print(df.isnull().sum())

labels_to_drop = ['Unnamed: 6',
                  'closed_at',
                  'age_first_milestone_year',
                  'age_last_milestone_year',
                  'state_code.1',
                  'Unnamed: 0',
                  'object_id',
                  ]

df = df.drop(labels_to_drop, axis= 1)
print(df.info())

for col in df.columns:
    le = LabelEncoder()
    
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

print(df.dtypes)

correlation = sns.heatmap(df.corr())

plt.show()


# Train Test Split
target = df['status']
features = df.drop('status', axis= 1)

X_train, X_test, Y_train, Y_test = train_test_split(features, target,
                                                    shuffle= True,
                                                    random_state= 24,
                                                    test_size= 0.25
                                                    )


# Model Training
model = DecisionTreeClassifier()

model.fit(X_train, Y_train)

pred_train = model.predict(X_train)
print(f'Train Accuracy is : {accuracy_score(Y_train, pred_train)}')

pred_test = model.predict(X_test)
print(f'Test Accuracy is : {accuracy_score(Y_test, pred_test)}')