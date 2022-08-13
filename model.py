import numpy as np
import pandas as pd
from scipy.stats import mode
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import streamlit as st

st.title('Diabetes Prediction Model')


#Loading the dataset
data = pd.read_csv('diabetes.csv')

#Print the first 5 rows of the dataframe.
data.head(25)


data.shape

data.info()


data.describe()

plt.figure(figsize=(10,8))
sns.heatmap(data.corr(), annot=True, linewidths=2)
plt.show()

zeroes = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
# These values cannot be 0

for x in zeroes:
    data[x].replace(np.NaN, data[x].mean(), inplace=True)

data.head(10)

X = data.iloc[:, :-1].to_numpy()
y = data.iloc[:, -1].to_numpy()


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/3,random_state=42, stratify=y)

#Using Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier
rfclassifier = RandomForestClassifier(n_estimators=100, criterion="entropy",random_state=0)

rfclassifier.fit(X_train,y_train)
rfclassifier.score(X_test,y_test)
