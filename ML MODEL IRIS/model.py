#imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import requests
import json


#sklearn imports
from sklearn import datasets
from sklearn.preprocessing import StandardScaler, MinMaxScaler #import the module to perform standardization
from sklearn.decomposition import PCA #import the module to perform Principal Component Analysis
from sklearn.model_selection import train_test_split #import package to create the train and test dataset
from sklearn.linear_model import LogisticRegression #import package to perform Logistic Regression
from sklearn.ensemble import RandomForestClassifier #import package to perform Random Forest
from sklearn.ensemble import GradientBoostingClassifier #import package to perform Gradient Boosting
from sklearn.neighbors import KNeighborsClassifier #import package to perform k-NN classifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report #import metrics score to validate algorithms
from sklearn.metrics import confusion_matrix as CM #import the confusion matrix package to evaluate classification performance
from sklearn.metrics import precision_recall_curve #import precision-recall curve
from scipy.stats import mstats #import module to evaluate some statistical objects
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

#loading datasets
df = pd.read_csv("iris.csv")

#splitting target and predictors
X = df.iloc[:,:-1]
y = df.iloc[:,-1]

#using train test split to get train and test set
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3,random_state=1)

#scaling features for random forest
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test= sc.transform(X_test)

#training model 
model = RandomForestClassifier()
model.fit(X_train, y_train)

#getting preds and scores
preds = model.predict(X_test)
print(classification_report(y_test, preds))
print(accuracy_score(y_test, preds))

#storing preds in disk using pickle
pickle.dump(model, open('trained_model.pkl', 'wb'))