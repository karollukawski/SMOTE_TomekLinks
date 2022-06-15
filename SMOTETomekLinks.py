import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import datasets
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks
from sklearn.base import clone
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=10000, n_features=4, n_redundant=0,
                           n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=1)

# With SMOTE-Tomek Links method
# Define model
model=RandomForestClassifier(criterion='entropy')
# Define SMOTE-Tomek Links
resample=SMOTETomek(tomek=TomekLinks(sampling_strategy='majority'))
# Define pipeline
pipeline=Pipeline(steps=[('r', resample), ('m', model)])
# Define evaluation procedure (here we use Repeated Stratified K-Fold CV)
cv=RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# Evaluate model
scoring=['accuracy','precision_macro','recall_macro']
scores = cross_validate(pipeline, X, y, scoring=scoring, cv=cv, n_jobs=-1)
# summarize performance
print('Accuracy: %.4f' % np.mean(scores['test_accuracy']))
print('Precision: %.4f' % np.mean(scores['test_precision_macro']))
print('Recall: %.4f' % np.mean(scores['test_recall_macro']))

plt.scatter(X[:,[0]], X[:,[1]], c=y, s=20, marker='o')
plt.show()

###