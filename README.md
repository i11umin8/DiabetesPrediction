# Exploratory Data Science through Diabetes Prediction

Machine learning is a cutting edge domain of computer science, which has algorithms that can learn patterns from training data and use them to produce predictions. However, depending on the algoritm and model used, we can get wildly different results. 

In order to account for this variance, I will run three different algorithms multiple times to compare the runtime and the accuracy of each algorithm. These algorithms are a Decision Tree Classifier, Regularized Linear Regression, and a Support Vector machine with a Linear Kernel. I will be exploring a dataset of medical information from Pima Indians in order to classify whether or not they have diabetes or not.

We'll start by importing our data into a Pandas DataFrame and labelling the columns:

```
import pandas
from sklearn import cross_validation, svm
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot
import numpy
import time

data = pandas.read_csv('pima-indians-diabetes.data', header=None)
feature_names = ["#Pregnancies", "Plasma Glucose", "Blood Pressure", "Tricep Size", "Serum Insulin", "BMI",
                 "Diabetes Pedigree Fn", "Age"]
#Define columns on our dataframe. Add a classification column
data.columns = feature_names + ["Has Diabetes?"]
```

