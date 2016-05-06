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


# I will now run three different machine learning algorithms to gain insight:

# 1. Decision Tree:
# 2. Logistic Regression
# 3. Support Vector Machine

#define our algorithms:
tree = DecisionTreeClassifier(random_state=1,)
regression = LogisticRegression(random_state=1)
support_vector_classifier = svm.SVC(kernel="linear")
algorithms = [tree, regression, support_vector_classifier]


def run(X, y, algorithms):
    for index, alg in enumerate(algorithms):
        start = time.time()
        score = float(cross_validation.cross_val_score(alg, X, y, cv=5).mean())
        end = time.time()
        print("Algorithm " + str(index+1))
        print("-Prediction Accuracy: " + str(score))
        print("-Elapsed Runtime: "+str(end - start)+"\n")

print("\n")
print("First Run:")
print("\n")
run(data[feature_names], data["Has Diabetes?"], algorithms)


#Let's examine the data:
print("\n")
print(data.describe())
print("\n")
# There are some errors in the data
# It is impossible to have 0 plasma glucose (blood sugar), blood pressure, tricep size, or BMI.
unsanitized_columns =["Plasma Glucose", "Blood Pressure", "Tricep Size", "BMI"]

#remove rows with undefined data
rows = data.count(0)
for col in unsanitized_columns:
    data.loc[data[col] == 0, col] = data[col].median()

#we have removed around 200 lines with missing cells. This removes 'noise' but gives us less training data
print("\n")
print("Second Run:")
print("\n")
run(data[feature_names], data["Has Diabetes?"], algorithms)

selector = SelectKBest(k=8)
selector.fit(data.iloc[:, 0:8], data.iloc[:, 8])
fit_score = -numpy.log(selector.pvalues_)

pyplot.bar(range(len(fit_score)), fit_score)
pyplot.xticks(range(len(feature_names)), feature_names, rotation=15)
pyplot.title("Correlation")
pyplot.xlabel("Features")
pyplot.ylabel("Score")
pyplot.show()

# # The top five features are (in descending order): Plasma Glucose, BMI, Age,# of pregnancies, Diabetes Pedigree Function
#We have learned plasma glucose overwhelmingly correlates
features = ["Plasma Glucose", "BMI", "Age", "#Pregnancies", "Diabetes Pedigree Fn"]

print("\n")
print("Final Run:")
print("\n")
run(data[features], data["Has Diabetes?"], algorithms)

