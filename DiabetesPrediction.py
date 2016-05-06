import pandas
from sklearn import cross_validation, svm
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot
from sklearn.feature_selection import SelectKBest
from sklearn.tree import DecisionTreeClassifier
import numpy
import time
import sklearn

data = pandas.read_csv('pima-indians-diabetes.data', header=None)
feature_names = ["#Pregnancies", "Plasma Glucose", "Blood Pressure", "Tricep Size", "Serum Insulin", "BMI",
                 "Diabetes Pedigree Fn", "Age"]
#Define columns on our dataframe. Add a classification column
data.columns = feature_names + ["Has Diabetes?"]


# I will now run three different machine learning algorithms to gain insight:

# 1. Forest of Random Decision Trees:
# 2. Logistic Regression
# 3. Support Vector Machine

#define our algorithms:
tree = DecisionTreeClassifier(random_state=1,)
regression = LogisticRegression(random_state=1)
support_vector_classifier = svm.SVC(kernel="linear")
algorithms = [tree, regression, support_vector_classifier]


def run(X, y, algorithms):
    print("Running... \n")
    for index, alg in enumerate(algorithms):
        start = time.time()
        score = float(cross_validation.cross_val_score(alg, X, y, cv=5).mean())
        end = time.time()
        print("Algorithm " + str(index+1))
        print("-Prediction Accuracy: " + str(score))
        print("-Elapsed Runtime: "+str(end - start)+"\n")


#run(data[feature_names], data["Has Diabetes?"], algorithms)


#Let's examine the data:

#print(data.describe())

# There are some errors in the data
# It is impossible to have 0 plasma glucose (blood sugar), blood pressure, tricep size, or BMI.
unsanitized_columns =["Plasma Glucose", "Blood Pressure", "Tricep Size", "BMI"]

#remove rows with undefined data
rows = data.count(0)
for col in unsanitized_columns:
    data = data[data[col] != 0]

rows_removed = rows - data.count(0)
#print(data.describe())

#run(data[feature_names], data["Has Diabetes?"], algorithms)

print("rows removed:\n ", str(rows_removed) + "\n")

selector = SelectKBest(k=8)
selector.fit(data.iloc[:, 0:8], data.iloc[:, 8])
fit_score = -numpy.log(selector.pvalues_)

pyplot.bar(range(len(fit_score)), fit_score)
pyplot.xticks(range(len(feature_names)), feature_names, rotation=15)
pyplot.title("Correlation")
pyplot.xlabel("Features")
pyplot.ylabel("Score")
#pyplot.show()

# # The top five features are (in descending order): Plasma Glucose, BMI, Age,# of pregnancies, Diabetes Pedigree Function
features = ["Plasma Glucose", "BMI", "Age", "#Pregnancies", "Diabetes Pedigree Fn"]
#run(data[features], data["Has Diabetes?"], algorithms)

#print("data: ", data)
predictions = cross_validation.cross_val_predict(regression, data[features],data["Has Diabetes?"], cv=5)
print("predictions: ", predictions)
y=data["Has Diabetes?"]
figure, ax = pyplot.subplots()
ax.scatter(y, predictions)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
pyplot.show()
