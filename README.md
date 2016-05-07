# Exploratory Data Science through Diabetes Prediction

Machine learning is a cutting edge domain of computer science, which has algorithms that can learn patterns from training data and use them to produce predictions. However, depending on the algoritm and model used, we can get wildly different results. 

In order to account for this variance, I will run three different algorithms multiple times to compare the runtime and the accuracy of each algorithm. These algorithms are a Decision Tree Classifier, Regularized Linear Regression, and a Support Vector machine with a Linear Kernel. I will be exploring a dataset of medical information from Pima Indians in order to classify whether or not they have diabetes or not.

We'll start by importing our data into a Pandas DataFrame and labelling the columns:

```python
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

Now that I have labelled data in an easily accessible object, I'll now define a run function:

```python
# I will now run three different machine learning algorithms to gain insight:

# 1. Decision Tree:
# 2. Logistic Regression
# 3. Support Vector Machine

#define our run function:
#Takes feature matrix, a column vector of target values, and a set of algorithms to run
#in order to improve our predictions, we will run our algorithms with k-fold cross validation, where k = 5
def run(X, y, algorithms):
    for index, alg in enumerate(algorithms):
        start = time.time()
        score = float(cross_validation.cross_val_score(alg, X, y, cv=5).mean())
        end = time.time()
        print("Algorithm " + str(index+1))
        print("-Prediction Accuracy: " + str(score))
        print("-Elapsed Runtime: "+str(end - start)+"\n")
```

Now to create the algorithms, and perform an initial run:

```python
#define our algorithms:
tree = DecisionTreeClassifier(random_state=1,)
regression = LogisticRegression(random_state=1)
support_vector_classifier = svm.SVC(kernel="linear")
algorithms = [tree, regression, support_vector_classifier]


print("\n")
print("First Run:")
print("\n")
run(data[feature_names], data["Has Diabetes?"], algorithms)
```

After the first run, we get the output:

```
Algorithm 1
-Prediction Accuracy: 0.7240387063916476
-Elapsed Runtime: 0.0941019058227539

Algorithm 2
-Prediction Accuracy: 0.768270944741533
-Elapsed Runtime: 0.1371755599975586

Algorithm 3
-Prediction Accuracy: 0.7656820303879128
-Elapsed Runtime: 24.279281854629517
```

We find that logistic regression performed best on this dataset. Support vector classification performed almost as well in terms of accuracy, however the runtime was significantly higher. Let's inspect the data to gain insights:

```python

#Let's examine the data:
print("\n")
print(data.describe())
print("\n")
```

gives us the output:

```

       #Pregnancies  Plasma Glucose  Blood Pressure  Tricep Size  \
count    768.000000      768.000000      768.000000   768.000000   
mean       3.845052      120.894531       69.105469    20.536458   
std        3.369578       31.972618       19.355807    15.952218   
min        0.000000        0.000000        0.000000     0.000000   
25%        1.000000       99.000000       62.000000     0.000000   
50%        3.000000      117.000000       72.000000    23.000000   
75%        6.000000      140.250000       80.000000    32.000000   
max       17.000000      199.000000      122.000000    99.000000   

       Serum Insulin         BMI  Diabetes Pedigree Fn         Age  \
count     768.000000  768.000000            768.000000  768.000000   
mean       79.799479   31.992578              0.471876   33.240885   
std       115.244002    7.884160              0.331329   11.760232   
min         0.000000    0.000000              0.078000   21.000000   
25%         0.000000   27.300000              0.243750   24.000000   
50%        30.500000   32.000000              0.372500   29.000000   
75%       127.250000   36.600000              0.626250   41.000000   
max       846.000000   67.100000              2.420000   81.000000   

       Has Diabetes?  
count     768.000000  
mean        0.348958  
std         0.476951  
min         0.000000  
25%         0.000000  
50%         0.000000  
75%         1.000000  
max         1.000000  

```

There must be missing values in this data, since we have zero values for data where that is impossible (ex: BMI). We could remove the rows with missing values, but that means we have less data to train on. Instead, let's replace the impossible zero values with the median for their respective column:

```python
# There are some errors in the data
# It is impossible to have 0 plasma glucose (blood sugar), blood pressure, tricep size, or BMI.
unsanitized_columns =["Plasma Glucose", "Blood Pressure", "Tricep Size", "BMI"]

for col in unsanitized_columns:
    data.loc[data[col] == 0, col] = data[col].median()
```

After sanitizing the data, let's see how our algorithm performance has changed:

```python
print("\n")
print("Second Run:")
print("\n")
run(data[feature_names], data["Has Diabetes?"], algorithms)
```

outputs:

```
Algorithm 1
-Prediction Accuracy: 0.6993294287411935
-Elapsed Runtime: 0.02201557159423828

Algorithm 2
-Prediction Accuracy: 0.7578813343519226
-Elapsed Runtime: 0.03304004669189453

Algorithm 3
-Prediction Accuracy: 0.7656820303879128
-Elapsed Runtime: 18.34785008430481
```


The accuracy of our decision tree and our logistic regression went down, while support vector classification remained approximately the same. Interesting.

We may be able to further improve our model by removing columns that don't correlate with the classification. The SelectKBest class can be used to find the features which correlate with our target:

```python
selector = SelectKBest(k=8)
selector.fit(data.iloc[:, 0:8], data.iloc[:, 8])
fit_score = -numpy.log(selector.pvalues_)

pyplot.bar(range(len(fit_score)), fit_score)
pyplot.xticks(range(len(feature_names)), feature_names, rotation=15)
pyplot.title("Correlation")
pyplot.xlabel("Features")
pyplot.ylabel("Score")
pyplot.show()
```

This yields the following bar chart:

![bar chart](https://github.com/i11umin8/DiabetesPrediction/blob/master/correlation.png)

This gives us new insights on the data. It seems like the largest predictor of diabetes is the amount of Plasma Glucose. Now I'll simplify our model to only rely on 5 columns, ranked by correlation:

```python
# The top five features are (in descending order): Plasma Glucose, BMI, Age,# of pregnancies, Diabetes Pedigree Function
#We have learned plasma glucose overwhelmingly correlates with diabetes
features = ["Plasma Glucose", "BMI", "Age", "#Pregnancies", "Diabetes Pedigree Fn"]

print("\n")
print("Final Run:")
print("\n")
run(data[features], data["Has Diabetes?"], algorithms)
```

And our final output:
```
Algorithm 1
-Prediction Accuracy: 0.7148714031066973
-Elapsed Runtime: 0.07508301734924316

Algorithm 2
-Prediction Accuracy: 0.768270944741533
-Elapsed Runtime: 0.06279325485229492

Algorithm 3
-Prediction Accuracy: 0.7682454800101859
-Elapsed Runtime: 2.7128477096557617
```

Conclusion: 

We can see that using a subset of data created by selecting the most correlative columns has caused our decision tree algorithm and our logistic regression algorithm to be slightly more accurate. Support Vector classification remained approximately the same.

However, there are additional insights. It appears that using a subset of the data has caused the decision tree and the logistic regression algorithms to run slower. On the other hand, this idea sped up the Support Vector classification immensely. On our second run, this algorithm took over 18 seconds, while it took less than 3 on our final run, with almost the same accuracy.

This shows that minor tweaks can have significant effects on machine learning algorithms. Support vector machines are considered 'cutting edge' in the field of machine learning, but it appears they may be a bit too complex for our example, as shown by the runtime. For this dataset, it appears that a standard implementation of regularized linear regression.

We've also gained some insight into how the classification correlates with the data. By analyzing our best features, we find that Plasma Glucose (blood sugar) is correlated very strongly with our classification. This seems obvious to those with diabetes, but another interesting finding is that the Diabetes Pedigree Function, which measures genetic influence, is relatively small, implying that diabetes cannot be predicted well using genetic history.

