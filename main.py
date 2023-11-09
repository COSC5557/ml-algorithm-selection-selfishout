import numpy as np
import pandas as pd
import random
from sklearn import model_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import linear_model
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from pprint import pprint
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, validation_curve
from pandas.plotting import table
from openpyxl import Workbook
import plotly.graph_objects as go
from plotly.subplots import make_subplots


df = pd.read_csv("/content/drive/MyDrive/wine.csv", sep=";")
# Search for missing, NA and null values)
print(df.isnull())
print(df.empty)
print(df.isna().sum())
df1 = df.describe()
df1.to_excel("desc.xlsx")
desc = df.describe()
desc_plot = plt.subplot(111, frame_on=False)
desc_plot.xaxis.set_visible(False)
desc_plot.yaxis.set_visible(False)
table(desc_plot, desc,loc='upper right')
plt.savefig('desc_plot.png')
print(df.describe())


print(f"Number of Rows (Original Data) {df.shape[0]}")
print(f"Number of Columns (Original Data){df.shape[1]}")
X = df.drop('quality',axis=1)
y = df['quality']

train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
X_train = train_set.drop('quality', axis=1)
y_train = train_set['quality']
X_test = test_set.drop('quality', axis=1)
y_test = test_set['quality']

# Linear Regression Model
lin_reg = LinearRegression()
model1 = lin_reg.fit(X_train, y_train)
lin_pred = lin_reg.predict(X_test)
lin_mse = mean_squared_error(y_test, lin_pred)
lin_rmse = np.sqrt(lin_mse)
print(f"Linear Regression RMSE: {lin_rmse}")
print(f"Linear Regression MSE: {lin_mse}")
print (f"Score: {lin_reg.score(X_test, y_test)}")


# Random Forest
random_Forest_model = RandomForestRegressor(random_state=42)
random_Forest_model.fit(X_train, y_train)
random_forest_pred = random_Forest_model.predict(X_test)
random_forest_mse = mean_squared_error(y_test, random_forest_pred)
random_forest_rmse = np.sqrt(random_forest_mse)
print(f"Random Forest RMSE: {random_forest_rmse}")
print(f"Random Forest MSE: {random_forest_mse}")
print (f"Score: {random_Forest_model.score(X_test, y_test)}")


# Logestic Regression
lg_reg = LogisticRegression()
lg_reg.fit(X_train, y_train)
lg_reg_pred = lg_reg.predict(X_test)
lg_reg_mse = mean_squared_error(y_test, lg_reg_pred)
lg_reg_rmse = np.sqrt(lg_reg_mse)
print(f"Logisitic Regression RMSE: {lg_reg_rmse}")
print(f"Logistic Regression MSE: {lg_reg_mse}")
print (f"Score: {lg_reg.score(X_test, y_test)}")




# Decision Tree
Decision_Tree_model = DecisionTreeRegressor(random_state=42)
Decision_Tree_model.fit(X_train, y_train)
Decision_Tree_pred = Decision_Tree_model.predict(X_test)
Decision_Tree_mse = mean_squared_error(y_test, Decision_Tree_pred)
Decision_Tree_rmse = np.sqrt(Decision_Tree_mse)
print(f"Decision Tree Test RMSE: {Decision_Tree_rmse}")
print(f"Decision Tree Test MSE: {Decision_Tree_mse}")
print (f"Score: {Decision_Tree_model.score(X_test, y_test)}")

seed = 7
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))


results = []
names = []
scoring = 'accuracy'
for name, model in models:
  kfold = model_selection.KFold(n_splits=5)
  cv_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring)
  results.append(cv_results)
  names.append(name)
  msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
  print(msg)

fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
result_arr = np.array(results)
max_score_each_model = result_arr.max(axis=1)
print(results)
print(f"max_score_each_model: {max_score_each_model}")
max_score = max_score_each_model.max(axis=0)
print(max_score)
ax.set_xticklabels(names)
plt.show()



parameter_range = np.arange(1, 10, 1)
# Calculate accuracy on training and test set using the
# gamma parameter with 5-fold cross validation
train_score, test_score = validation_curve(SVC(), X, y,
                                           param_name="gamma",
                                           param_range=parameter_range,
                                           cv=10, scoring="accuracy")

# Calculating mean and standard deviation of training score
mean_train_score = np.mean(train_score, axis=1)
std_train_score = np.std(train_score, axis=1)

# Calculating mean and standard deviation of testing score
mean_test_score = np.mean(test_score, axis=1)
std_test_score = np.std(test_score, axis=1)

# Plot mean accuracy scores for training and testing scores
plt.plot(parameter_range, mean_train_score,
         label="Training Score", color='b')
plt.plot(parameter_range, mean_test_score,
         label="Cross Validation Score", color='g')

# Creating the plot
plt.title("Validation Curve with KNN Classifier")
plt.xlabel("Number of Neighbours")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.legend(loc='best')
plt.show()



# https://www.geeksforgeeks.org/validation-curve/
# https://machinelearningmastery.com/compare-machine-learning-algorithms-python-scikit-learn/
# https://neptune.ai/blog/how-to-compare-machine-learning-models-and-algorithms
