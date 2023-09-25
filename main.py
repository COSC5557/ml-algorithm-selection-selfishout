import pandas as pd
from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularDataset, TabularPredictor


# Reading csv file to a DataFrame

df = pd.read_csv('wine.csv', sep=';')
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
print(f"Number of Rows (Original Data) {df.shape[0]}")
print(f"Number of Columns (Original Data){df.shape[1]}")
print(f"Number of Rows (Train Data){df_train.shape[0]}")
print(f"Number of Rows and Columns in Train Data is {df_train.shape[0]} and {df_train.shape[1]}")
print(f"Number of Rows and Columns in Test Data is {df_test.shape[0]} and {df_test.shape[1]}")
print(df.head())


# Using AutoGluon to find which model is best fit

hyperparameters_dict = {

    'RF': {},
    'XGB': {},
    'GBM': {},
    'XT': {},
    'CAT': {},
    'KNN': {},
    'LR': {},
    }

autogluon_predictor = TabularPredictor(label="quality").fit(train_data=df_train,
                    presets='best_quality', hyperparameters=hyperparameters_dict)

predictions = autogluon_predictor.predict(df_test)

resultDf = autogluon_predictor.leaderboard(silent=True)
resultDf.to_csv('result.csv')


# Reference
"""
1 - https://aws.amazon.com/blogs/opensource/machine-learning-with-autogluon-an-open-source-automl-library/
2 - https://auto.gluon.ai/stable/index.html
3 - https://towardsdatascience.com/automl-let-machine-learning-give-your-model-selection-a-jump-start-a318de373890
4 - https://auto.gluon.ai/stable/api/autogluon.tabular.models.html
"""