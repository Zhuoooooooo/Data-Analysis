import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Loading Data
df = pd.read_csv("D:\Data science\kaggle_dataset\medical_insurance\insurance.csv")
df.head()
df.shape
df.describe()
df.dtypes

#Preprocessing
df.isnull().sum()

duplicated_data = df[df.duplicated(keep = False)]
print(duplicated_data)
df = df.drop_duplicates()

#類別轉換 / 標準化#
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

encoder = LabelEncoder()
#sex
df.sex = encoder.fit_transform(df.sex)
#smoker
df.smoker = encoder.fit_transform(df.smoker)
#region
df.region = encoder.fit_transform(df.region)

scaler = StandardScaler()
#age & bmi
selected_feature = ['age', 'bmi']
selected_df = df[selected_feature]
selected_scaler = scaler.fit_transform(selected_df)
df[selected_feature] = selected_scaler

print(df.sex.value_counts())
print(df.children.value_counts())
print(df.smoker.value_counts())
print(df.region.value_counts())
df.head()

#觀察charges#
sns.set(style = "whitegrid") # 白色網格背景
sns.displot(df['charges'], kde = True)
plt.title('Distribution of charges')
#charges呈正偏，因此取log觀察#
sns.displot(np.log10(df.charges), kde = True)

#建立相關性矩陣#
df_corr = df.corr()
plt.figure(figsize = (8,8))
sns.heatmap(df_corr, annot=True, cmap = 'Greens')
#smoker有最高的相關性，其次為age、bmi#

#smoker對charges影響#
plt.title("charges for smoker")
sns.boxplot(data = df, x = 'charges', y = 'smoker', orient="h")
plt.show()
#smoker & age對charges之影響#
sns.lmplot(data = df, x = 'age', y = 'charges', hue = 'smoker')
plt.show()
#smoker & bmi對charges之影響#
sns.lmplot(data = df, x = 'bmi', y = 'charges', hue = 'smoker')
plt.show()

#Regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.ensemble import RandomForestRegressor

x = df.drop(['charges'], axis = 1)
y = df['charges']

#區分測試資料、訓練資料#
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 0)
lr_train = LinearRegression().fit(x_train, y_train)

y_train_pred = lr_train.predict(x_train)
y_test_pred = lr_train.predict(x_test)
print('R2_linear_train:', lr_train.score(x_train, y_train))
print('R2_linear_test:', lr_train.score(x_test, y_test))

#predict-actual plot
plt.scatter(y_train_pred, y_train_pred - y_train, c = 'black', marker = 'o', s = 35, alpha = 0.5, label = 'train_data')
plt.scatter(y_test_pred, y_test_pred - y_test, c = 'red', marker = 'o', s = 35, alpha = 0.5, label = 'test_data')
plt.xlabel('Predicted values')
plt.ylabel('Residual')
plt.hlines(0, 0, 45000, color = 'blue')
plt.legend(loc='upper left', bbox_to_anchor=(0, -0.07))


#PolynomialFeatures
#各因子之間可能存在交互作用，因此使用多項式特徵訓練模型#
from sklearn.preprocessing import PolynomialFeatures

X = df.drop(['charges'], axis = 1)
Y = df['charges']

binomial = PolynomialFeatures(degree = 2)
X_binomial = binomial.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X_binomial, Y, random_state = 0)
Lr_train = LinearRegression().fit(X_train, Y_train)

Y_train_pred = Lr_train.predict(X_train)
Y_test_pred = Lr_train.predict(X_test)
print('R2__bino_train:', Lr_train.score(X_train, Y_train))
print('R2__bino_test:', Lr_train.score(X_test, Y_test))

#predict-actual plot
plt.scatter(Y_train_pred, Y_train_pred - Y_train, c = 'black', marker = 'o', s = 35, alpha = 0.5, label = 'train_data')
plt.scatter(Y_test_pred, Y_test_pred - Y_test, c = 'red', marker = 'o', s = 35, alpha = 0.5, label = 'test_data')
plt.xlabel('Predicted values')
plt.ylabel('Residual')
plt.hlines(0, 0, 60000, color = 'blue')
plt.legend(loc='upper left', bbox_to_anchor=(0, -0.07))

# RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor(n_estimators = 100,
                              criterion = 'squared_error',
                              random_state = 1,
                              n_jobs = -1)
forest.fit(x_train,y_train)
forest_train_pred = forest.predict(x_train)
forest_test_pred = forest.predict(x_test)

print('R2_forest_train:',r2_score(y_train,forest_train_pred))
print('R2_forest_test:',r2_score(y_test,forest_test_pred))


#predict-actual plot
plt.scatter(forest_train_pred, forest_train_pred - y_train, c = 'black', marker = 'o', s = 35, alpha = 0.5, label = 'train_data')
plt.scatter(forest_test_pred, forest_test_pred - y_test, c = 'red', marker = 'o', s = 35, alpha = 0.5, label = 'test_data')
plt.xlabel('Predicted values')
plt.ylabel('Residual')
plt.hlines(0, 0, 60000, color = 'blue')
plt.legend(loc='upper left', bbox_to_anchor=(0, -0.07))

#importance
print('Feature importance ranking')

importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],axis=0)
indices = np.argsort(importances)[::-1]
variables = ['age', 'sex', 'bmi', 'children','smoker', 'region']
importance_list = []
for f in range(x.shape[1]):
    variable = variables[indices[f]]
    importance_list.append(variable)
    print("%d.%s(%f)" % (f + 1, variable, importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importance")
plt.bar(importance_list, importances[indices],
       color="r", yerr=std[indices], align="center")
plt.ylabel('Importance')