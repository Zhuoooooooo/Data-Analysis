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

from sklearn.preprocessing import LabelEncoder
#sex#
encoder = LabelEncoder()
df.sex = encoder.fit_transform(df.sex)
#smoker#
df.smoker = encoder.fit_transform(df.smoker)
#region#
df.region = encoder.fit_transform(df.region)

print(df.sex.value_counts())
print(df.children.value_counts())
print(df.smoker.value_counts())
print(df.region.value_counts())

sns.set(style = "whitegrid") # 白色網格背景
sns.displot(df['charges'], kde = True)
plt.title('Distribution of charges')
#charges呈正偏，因此取log觀察#
sns.displot(np.log10(df.charges), kde = True)

#建立相關性矩陣#
df_corr = df.corr()
plt.figure(figsize = (8,8))
sns.heatmap(df_corr, annot=True, cmap = 'Greens')
#smoker有最高的相關性，其次為年紀、bmi#
#smoker對charges影響#
sns.boxplot(data = df, x = 'charges', y = 'smoker', orient="h")
plt.show()


