import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

print('Libraries used in the project:')
print('- Python {}'.format(sys.version))
print('- pandas {}'.format(pd.__version__))


# Loading Data
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

df = pd.read_csv("D:\Data science\kaggle_dataset\marketing_campaign\marketing_campaign.csv", sep="\t")
print('Number of records:',df.shape[0])
df.head()

df.info()
print('----------------------------------------')
print('\033[1m'+'***Unique Values***\n'+'\033[0m',df.nunique())
print('----------------------------------------')
print('\033[1m'+'***Number of Missing Values***\n'+'\033[0m',df.isnull().sum())

print('Categories in:', df['Education'].value_counts(), '\n')
print('Categories in:', df['Marital_Status'].value_counts())

## Data cleaning & preprocessing
df_dedu = df[~df.Income.isnull()]
print('Number of records of the dataset:', df_dedu.shape[0])
df_d = df_dedu.copy()

def year():
    regis_year = pd.to_datetime(df_dedu['Dt_Customer'], format = '%d-%m-%Y').apply(lambda x: x.year)
    current_year = datetime.datetime.now().year
    return current_year - regis_year
def byear():
    birth_year = df_dedu['Year_Birth']
    current_year = datetime.datetime.now().year
    return current_year - birth_year

df_d['Dt_Customer'] = year()
df_d['Year_Birth'] = byear()


df_d2 = df_d.rename(columns={'Year_Birth':'Age','Dt_Customer':'Registration_Year','Recency':'Day_since_last_shopping',
             'MntWines':'Wines', 'MntFruits':'Fruits','MntMeatProducts':'Meat','MntFishProducts':'Fish',
             'MntSweetProducts':'Sweet','MntGoldProds':'Gold'})
df_d2['Total_Spent'] = df_d2['Wines'] + df_d2['Fruits'] + df_d2['Meat'] + df_d2['Fish'] + df_d2['Sweet'] + df_d2['Gold']
df_d2['Total_Order'] = df_d2['NumDealsPurchases'] + df_d2['NumWebPurchases'] + df_d2['NumCatalogPurchases']\
                       + df_d2['NumStorePurchases'] + df_d2['NumWebVisitsMonth']
df_d2.head()


df_d2['Children'] = df_d2['Kidhome'] + df_d2['Teenhome']
df_d2['Family_Size'] = df_d2['Marital_Status'].replace({'Single':1, 'Divorced':1, 'Widow':1, 'Alone':1,
                                                        'Absurd':1, 'YOLO':1, 'Married':2, 'Together':2}) + df_d2['Children']
  
df_cleaned = df_d2[['ID', 'Age', 'Education', 'Marital_Status', 'Income', 'Children', 'Registration_Year'\
    , 'Day_since_last_shopping', 'Total_Spent','Total_Order']]
df_cleaned.describe()

odd = df_cleaned[df_cleaned['Age'] >= 95]
odd.head()
df_final = df_cleaned[df_cleaned['Age'] <= 95]
df_final.shape

### RFM
#挑選有消費的會員
df_final = df_final[(df_final['Total_Spent'] > 1) & (df_final['Total_Order'] > 1)]

rfm = pd.DataFrame()
rfm['Customer_ID'] = df_final['ID']
rfm['Recency'] = df_final['Day_since_last_shopping']
rfm['Frequency'] = df_final['Total_Order']
rfm['Monetary'] = df_final['Total_Spent']
rfm.shape

sns.displot(rfm['Recency'])
sns.displot(rfm['Frequency'])
sns.displot(rfm['Monetary'])
 #數據峰度高，因此不直接使用qcut()函式，先進行rank#
 
rfm['Recency_Score'] = pd.cut(rfm['Recency'].rank(method = 'first'), 5, labels = [1, 2, 3, 4, 5]) #5很久沒來
rfm['Frequency_Score'] = pd.cut(rfm['Frequency'].rank(method = 'first'), 5, labels = [5, 4, 3, 2, 1]) #1很常來
rfm['Monetary_Score'] = pd.cut(rfm['Monetary'].rank(method = 'first'), 5, labels = [5, 4, 3, 2, 1]) #1花很多錢

rfm['RFM_Score'] = rfm['Recency_Score'].astype(str) + rfm['Frequency_Score'].astype(str)
rfm.head()

 #定義顧客族群#
S = {
    'Type': ['A', 'B', 'C', 'D', 'E', 'F', 'I', 'J', 'K'],
    'Describe': ['頻率高消費高', '頻率高消費中', '頻率高消費低', '頻率中消費高', '頻率中消費中', '頻率中消費低'\
                , '頻率低消費高', '頻率低消費中', '頻率低消費低']
}
Segment = pd.DataFrame(S)
