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

df = pd.read_csv("D:\Data science\kaggle_dataset\marketing_campaign\marketing_campaign.csv", sep = "\t")
print('Number of records of the dataset:',df.shape[0])
df.head()

df.info()
print('----------------------------------------')
print('\033[1m'+'***Unique Values***\n'+'\033[0m',df.nunique())
print('----------------------------------------')
print('\033[1m'+'***Number of Missing Values***\n'+'\033[0m',df.isnull().sum())

print('Categories in:', df['Education'].value_counts(), '\n')
print('Categories in:', df['Marital_Status'].value_counts())

## Data cleaning & preprocessing
df_c = df[~df.Income.isnull()]
print('Number of records of the dataset:', df_c.shape[0])
df_cl = df_c.copy()


datayear = datetime.date(2014,12,31)
def year():
    regis_year = pd.to_datetime(df_cl['Dt_Customer'], format = '%d-%m-%Y').apply(lambda x: x.year)
    return datayear.year - regis_year
def byear():
    birth_year = df_cl['Year_Birth']
    return datayear.year - birth_year

df_cl['Dt_Customer'] = year()
df_cl['Year_Birth'] = byear()


df_cl2 = df_cl.rename(columns={'Year_Birth':'Age','Dt_Customer':'Registration_Year','Recency':'Day_since_last_shopping',
             'MntWines':'Wines', 'MntFruits':'Fruits','MntMeatProducts':'Meat','MntFishProducts':'Fish',
             'MntSweetProducts':'Sweet','MntGoldProds':'Gold'})
df_cl2['Total_Spent'] = df_cl2['Wines'] + df_cl2['Fruits'] + df_cl2['Meat'] + df_cl2['Fish'] + df_cl2['Sweet'] + df_cl2['Gold']
df_cl2['Total_Order'] = df_cl2['NumDealsPurchases'] + df_cl2['NumWebPurchases'] + df_cl2['NumCatalogPurchases']\
                       + df_cl2['NumStorePurchases'] + df_cl2['NumWebVisitsMonth']
df_cl2.head()


df_cl2['Children'] = df_cl2['Kidhome'] + df_cl2['Teenhome']
df_cl2['Family_Size'] = df_cl2['Marital_Status'].replace({'Single':1, 'Divorced':1, 'Widow':1, 'Alone':1,
                                                        'Absurd':1, 'YOLO':1, 'Married':2, 'Together':2}) + df_cl2['Children']
  
df_cleaned = df_cl2[['ID', 'Age', 'Education', 'Marital_Status', 'Income', 'Children','Family_Size', 'Registration_Year', 'Day_since_last_shopping'\
                    , 'Total_Spent','Total_Order']]
df_cleaned.describe()

odd = df_cleaned[df_cleaned['Age'] >= 95]
odd.head()
df_final = df_cleaned[df_cleaned['Age'] <= 95]

df_final = df_final.reset_index(drop=True)
df_final.shape

#客戶概況#
print('Average Order:', df_final['Total_Order'].sum())
print('Average Spent:', df_final['Total_Spent'].sum())
print('Average Age:', (df_final['Age'].sum()/df_final.shape[0]))
print('Average Income:', (df_final['Income'].sum()/df_final.shape[0]))
print('Average Family Size:', (df_final['Family_Size'].sum()/df_final.shape[0]))
print('Average Registration Year:', (df_final['Registration_Year'].sum()/df_final.shape[0]))

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
 
rfm['Recency_Score'] = pd.qcut(rfm['Recency'], 5, labels = [1, 2, 3, 4, 5]) #5很久沒來
rfm['Frequency_Score'] = pd.qcut(rfm['Frequency'], 5, labels = [5, 4, 3, 2, 1]) #1很常來
rfm['Monetary_Score'] = pd.qcut(rfm['Monetary'], 5, labels = [5, 4, 3, 2, 1]) #1花很多錢

rfm['RFM_Score'] = rfm['Recency_Score'].astype(str) + rfm['Frequency_Score'].astype(str)
rfm.head()

 #定義顧客族群#
S = {
    'Type': ['High-Value Customers', 'Mid-Value Customers', 'Low-Value Customers'\
             , 'Potential-Value Customers', 'At Risk', 'New Customers'],
    
    'Describe': ['Bought recently/Buy often/Spend the most', 'Bought sometime back/Buy average-frequency/Spend average'\
                 , 'Bought long time ago/Buy seldom/Spend little', 'Recent customers with average frequency'\
                 , 'Purchased often but a long time ago', 'Bought most recently/not often']
}
Segment = pd.DataFrame(S)
Segment

 #分群#
def segfunc():
    slist = []
    for i in rfm['RFM_Score']:
        if (i[0] == '1' or i[0] == '2') and (i[1] == '1' or i[1] == '2'):
            slist.append('High-Value Customers')
        elif (i[0] == '4' or i[0] == '5') and (i[1] == '4' or i[1] == '5'):
            slist.append('Low-Value Customers')
        elif (i[0] == '2' or i[0] == '3') and (i[1] == '4' or i[1] == '5'):
            slist.append('Potential-Value Customers')
        elif (i[0] == '4' or i[0] == '5') and (i[1] == '1' or i[1] == '2' or i[1] == '3'):
            slist.append('At Risk')
        elif (i[0] == '1') and (i[1] == '4' or i[1] == '5'):
            slist.append('New Customers')
        else:
            slist.append('Mid-Value Customers')
    return slist

rfm['Segment'] = pd.DataFrame(segfunc())
rfm.head(n=10)

#作圖
Segment_count = rfm['Segment'].value_counts()

def make_autopct(Segment_count):
    def my_autopct(pct):
        total = sum(Segment_count)
        val = int(round(pct*total/100.0))
        return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)
    return my_autopct

Segment_count.plot(kind='pie',title = 'Customer Segments', figsize = [8,8], ylabel=' ', autopct=make_autopct(Segment_count), explode=[0.05, 0.05, 0.05, 0.05, 0.05, 0.05])

#分組比較#
C_RFM = df_final.copy()
C_RFM['Segment'] = rfm['Segment']
C_RFM.head()

Seg_avg_Age = C_RFM.groupby('Segment')['Age'].mean()
Seg_avg_Regis = C_RFM.groupby('Segment')['Registration_Year'].mean()
Seg_avg_Income = C_RFM.groupby('Segment')['Income'].mean()
Seg_avg_Children = C_RFM.groupby('Segment')['Children'].mean()
Seg_avg_Familysize = C_RFM.groupby('Segment')['Family_Size'].mean()
Seg_avg_Recency = C_RFM.groupby('Segment')['Day_since_last_shopping'].mean()
Seg_avg_Freq = C_RFM.groupby('Segment')['Total_Order'].mean()
Seg_avg_Spent = C_RFM.groupby('Segment')['Total_Spent'].mean()


a = Seg_avg_Age.plot(kind='bar', xlabel='Segment', ylabel='Average Age', title='Comparison of Average Age by Segment',color=['red', 'green', 'blue', 'orange', 'black', 'pink'])
plt.show()
b = Seg_avg_Regis.plot(kind='bar', xlabel='Segment', ylabel='Average Regis', title='Comparison of Average Regis by Segment', color = ['red', 'green', 'blue', 'orange', 'black', 'pink'])
plt.show()
c = Seg_avg_Income.plot(kind='bar', xlabel='Segment', ylabel='Average Income', title='Comparison of Average Income by Segment', color = ['red', 'green', 'blue', 'orange', 'black', 'pink'])
plt.show()
d = Seg_avg_Children.plot(kind='bar', xlabel='Segment', ylabel='Average Children', title='Comparison of Average Children by Segment', color = ['red', 'green', 'blue', 'orange', 'black', 'pink'])
plt.show()
e = Seg_avg_Familysize.plot(kind='bar', xlabel='Segment', ylabel='Average Familysize', title='Comparison of Average Familysize by Segment', color = ['red', 'green', 'blue', 'orange', 'black', 'pink'])
plt.show()
f = Seg_avg_Recency.plot(kind='bar', xlabel='Segment', ylabel='Average Recency', title='Comparison of Average Recency by Segment', color = ['red', 'green', 'blue', 'orange', 'black', 'pink'])
plt.show()
g = Seg_avg_Freq.plot(kind='bar', xlabel='Segment', ylabel='Average Freq', title='Comparison of Average Freq by Segment', color = ['red', 'green', 'blue', 'orange', 'black', 'pink'])
plt.show()
h = Seg_avg_Spent.plot(kind='bar', xlabel='Segment', ylabel='Average Spent', title='Comparison of Average Spent by Segment', color = ['red', 'green', 'blue', 'orange', 'black', 'pink'])
plt.show()

