import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Loading Data
df = pd.read_csv("D:\Data science\kaggle_dataset\marketing_campaign\marketing_campaign.csv", sep="\t")
print('Number of records:',df.shape[0])
df.head()
df.info()

#EDA : Data Cleaning & Data Preprocessing
