import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split



PATH = 'C:/Users/Harsh/Desktop/Input/' #set path where data set files are stored



train_data = pd.read_csv(PATH + 'train.csv')

test_data = pd.read_csv(PATH + 'test.csv')



combined_data=pd.concat([test_data, train_data]) # to add both train and test data

combined_data.to_csv( "combined_data.csv", index=False )



combined_data.head() #print top 5

print("Numerical data analysis is: ")
print(combined_data.describe()) #count, mean, std, min, 25%, 50%, 75%, max


print("The missing value data is: ")
print(combined_data.isnull().sum()) #find number of missing values



#categorical data analysis



responses=pd.Categorical(combined_data.Pclass, ordered=True)

df_responses = pd.DataFrame({"response": responses})
print("Pclass data analysis: ")
print(df_responses.astype(str).describe())



responses=pd.Categorical(combined_data.Sex, ordered=True)

df_responses = pd.DataFrame({"response": responses})
print("Sex data analysis: ")
print(df_responses.describe())



responses=pd.Categorical(combined_data.SibSp, ordered=True)

df_responses = pd.DataFrame({"response": responses})
print("SibSp data analysis: ")
print(df_responses.describe())



responses=pd.Categorical(combined_data.Embarked, ordered=True)

df_responses = pd.DataFrame({"response": responses})
print("Embarked data analysis: ")
print(df_responses.describe())



responses=pd.Categorical(combined_data.Ticket, ordered=True)

df_responses = pd.DataFrame({"response": responses})
print("Ticket data analysis: ")
print(df_responses.describe())



responses=pd.Categorical(combined_data.Cabin, ordered=True)

df_responses = pd.DataFrame({"response": responses})
print("Cabin data analysis: ")
print(df_responses.describe())



responses=pd.Categorical(combined_data.Parch, ordered=True)

df_responses = pd.DataFrame({"response": responses})
print("Parch data analysis: ")
print(df_responses.describe())



responses=pd.Categorical(combined_data.PassengerId, ordered=True)

df_responses = pd.DataFrame({"response": responses})
print("Passenger ID data analysis: ")
print(df_responses.astype(str).describe())
