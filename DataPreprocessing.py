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

#9
plt = train_data.Pclass.value_counts().sort_index().plot('bar', title='')
plt.set_xlabel('Pclass')
plt.set_ylabel('Survival Probability')

#10

plt = train_data[['Pclass', 'Survived']].groupby('Pclass').mean().Survived.plot('bar')
plt.set_xlabel('Pclass')
plt.set_ylabel('Survival Probability')

#11

histogram_1= sns.FacetGrid(train_data, col ='Survived', size=6)
histogram_1.map(plt.hist, 'Age', bins=range(0, 130, 5))
plt.xticks(range(0, 130, 5))

#12

histogram_2 = sns.FacetGrid(train_data, col='Survived', row='Pclass', size=3, aspect=1.5)
histogram_2.map(plt.hist, 'Age', bins=20)
histogram_2.add_legend();

#13

histogram_3 = sns.FacetGrid(train_data, row='Embarked', col='Survived', height=3, aspect=1.5)
histogram_3.map(sns.barplot, 'Sex', 'Fare')
histogram_3.add_legend()

#14
train_data['Ticket'].describe()
duplicacy_rate = ((891-681)/891)*100
print(duplicacy_rate)

#15

combined_data.describe(include=["O"])

#16

sex = {'male' :0, 'female':1}
combined_data['Gender'] = combined_data['Sex'].map(sex)
combined_data.head()

#17
combined_data['Age'].fillna(np.random.uniform(low=combined_data['Age'].std(), high=combined_data['Age'].mean()),inplace=True)
combined_data['Age'].isna().sum()

#18
combined_data['Embarked'].isna().sum()
combined_data['Embarked'] = combined_data['Embarked'].fillna(train_data.Embarked.describe().top)
combined_data['Embarked'].isna().sum()

#19
combined_data['Fare'].fillna(combined_data['Fare'].mode()[0],inplace =True)
combined_data['Fare'].isna().sum()

#20
combined_data['FareBand'] = pd.qcut(combined_data['Fare'], 4)
val = combined_data.FareBand.unique().get_values()
val.sort()
for i in range(len(val)):
    combined_data.loc[(combined_data['Fare'] > val[i].left) & (combined_data['Fare'] <= val[i].right), 'Fare'] = i
combined_data['Fare'] = combined_data['Fare'].astype(int)

combined_data.head(10)


#RUN EACH LINE OF CODE ONE BY ONE

