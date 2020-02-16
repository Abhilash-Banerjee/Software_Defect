import pandas as pd
import numpy as np
import seaborn 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt

missing_values = ['?' , 'na' , 'n/a']
soft = pd.read_csv('software.csv',  na_values = missing_values)
print(soft.head())

#to check for null value
soft.isnull().sum() 
soft.info()
soft = soft.fillna(0)
soft

print(soft.shape)
print(soft.describe())

print(soft.groupby('b').first())
print(soft.groupby('b').size())


#Visulisation
count = seaborn.countplot(x='defects', data=soft, palette='coolwarm')
count.set(xlabel='defects', ylabel='Count', title='defects')
plt.show()

#Covarience To determine the relationship between the movement of two asset prices
soft.corr()

ax = plt.subplots(figsize = (10, 10))
seaborn.heatmap(soft.corr(), annot = True)
plt.show()

label_encoder = LabelEncoder()
soft['defects'] =label_encoder.fit_transform(soft['defects'].astype('bool')) 


X = soft.drop('defects', axis=1)
y = soft['defects'] 

#Split the data
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=41)

#Classification

rand = RandomForestRegressor() 

# fit the regressor with x and y data 
rand.fit(X_test,y_test)  
y_pred = rand.predict(X_test) 

accuracy = rand.score(X_test,y_test)
print(accuracy*100,'%')



df = pd.DataFrame({'Actual': y_test , 'Predicted': y_pred})
df

df1 = df.head(30)
df1.plot(kind='kde',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()
