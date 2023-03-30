import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from LinearRegression import LinearRegression
from sklearn.preprocessing import OneHotEncoder
import numpy as np

#import the data set ,evaluate it and preprocess the data
df=pd.read_csv("GYM.csv")
cols_to_drop = ['playerId', 'Name']
# Drop columns from dataframe
df = df.drop(cols_to_drop, axis=1)
df[['Age', 'BodyweightKg', 'BestSquatKg','BestDeadliftKg']] = df[['Age', 'BodyweightKg','BestSquatKg','BestDeadliftKg']].apply(lambda x: x.astype(float))


print(df.head())


#find the correlation among the features
corr= df.corr()
print(corr)
#convert the data frame into a numpy matrix
encoder = OneHotEncoder()
one_hot = pd.get_dummies(df['Equipment'])
two_hot = pd.get_dummies(df['Sex'])
cols_to_drop_2= ['Sex', 'Equipment']
df = df.drop(cols_to_drop_2, axis=1)
df = pd.concat([df, one_hot,two_hot], axis=1)
print(df)



#split the data 
x=df[['Age','BestDeadliftKg','BodyweightKg','F','M','Multi-ply','Raw','Single-ply','Wraps']] #Indexing all numerical values in the dataframe
y=df['BestSquatKg'] #target value, in this case the Best squat
#split the data in training and testing sets
x=x.to_numpy()
y=y.to_numpy()
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)


reg= LinearRegression(lr=0.0000001, n_iters=100000)
print('training')
reg.fit(x_train, y_train)
y_pred= reg.predict(x_test)
mse = reg.mse(y_test,y_pred)
r2= reg.r_squared(y_test, y_pred)
print(mse)
print(r2)
while True:
    print("type in just numbers  ")
    age = float(input("How old are you? "))
    bmi = float(input("What is your Body weight?(Kg) "))
    Max_dl = float(input("Whats is your max deadlift?(Kg)  ")) 
    sex = input("Are you a homie or a chick? (F, M)")
    Eq = input("What equipment do you use? (Multi-ply, Raw, Single-ply, Wraps)")
    if Eq=="Multi-ply":
        Eq=[1,0,0,0]
    elif Eq=="Raw":
        Eq=[0,1,0,0]
    elif Eq=="Single-ply":
        Eq=[0,0,1,0]
    elif Eq=="Wraps":
        Eq=[0,0,0,1]
    else:
        print("please type in the correct equipment")
        continue
    
    if sex=="F":
       sex=[1,0]
    elif sex=="M":
        sex=[0,1]
    else:
        print("we are not that progressive yet")
        continue

    X_user=[age,bmi,Max_dl,sex[0],sex[1],Eq[0],Eq[1],Eq[2],Eq[3]]
    X_user=np.array(X_user)
    X_user=X_user.reshape(1,-1)
    
    y_pred_user= reg.predict(X_user)

    print("your max squat is: ",y_pred_user)
    
    user =input("Wanna make another prediction hommie?")

    if user=="n":
        break