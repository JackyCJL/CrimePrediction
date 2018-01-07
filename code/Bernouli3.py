# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 19:06:11 2016
@author: Aditya
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 15:42:01 2016
@author: Aditya
"""
#Importing required libraries and packages
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, label_binarize, StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import log_loss
#Loading the data
train = pd.read_csv("train.csv", parse_dates = ["Dates"], index_col= False)
test = pd.read_csv("test.csv",parse_dates=["Dates"], index_col = False)
train.info()
#Sampling
#sampling = 0.65

#dropping the variables Description and Resolution from test as they are not present in test
train = train.drop(["Descript", "Resolution"], axis=1)
#test= test.drop(["Address"], axis = 1)
train.info()

#Splitting the date into year, time
def datesplit(data):
    data["Year"] = data["Dates"].dt.year
    data["Month"] = data["Dates"].dt.month
    data["Day"] = data["Dates"].dt.day
    data["Hour"] = data["Dates"].dt.hour
    data["Minute"] = data["Dates"].dt.minute
    return data
train = datesplit(train)
test = datesplit(test)
#Encoding the categorical variables
cat_encoder = LabelEncoder()
add_encoder = LabelEncoder()
cat_encoder.fit(train["Category"])
train["CategoryEncoded"]= cat_encoder.transform(train["Category"])
train = pd.concat([train,pd.get_dummies(train.PdDistrict)], axis=1)
train = pd.concat([train,pd.get_dummies(train.DayOfWeek)], axis=1)
train['StreetNo'] = train['Address'].apply(lambda x: x.split(' ', 1)[0] if x.split(' ', 1)[0].isdigit() else 0)
train["Intersection"]= train["Address"].apply(lambda x: 1 if "/" in x else 0)
train['Address'] = train['Address'].apply(lambda x: x.split(' ', 1)[1] if x.split(' ', 1)[0].isdigit() else x)
add_encoder.fit(train["Address"])
train["Address"]= add_encoder.transform(train["Address"])
#train["Address"]= train["Address"].apply(lambda x: 1 if "/" in x else 0)

train["Morning"] = train["Hour"].apply(lambda x: 1 if x>= 6 and x < 12 else 0)
train["Noon"] = train["Hour"].apply(lambda x: 1 if x>= 12 and x < 17 else 0)
train["Evening"] = train["Hour"].apply(lambda x: 1 if x>= 17 and x < 20 else 0)
train["Night"] = train["Hour"].apply(lambda x: 1 if x >= 20 and x < 6 else 0)

for i in range(0,24):
    train[str(i)] = train["Hour"].apply(lambda x: 1 if x == i else 0)
'''

train["Fall"] = train["Month"].apply(lambda x: 1 if x>=3 and x <=5 else 0)
train["Winter"] = train["Month"].apply(lambda x: 1 if x>=6 and x <=8 else 0)
train["Spring"] = train["Month"].apply(lambda x: 1 if x>=9 and x <=11 else 0)
train["Summer"] = train["Month"].apply(lambda x: 1 if x>=12 and x <=2 else 0)
'''
#add_encoder = LabelEncoder()
test = pd.concat([test,pd.get_dummies(test.PdDistrict)], axis=1)
test = pd.concat([test,pd.get_dummies(test.DayOfWeek)], axis=1)
test['StreetNo'] = test['Address'].apply(lambda x: x.split(' ', 1)[0] if x.split(' ', 1)[0].isdigit() else 0)
test["Intersection"] = test["Address"].apply(lambda x: 1 if "/" in x else 0)
test['Address'] = test['Address'].apply(lambda x: x.split(' ', 1)[1] if x.split(' ', 1)[0].isdigit() else x)
add_encoder.fit(test["Address"])
test["Address"]= add_encoder.transform(test["Address"])
#test["Address"] = test["Address"].apply(lambda x: 1 if "/" in x else 0)
#test["Dark"] = test["Hour"].apply(lambda x: 1 if x>=18 or x<=5 else 0)

test["Morning"] = test["Hour"].apply(lambda x: 1 if x>= 6 and x < 12 else 0)
test["Noon"] = test["Hour"].apply(lambda x: 1 if x>= 12 and x < 17 else 0)
test["Evening"] = test["Hour"].apply(lambda x: 1 if x>= 17 and x < 20 else 0)
test["Night"] = test["Hour"].apply(lambda x: 1 if x >= 20 and x < 6 else 0)


test["0"]  =test["Hour"].apply(lambda x: 1 if x == 0 else 0)
test["1"]  =test["Hour"].apply(lambda x: 1 if x == 1 else 0)
test["2"]  =test["Hour"].apply(lambda x: 1 if x == 2 else 0)
test["3"]  =test["Hour"].apply(lambda x: 1 if x == 3 else 0)
test["4"]  =test["Hour"].apply(lambda x: 1 if x == 4 else 0)
test["5"]  =test["Hour"].apply(lambda x: 1 if x == 5 else 0)
test["6"]  =test["Hour"].apply(lambda x: 1 if x == 6 else 0)
test["7"]  =test["Hour"].apply(lambda x: 1 if x == 7 else 0)
test["8"]  =test["Hour"].apply(lambda x: 1 if x == 8 else 0)
test["9"]  =test["Hour"].apply(lambda x: 1 if x == 9 else 0)
test["10"]  =test["Hour"].apply(lambda x: 1 if x == 10 else 0)
test["11"]  =test["Hour"].apply(lambda x: 1 if x == 11 else 0)
test["12"]  =test["Hour"].apply(lambda x: 1 if x == 12 else 0)
test["13"]  =test["Hour"].apply(lambda x: 1 if x == 13 else 0)
test["14"]  =test["Hour"].apply(lambda x: 1 if x == 14 else 0)
test["15"]  =test["Hour"].apply(lambda x: 1 if x == 15 else 0)
test["16"]  =test["Hour"].apply(lambda x: 1 if x == 16 else 0)
test["17"]  =test["Hour"].apply(lambda x: 1 if x == 17 else 0)
test["18"]  =test["Hour"].apply(lambda x: 1 if x == 18 else 0)
test["19"]  =test["Hour"].apply(lambda x: 1 if x == 19 else 0)
test["20"]  =test["Hour"].apply(lambda x: 1 if x == 20 else 0)
test["21"]  =test["Hour"].apply(lambda x: 1 if x == 21 else 0)
test["22"]  =test["Hour"].apply(lambda x: 1 if x == 22 else 0)
test["23"]  =test["Hour"].apply(lambda x: 1 if x == 23 else 0)
'''
test["Fall"] = test["Month"].apply(lambda x: 1 if x>=3 and x <=5 else 0)
test["Winter"] = test["Month"].apply(lambda x: 1 if x>=6 and x <=8 else 0)
test["Spring"] = test["Month"].apply(lambda x: 1 if x>=9 and x <=11 else 0)
test["Summer"] = test["Month"].apply(lambda x: 1 if x>=12 and x <=2 else 0)
'''
#add_encoder.fit(test["Address"])
#test["Address"]=add_encoder.transform(test["Address"])
#PC = PCA(n_components=2, copy = False)
#train["X"] = PC.fit_transform(train["X"])
#train["Y"] = PC.fit_transform(train["Y"])
#test["X"] = PC.fit_transform(test["X"])
#test["Y"] = PC.fit_transform(test["Y"])
print(cat_encoder.classes_)
print(train.columns)
print(test.columns)
train = train.drop(["CategoryEncoded","StreetNo","Address"], axis=1)
test = test.drop(["StreetNo","Address"], axis = 1)
#Select only the required columns
print(list(train.columns[:].values))
train_columns = list(train.columns[9:].values)
#train_columns = list(train.columns[7:].values)
print(train_columns)
test_columns = list(test.columns[9:].values)
#test_columns = list(test.columns[7:].values)
print(test_columns)
#Split Coordinates:
#x_train = train[:int(len(train) * sampling)]
#x_test = train[int(len(train) * sampling):]
#scores=[]
#RandomForest Classifier
#classifier = RandomForestClassifier(n_estimators=75,criterion="entropy",bootstrap=True)
#classifier.fit(train[train_columns], train["CategoryEncoded"])
#scores.append(classifier.score(x_test[train_columns], x_test["CategoryEncoded"]))
#test["predictions"] = classifier.predict(test[test_columns])
#Creating the submission file
#def field_to_columns(data, field, new_columns):
#    for i in range(len(new_columns)):
#        data[new_columns[i]] = (data[field] == new_columns[i]).astype(int)
#    return data
#test["Category"]= cat_encoder.inverse_transform(test["predictions"])
#categories = list(cat_encoder.classes_)
#test = field_to_columns(test, "Category", categories)
#print(test.columns)
#PREDICTIONS_FILENAME = 'predictions_'+ '.csv'
#submission_cols = [test.columns[0]]+list(test.columns[13:])
#print(submission_cols)
#test[submission_cols].to_csv(PREDICTIONS_FILENAME, index = False)
#print(scores)
#scaler = preprocessing.StandardScaler().fit(train[train_columns])

#knn=KNeighborsClassifier(n_neighbors=23, weights='distance',algorithm='auto',metric="minkowski", p=3)

#knn.fit(scaler.transform(train[train_columns]),
#       train['Category'])
#test['hour'] = test['date'].str[11:13]
# Separate test and train set out of orignal train set.

#train['pred'] = knn.predict(scaler.transform(train[train_columns]))
#test_pred = knn.predict_proba(scaler.transform(test[test_columns]))
#Bernouli model
training,validation = train_test_split(train, train_size=.75)
model = BernoulliNB(alpha=1.0, fit_prior = True)
#print(train)
print(train[train_columns])
model.fit(train[train_columns], train["Category"])
predicted = model.predict_proba(test[test_columns])
predict = model.predict_proba(validation[train_columns])
lgloss = log_loss(validation["Category"], predict)
print(lgloss)
# EXPORT TEST SET PREDICTIONS.
# This section exports test predictions to a csv in the format specified by Kaggle.com.
#result = pd.DataFrame(predicted,columns=cat_encoder.classes_)
#result.to_csv("final.csv", index = True, index_label = "Id")