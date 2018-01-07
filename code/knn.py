import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing, cross_validation
from sklearn.neighbors import KNeighborsClassifier

# Loading the data
train = pd.read_csv("train.csv", parse_dates=["Dates"],
                    index_col=False)
test = pd.read_csv("test.csv", parse_dates=["Dates"],
                   index_col=False)
train.info()
# Sampling
# sampling = 0.65

# dropping the variables Description and Resolution from test as they are not present in test
train = train.drop(["Descript", "Resolution", "Address"], axis=1)
test = test.drop(["Address"], axis=1)
train.info()


# Splitting the date into year, time
def datesplit(data):
    data["Year"] = data["Dates"].dt.year
    data["Month"] = data["Dates"].dt.month
    data["Day"] = data["Dates"].dt.day
    data["Hour"] = data["Dates"].dt.hour
    data["Minute"] = data["Dates"].dt.minute
    return data


train = datesplit(train)
test = datesplit(test)
# Encoding the categorical variables
enc = LabelEncoder()
train["PdDistrict"] = enc.fit_transform(train["PdDistrict"])
wnc = LabelEncoder()
train["DayOfWeek"] = wnc.fit_transform(train["DayOfWeek"])
cat_encoder = LabelEncoder()
cat_encoder.fit(train["Category"])
train["CategoryEncoded"] = cat_encoder.transform(train["Category"])
print(cat_encoder.classes_)
enc = LabelEncoder()
test["PdDistrict"] = enc.fit_transform(test["PdDistrict"])
wnc = LabelEncoder()
test["DayOfWeek"] = wnc.fit_transform(test["DayOfWeek"])
print(train.columns)
print(test.columns)
# Select only the required columns
train_columns = list(train.columns[2:11].values)
print(train_columns)
test_columns = list(test.columns[2:11].values)
print(test_columns)
# Split Coordinates:
# x_train = train[:int(len(train) * sampling)]
# x_test = train[int(len(train) * sampling):]
# scores=[]
# RandomForest Classifier
# classifier = RandomForestClassifier(n_estimators=75,criterion="entropy",bootstrap=True)
# classifier.fit(train[train_columns], train["CategoryEncoded"])
# scores.append(classifier.score(x_test[train_columns], x_test["CategoryEncoded"]))
# test["predictions"] = classifier.predict(test[test_columns])
# Creating the submission file
# def field_to_columns(data, field, new_columns):
#    for i in range(len(new_columns)):
#        data[new_columns[i]] = (data[field] == new_columns[i]).astype(int)
#    return data
# test["Category"]= cat_encoder.inverse_transform(test["predictions"])
# categories = list(cat_encoder.classes_)
# test = field_to_columns(test, "Category", categories)
# print(test.columns)
# PREDICTIONS_FILENAME = 'predictions_'+ '.csv'
# submission_cols = [test.columns[0]]+list(test.columns[13:])
# print(submission_cols)
# test[submission_cols].to_csv(PREDICTIONS_FILENAME, index = False)
# print(scores)
scaler = preprocessing.StandardScaler().fit(train[train_columns])

knn = KNeighborsClassifier(n_neighbors=23, weights='distance', algorithm='auto', metric="minkowski", p=3)

knn.fit(scaler.transform(train[train_columns]),
        train['Category'])
# test['hour'] = test['date'].str[11:13]
# Separate test and train set out of orignal train set.

train['pred'] = knn.predict(scaler.transform(train[train_columns]))
test_pred = knn.predict_proba(scaler.transform(test[test_columns]))

# CHECK TRAINING SET ACCURACY.

# Compute training set accuracy.
print('Training Set Accuracy :', sum(train['Category'] == train['pred']) / len(train['Dates']))

# CROSS VALIDATION.

# Get cross validation scores.
cv_scores = cross_validation.cross_val_score(knn,
                                             scaler.transform(train[train_columns]),
                                             train['Category'],
                                             cv=2)

# Take the mean accuracy across all cross validation segments.
print('Cross Validation Accuracy: ', cv_scores.mean())

# EXPORT TEST SET PREDICTIONS.
# This section exports test predictions to a csv in the format specified by Kaggle.com.
'''
# Turn 'test_pred' into data frame.
test_pred = pd.DataFrame(test_pred)

# Add column names to 'test_pred'.
test_pred.columns = knn.classes_

# Name index column.
test_pred.index.name = 'Id'

# Write csv.
test_pred.to_csv('test_pred_benchmark_knn.csv')
'''