import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.naive_bayes import BernoulliNB
import time
#用pandas载入csv训练数据，并解析第一列为日期格式
train=pd.read_csv(r'train.csv', parse_dates = ['Dates'])
test=pd.read_csv(r'test.csv', parse_dates = ['Dates'])
#print(train.info(),train.describe())
#用LabelEncoder对不同的犯罪类型编号
leCrime = preprocessing.LabelEncoder()
crime = leCrime.fit_transform(train.Category)
#因子化星期几，街区，小时等特征
days = pd.get_dummies(train.DayOfWeek)
district = pd.get_dummies(train.PdDistrict)
hour = train.Dates.dt.hour
hour = pd.get_dummies(hour)
#组合特征
trainData = pd.concat([hour, days, district], axis=1)
trainData['crime']=crime
#对于测试数据做同样的处理
days = pd.get_dummies(test.DayOfWeek)
district = pd.get_dummies(test.PdDistrict)
hour = test.Dates.dt.hour
hour = pd.get_dummies(hour)
testData = pd.concat([hour, days, district], axis=1)
features = ['Friday', 'Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday',
'Wednesday', 'BAYVIEW', 'CENTRAL', 'INGLESIDE', 'MISSION',
'NORTHERN', 'PARK', 'RICHMOND', 'SOUTHERN', 'TARAVAL', 'TENDERLOIN']
hourFea = [x for x in range(0,24)]
features = features + hourFea
print(features)
# 分割训练集(3/5)和测试集(2/5)
training, validation = train_test_split(trainData, test_size=0.75)
'''
X_train,X_test, y_train, y_test =
cross_validation.train_test_split(train_data,train_target,test_size=0.4, random_state=0)
train_data：所要划分的样本特征集
train_target：所要划分的样本结果
test_size：样本占比，如果是整数的话就是样本的数量
random_state：是随机数的种子。
随机数种子：其实就是该组随机数的编号，在需要重复试验的时候，保证得到一组一样的随机数。比如你每次都填1，其他参数一样的情况下你得到的随机数组是一样的。但填0或不填，每次都会不一样。
随机数的产生取决于种子，随机数和种子之间的关系遵从以下两个规则：
种子不同，产生不同的随机数；种子相同，即使实例不同也产生相同的随机数。
'''
# 朴素贝叶斯建模，计算log_loss
model = BernoulliNB()
'''ernoulliNB假设特征的先验概率为二元伯努利分布，即如下式：
P(Xj=xjl|Y=Ck)=P(j|Y=Ck)xjl+(1−P(j|Y=Ck)(1−xjl)
此时l只有两种取值。xjl只能取值0或者1。
BernoulliNB一共有4个参数，其中3个参数的名字和意义和MultinomialNB完全相同。
唯一增加的一个参数是binarize。这个参数主要是用来帮BernoulliNB处理二项分布的，
可以是数值或者不输入。如果不输入，则BernoulliNB认为每个数据特征都已经是二元的。
否则的话，小于binarize的ba会归为一类，大于binarize的会归为另外一类。
在使用BernoulliNB的fit或者partial_fit方法拟合数据后，我们可以进行预测。
此时预测有三种方法，包括predict，predict_log_proba和predict_proba。　
predict方法就是我们最常用的预测方法，直接给出测试集的预测类别输出。
predict_proba则不同，它会给出测试集样本在各个类别上预测的概率。
容易理解，predict_proba预测出的各个类别概率里的最大值对应的类别，也就是predict方法得到类别。
predict_log_proba和predict_proba类似，它会给出测试集样本在各个类别上预测的概率的一个对数转化。
转化后predict_log_proba预测出的各个类别对数概率里的最大值对应的类别，也就是predict方法得到类别。。'''
nbStart = time.time()
model.fit(training[features], training['crime'])
'''partial_fit说明：增量的训练一批样本 
这种方法被称为连续几次在不同的数据集，从而实现核心和在线学习，这是特别有用的，当数据集很大的时候，不适合在内存中运算 
该方法具有一定的性能和数值稳定性的开销，因此最好是作用在尽可能大的数据块（只要符合内存的预算开销） '''
print(training[features])
nbCostTime = time.time() - nbStart#消耗的时间
predicted = np.array(model.predict_proba(validation[features]))
print("朴素贝叶斯建模耗时 %f 秒" %(nbCostTime))
print("朴素贝叶斯log损失为 %f" %(log_loss(validation['crime'], predicted)))

#print(validation['crime'])
#print(predicted)
#model.fit(trainData[features],trainData['crime'])
predicted = np.array(model.predict_proba(testData[features]))
#print(predicted)

'''
result = pd.DataFrame(predicted,columns=['ARSON','ASSAULT','BAD CHECKS','BRIBERY','BURGLARY','DISORDERLY CONDUCT','DRIVING UNDER THE INFLUENCE','DRUG/NARCOTIC','DRUNKENNESS','EMBEZZLEMENT',
                                         'EXTORTION','FAMILY OFFENSES','FORGERY/COUNTERFEITING','FRAUD','GAMBLING','KIDNAPPING','LARCENY/THEFT','LIQUOR LAWS','LOITERING','MISSING PERSON',
                                         'NON-CRIMINAL','OTHER OFFENSES','PORNOGRAPHY/OBSCENE MAT','PROSTITUTION','RECOVERED VEHICLE','ROBBERY','RUNAWAY','SECONDARY CODES','SEX OFFENSES FORCIBLE',
                                         'SEX OFFENSES NON FORCIBLE','STOLEN PROPERTY','SUICIDE','SUSPICIOUS OCC','TREA','TRESPASS','VANDALISM','VEHICLE THEFT','WARRANTS','WEAPON LAWS'])
result.to_csv("testresult1.csv", index = True, index_label = "Id")
'''