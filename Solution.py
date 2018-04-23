import pandas as pd
import matplotlib.pyplot as plt
train_features = pd.read_csv("E:\\download\\Criminal\\criminal_train.csv")
from sklearn.model_selection import train_test_split
IQ_X=train_features.iloc[:,:-1]
IQ_Y=train_features.iloc[:,-1]
from sklearn.ensemble import AdaBoostClassifier
#abc = AdaBoostClassifier()

abc = AdaBoostClassifier(n_estimators=100,algorithm='SAMME.R')
abc.fit(IQ_X,IQ_Y)
test_features = pd.read_csv("E:\\download\\Criminal\\criminal_test.csv")

y_pred=abc.predict(test_features)
IQ_PerID1=test_features['PERID']
data_frame = pd.DataFrame(IQ_PerID1)
data_frame['Criminal'] = pd.Series(y_pred,index=data_frame.index)

#data_frame = pd.DataFrame(y_pred,columns=['PerID'])

#data_frame['Criminal'] = pd.Series(IQ_PerID1, index=data_frame.index)

#print data_frame
data_frame.to_csv("E:\\download\\Criminal\\submission.csv")