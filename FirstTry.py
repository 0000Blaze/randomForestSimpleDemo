import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import seaborn as sn
import matplotlib.pyplot as plt

#data frame created
candidates = {'gmat': [780,750,690,710,780,730,690,720,740,690,610,690,710,680,770,610,580,650,540,590,620,600,550,550,570,670,660,580,650,760,640,620,660,660,680,650,670,580,590,790],
              'gpa': [4,3.9,3.3,3.7,3.9,3.7,2.3,3.3,3.3,1.7,2.7,3.7,3.7,3.3,3.3,3,2.7,3.7,2.7,2.3,3.3,2,2.3,2.7,3,3.3,3.7,2.3,3.7,3.3,3,2.7,4,3.3,3.3,2.3,2.7,3.3,1.7,3.7],
              'work_experience': [3,4,3,5,4,6,1,4,5,1,3,5,6,4,3,1,4,6,2,3,2,1,4,1,2,6,4,2,6,5,1,2,4,6,5,1,2,1,4,5],
              'age': [25,28,24,27,26,31,24,25,28,23,25,27,30,28,26,23,29,31,26,26,25,24,28,23,25,29,28,26,30,30,23,24,27,29,28,22,23,24,28,31],
              'admitted': [2,2,1,2,2,2,0,2,2,0,0,2,2,1,2,0,0,1,0,0,1,0,0,0,0,1,1,0,1,2,0,0,1,1,1,0,0,0,0,2]
              }

df = pd.DataFrame(candidates,columns= ['gmat', 'gpa','work_experience','age','admitted'])
#print (df)

#setting features(input to model) and label(output of model)
X = df[['gmat', 'gpa','work_experience','age']]
y = df['admitted']

#spliting data into two parts , here train split is 75% and test split is 25% 
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)

#applying random forest classifier function
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train,y_train)    #train
y_pred=clf.predict(X_test)  #test

'''
#applying obtained to be displayed in the confusion matrix
confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
sn.heatmap(confusion_matrix, annot=True)

#display accuracy and plot
print('Accuracy: ',metrics.accuracy_score(y_test, y_pred))
plt.show()
'''

'''
#observing the input and predictions for test data
#accuracy was seen to be right 8/10 time ,so accuracy = 80%
print(X_test)
print(y_pred)
'''

#test the model with input data 
prediction = clf.predict([[730,3.7,4,27]]) 
print ('Predicted Result: ', prediction)
