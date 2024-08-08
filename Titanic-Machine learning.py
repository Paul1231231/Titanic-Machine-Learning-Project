import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

#import data
test = pd.read_csv('test.csv')
train = pd.read_csv('train.csv')

#drop column
PassengerId = test['PassengerId']
y_train = train['Survived']
test = test.drop(columns=['Name'])
test = test.drop(columns=['Ticket'])
test = test.drop(columns=['Cabin'])
test = test.drop(columns=['PassengerId'])
train = train.drop(columns=['Name'])
train = train.drop(columns=['Ticket'])
train = train.drop(columns=['Cabin'])
train = train.drop(columns=['PassengerId'])
train = train.drop(columns=['Survived'])

#one hot encoding
le = LabelEncoder()
train['Sex'] = le.fit_transform(train['Sex'])
test['Sex'] = le.fit_transform(test['Sex'])
train['Embarked'] = le.fit_transform(train['Embarked'])
test['Embarked'] = le.fit_transform(test['Embarked'])
y_train = to_categorical(y_train)
#Normalization
scaler = StandardScaler()
train['Fare'] = scaler.fit_transform(train[['Fare']])
test['Fare'] = scaler.transform(test[['Fare']])

#set up model and fit
model = RandomForestClassifier()
param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 8],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True, False],
    'class_weight': [None, 'balanced'],
    'random_state': [42]
}
clf = GridSearchCV(model, param_grid, cv=3, scoring='accuracy')
clf.fit(train, y_train)

#Accuracy
best_model = clf.best_estimator_
y_predict = best_model.predict(test)
y_predict = np.argmax(y_predict, axis=1)
print(y_predict)

#output csv
output = {'PassengerId':PassengerId, 'Survived': y_predict}
output = pd.DataFrame(output)
file_path = "submission.csv"
output.to_csv(file_path, index=False)