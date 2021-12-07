import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

# data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


def cleanData(data):
    '''Function to clean the datasets'''
    
#     useless columns
    data.drop(columns=['PassengerId', 'Name', 'Cabin', 'Ticket'], inplace=True)
    
#     useless rows
    data.dropna(subset=['Embarked'], inplace=True)
    
#     missing values
    data['Age'] = data.groupby(['Pclass','Sex'])['Age'].transform(lambda x: x.fillna(x.median()))
    
    data['Fare'] = data.groupby(['Pclass','Sex'])['Fare'].transform(lambda x: x.fillna(x.median()))
    
#     transforming categoric into numeric data
    data['Sex'].replace({'male':0, 'female':1}, inplace=True)
    
    data['Embarked'].replace({'S':0, 'C':1, 'Q':2}, inplace=True)
    
    return data

clean_train = cleanData(train)
clean_test = cleanData(test)

X = train.drop(columns='Survived')
y = train['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42)

# the most accurate model / o modelo mais acurado
model = GradientBoostingClassifier()
model.fit(X_train, y_train)

survived = model.predict(clean_test)

submission = pd.DataFrame({"PassengerId": test.PassengerId, "Survived": survived})

def main():
    return submission

main()