#import self-made libraries
import data_input as di
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

if __name__ == "__main__":
    trainingData = di.DataInput()
    trainingData.readStatus('./data/status.csv')
    trainingData.readIDS('./data/train_IDs.csv')
    trainingData.readDemographics('./data/demographics.csv')
    trainingData.readSatisfaction('./data/satisfaction.csv')
    trainingData.readServices('./data/services.csv')
    trainingData.readLocation('./data/location.csv')
    trainingData.processData(True)
    trainingData.finalData.to_csv('./training.csv')
    
    X = trainingData.finalData.to_numpy()
    y = np.ravel(trainingData.finalLabels.to_numpy())

    testingData = di.DataInput()
    testingData.readIDS('./data/test_IDs.csv')
    testingData.readDemographics('./data/demographics.csv')
    testingData.readSatisfaction('./data/satisfaction.csv')
    testingData.readServices('./data/services.csv')
    testingData.readLocation('./data/location.csv')
    testingData.processData(False)
    testingData.finalData.to_csv('./testing.csv')

    Z = testingData.finalData.to_numpy()

    #myModel = DecisionTreeClassifier()
    #myModel = GaussianNB()
    #myModel = SVC()
    
    myModel = LinearDiscriminantAnalysis()

    myModel.fit(X, y)

    Ein = myModel.predict(X)

    print(np.sum(Ein != y) / y.size)

    y_pred = myModel.predict(Z)

    churnDF = pd.DataFrame(y_pred, columns = ['Churn Category'])

    tempResult = pd.concat([testingData.IDS, churnDF], axis = 1)

    resultDF = tempResult[['id', 'Churn Category']].copy()

    resultDF = resultDF.rename(columns = {'id': 'Customer ID'})
    resultDF.to_csv('./submission.csv', index = False)

    

