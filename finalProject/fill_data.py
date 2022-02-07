import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def fillMonthlyCharge(data):
    data_with_null = data[[
            'typeContract',
            'hasUnlimitedData',
            'hasPremiumSupport',
            'hasOnlineSecurity',
            'hasOnlineBackup',
            'hasDeviceProtectionPlan',
            'hasMultipleLines',
            'typeInternet',
            'hasPhoneService',
            'offerType',
            'monthlyCharge'
        ]]

    values = {
        'typeContract' : 0,
        'hasUnlimitedData' : 0.5,
        'hasPremiumSupport' : 0.5,
        'hasOnlineSecurity' : 0.5,
        'hasOnlineBackup' : 0.5,
        'hasDeviceProtectionPlan' : 0.5,
        'hasMultipleLines' : 0.5,
        'typeInternet' : data_with_null['typeInternet'].median(),
        'hasPhoneService' : 1.0,
        'offerType' : data_with_null['offerType'].median()
    }

    def checkEmpty(row):
        if (np.isnan(row['monthlyCharge'])):
            if (row['typeInternet'] > 0):
                return row['monthlyCharge']
            else:
                return 20.0
        else:
            return row['monthlyCharge']

    data_with_null = data_with_null.fillna(value = values)
    
    data_with_null['monthlyCharge'] = data.apply(lambda row: checkEmpty(row), axis = 1)

    data_without_null = data[[
            'typeContract',
            'hasUnlimitedData',
            'hasPremiumSupport',
            'hasOnlineSecurity',
            'hasOnlineBackup',
            'hasDeviceProtectionPlan',
            'hasMultipleLines',
            'typeInternet',
            'hasPhoneService',
            'offerType',
            'monthlyCharge'
        ]].dropna()

    train_data_x = data_without_null.iloc[:,:10]
    train_data_y = data_without_null.iloc[:,10]

    linreg = LinearRegression()
    linreg.fit(train_data_x, train_data_y)
    test_data = data_with_null[data_with_null['monthlyCharge'].isnull()].iloc[:,:10]
    predictions = linreg.predict(test_data)

    charge_predicted = pd.DataFrame(data = {'': test_data.iloc[:, 0], 'monthlyCharge': predictions})
    data_with_null.monthlyCharge.fillna(charge_predicted.monthlyCharge, inplace = True)

    data.fillna(data_with_null, inplace = True)

def fillMonthsTenure(data):
    median = data['monthsTenure'].median()

    def checkEmpty(row):
        if (np.isnan(row['monthsTenure'])):
            if (np.isfinite(row['monthlyCharge']) and np.isfinite(row['totalCharges'])):
                return int(row['totalCharges'] / row['monthlyCharge'])
            else:
                return median
        else:
            return row['monthsTenure']

    data['monthsTenure'] = data.apply(lambda row: checkEmpty(row), axis = 1)

def fillTypeInternet(data):
    median = data['typeInternet'].median()

    def checkEmpty(row):
        if (np.isnan(row['hasInternet'])):
            if (np.isnan(row['avgMonthlyDownload'])):
                return 0.0
            elif (row['avgMonthlyDownload'] > 0.0):
                return median
            else:
                return 0.0
        elif (row['hasInternet'] == 0.0):
            return 0.0
        elif (row['hasInternet'] == 1.0 and row['typeInternet'] == 0.0):
            return median
        else:
            return row['typeInternet']

    data['typeInternet'] = data.apply(lambda row: checkEmpty(row), axis = 1)

def fillAvgMonthlyDownload(data):
    median = data['avgMonthlyDownload'].median()

    def checkEmpty(row):
        if (row['typeInternet'] == 0.0):
            return 0.0
        elif (np.isnan(row['avgMonthlyDownload'])):
            return median
        else:
            return row['avgMonthlyDownload']

    data['avgMonthlyDownload'] = data.apply(lambda row: checkEmpty(row), axis = 1)

def fillHasUnlimitedData(data):
    medianDownload = data['avgMonthlyDownload'].median()
    std = data['avgMonthlyDownload'].std()

    def checkEmpty(row):
        if (np.isnan(row['hasUnlimitedData'])):
            if (row['avgMonthlyDownload'] > medianDownload + std):
                return 1.0
            else:
                return 0.0
        else:
            return row['hasUnlimitedData']

    data['hasUnlimitedData'] = data.apply(lambda row: checkEmpty(row), axis = 1)

def fillTotalCharges(data):
    median = data['totalCharges'].median()

    def checkEmpty(row):
        if (np.isnan(row['totalCharges'])):
            if (np.isfinite(row['monthlyCharge']) and np.isfinite(row['monthsTenure'])):
                return row['monthlyCharge'] * row['monthsTenure']
            else:
                return median
        else:
            return row['totalCharges']

    data['totalCharges'] = data.apply(lambda row: checkEmpty(row), axis = 1)

def fillNumReferrals(data):
    median = data['numReferrals'].median()

    def checkEmpty(row):
        if (np.isnan(row['numReferrals'])):
            if (np.isnan(row['hasReferred'])):
                return 0.5
            elif (row['hasReferred'] == 1.0):
                return median
            else:
                return 0.0
        elif (row['hasReferred'] == 0.0):
            return 0.0
        else:
            return row['numReferrals']

    data['numReferrals'] = data.apply(lambda row: checkEmpty(row), axis = 1)

def fillAvgMonthlyLongDistance(data):
    median = data['avgMonthlyLongDistance'].median()

    def checkEmpty(row):
        if (np.isnan(row['avgMonthlyLongDistance'])):
            return median
        else:
            return row['avgMonthlyLongDistance']

    data['avgMonthlyLongDistance'] = data.apply(lambda row: checkEmpty(row), axis = 1)

def fillIsFields(data):
    medianDownload = data['avgMonthlyDownload'].median()
    std = data['avgMonthlyDownload'].std()

    def checkEmpty(row, field):
        if (np.isnan(row[field])):
            if (row['avgMonthlyDownload'] > medianDownload):
                return 1.0
            elif (row['avgMonthlyDownload'] > medianDownload - std):
                return 0.5
            else:
                return 0.0
        else:
            return row[field]

    data['isStreamingTV'] = data.apply(lambda row: checkEmpty(row, 'isStreamingTV'), axis = 1)
    data['isStreamingMovies'] = data.apply(lambda row: checkEmpty(row, 'isStreamingMovies'), axis = 1)
    data['isStreamingMusic'] = data.apply(lambda row: checkEmpty(row, 'isStreamingMusic'), axis = 1)

def fillTotalRefunds(data):
    def checkEmpty(row):
        if (np.isnan(row['totalRefunds'])):
            if (np.isfinite(row['totalExtraData']) and np.isfinite(row['totalLongDistance']) and np.isfinite(row['totalRevenue'])):
                temp = row['totalCharges'] + row['totalExtraData'] + row['totalLongDistance'] - row['totalRevenue']
                if (temp > 0):
                    return temp
                else:
                    return 0.0
            else:
                return 0.0
        else:
            return row['totalRefunds']

    data['totalRefunds'] = data.apply(lambda row: checkEmpty(row), axis = 1)

def fillExtraData(data):
    def checkEmpty(row):
        if (np.isnan(row['totalExtraData'])):
            if (np.isfinite(row['totalLongDistance']) and np.isfinite(row['totalRevenue'])):
                temp = row['totalRevenue'] + row['totalRefunds'] - row['totalCharges'] - row['totalLongDistance']
                if (temp > 0):
                    return temp
                else:
                    return 0.0
            else:
                return 0.0
        else:
            return row['totalExtraData']

    data['totalExtraData'] = data.apply(lambda row: checkEmpty(row), axis = 1)

def fillLongDistance(data):
    medianLongDistance = data['totalLongDistance'].median()

    def checkEmpty(row):
        if (np.isnan(row['totalLongDistance'])):
            if (np.isfinite(row['totalRevenue'])):
                temp = row['totalRevenue'] + row['totalRefunds'] - row['totalCharges'] - row['totalExtraData']
                if (temp > 0):
                    return temp
                else:
                    return 0.0
            else:
                return medianLongDistance
        else:
            return row['totalLongDistance']

    data['totalLongDistance'] = data.apply(lambda row: checkEmpty(row), axis = 1)

def fillTotalRevenue(data):
    def checkEmpty(row):
        if (np.isnan(row['totalRevenue'])):
            return row['totalLongDistance'] - row['totalRefunds'] + row['totalCharges'] + row['totalExtraData']
        else:
            return row['totalRevenue']

    data['totalRevenue'] = data.apply(lambda row: checkEmpty(row), axis = 1)

def fillAge(data):
    medianAge = data['age'].median()

    def checkEmpty(row):
        if (np.isnan(row['age'])):
            if (np.isfinite(row['u30'])):
                if (row['u30'] == 1.0):
                    return 25
            if (np.isfinite(row['senior'])):
                if (row['senior'] == 1.0):
                    return 70
            return medianAge 
        else:
            return row['age']

    data['age'] = data.apply(lambda row: checkEmpty(row), axis = 1)

def fillDependents(data):
    def checkEmpty(row):
        if (np.isnan(row['nDependents'])):
            if (np.isnan(row['dependents'])):
                return 0.0
            elif (row['dependents'] == 0.0):
                return 0.0
            else:
                return 1.0
        else:
            return row['nDependents']

    data['nDependents'] = data.apply(lambda row: checkEmpty(row), axis = 1)

def fillOthers(data):
    values = {
        'hasPaperlessBilling': 1.0,
        'gender': 0.0,
        'married': 0.0,
        'score': 3.0,
        'zip': 90201,
        'paymentMethod': 0.0
    }
    data.fillna(value = values, inplace = True)