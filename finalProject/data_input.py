import pandas as pd
import fill_data as fd

class DataInput:
    def __init__(self):
        self.statusData = None
        self.IDS = None
        self.Demographics = None
        self.Services = None
        self.Satisfaction = None
        self.Location = None
        self.finalData = None
        self.finalLabels = None

    def readStatus(self, filename):
        def convertChurn(churn):
            if (churn == 'No Churn'):
                return 0
            elif (churn == 'Competitor'):
                return 1
            elif (churn == 'Dissatisfaction'):
                return 2
            elif (churn == 'Attitude'):
                return 3
            elif (churn == 'Price'):
                return 4
            elif (churn == 'Other'):
                return 5
            else:
                return -1

        headerNames = ['id', 'churn']
        self.statusData = pd.read_csv(
                filename,
                header = 0,
                names = headerNames,
                converters = {
                    'churn': convertChurn
                }
            )
    
    def readIDS(self, filename):
        headerNames = ['id']
        self.IDS = pd.read_csv(
                filename,
                header = 0,
                names = headerNames
            )

    def readDemographics(self, filename):
        headerNames = [
                'id',
                'count',
                'gender',
                'age',
                'u30',
                'senior',
                'married',
                'dependents',
                'nDependents'
            ]
        demographics = pd.read_csv(
                filename,
                header = 0,
                names = headerNames
            )

        genderDictionary = {'Female': 1.0, 'Male': -1.0}
        trueFalseDictionary = {'Yes': 1.0, 'No': 0.0}

        # Changing gender to -1, 1
        demographics['gender'] = demographics['gender'].map(genderDictionary)

        # Changing some categories to True, False'''
        demographics['u30'] = demographics['u30'].map(trueFalseDictionary)
        demographics['senior'] = demographics['senior'].map(trueFalseDictionary)
        demographics['married'] = demographics['married'].map(trueFalseDictionary)
        demographics['dependents'] = demographics['dependents'].map(trueFalseDictionary)

        self.Demographics = demographics.drop(
                'count',
                axis = 1
            )

    def readSatisfaction(self, filename):
        headerNames = ['id', 'score']
        self.Satisfaction = pd.read_csv(
                filename,
                header = 0,
                names = headerNames
            )

    def readServices(self, filename):
        def convertOffer(offer):
            if (offer == 'None'): 
                return 0.0
            elif (offer == 'Offer A'): 
                return 1.0
            elif (offer == 'Offer B'): 
                return 2.0
            elif (offer == 'Offer C'): 
                return 3.0
            elif (offer == 'Offer D'): 
                return 4.0
            elif (offer == 'Offer E'): 
                return 5.0
            else:
                return 0.0

        def converterInternet(internet):
            if (internet == 'Fiber Optic'):
                return 3.0
            elif (internet == 'DSL'):
                return 2.0
            elif (internet == 'Cable'):
                return 1.0
            else:
                return 0.0

        def convertTypeContract(contract):
            if (contract == 'Month-to-Month'):
                return 1.0
            elif (contract == 'One Year'):
                return 2.0
            elif (contract == 'Two Year'):
                return 3.0
            else:
                return 0.0

        def convertPaymentMethod(payment):
            if (payment == 'Mailed Check'):
                return 1.0
            elif (payment == 'Bank Withdrawal'):
                return 2.0
            elif (payment == 'Credit Card'):
                return 3.0
            else:
                return 0.0


        headerNames = [
                'id',
                'count',
                'quarter',
                'hasReferred',
                'numReferrals',
                'monthsTenure',
                'offerType',
                'hasPhoneService',
                'avgMonthlyLongDistance',
                'hasMultipleLines',
                'hasInternet',
                'typeInternet',
                'avgMonthlyDownload',
                'hasOnlineSecurity',
                'hasOnlineBackup',
                'hasDeviceProtectionPlan',
                'hasPremiumSupport',
                'isStreamingTV',
                'isStreamingMovies',
                'isStreamingMusic',
                'hasUnlimitedData',
                'typeContract',
                'hasPaperlessBilling',
                'paymentMethod',
                'monthlyCharge',
                'totalCharges',
                'totalRefunds',
                'totalExtraData',
                'totalLongDistance',
                'totalRevenue'
            ]
        services = pd.read_csv(
                filename,
                header = 0,
                names = headerNames,
                converters = {
                    'offerType': convertOffer,
                    'typeInternet': converterInternet,
                    'typeContract': convertTypeContract,
                    'paymentMethod': convertPaymentMethod
                }
            )

        yesNoDictionary = {'Yes': 1.0, 'No': 0.0}
        trueFalseDictionary = {'Yes': 1.0, 'No': 0.0}

        # Changing the services to 0, 1 values
        services['hasOnlineSecurity'] = services['hasOnlineSecurity'].map(yesNoDictionary)
        services['hasPhoneService'] = services['hasPhoneService'].map(yesNoDictionary)
        services['hasOnlineBackup'] = services['hasOnlineBackup'].map(yesNoDictionary)
        services['hasDeviceProtectionPlan'] = services['hasDeviceProtectionPlan'].map(yesNoDictionary)
        services['hasPremiumSupport'] = services['hasPremiumSupport'].map(yesNoDictionary)
        services['isStreamingTV'] = services['isStreamingTV'].map(yesNoDictionary)
        services['isStreamingMusic'] = services['isStreamingMusic'].map(yesNoDictionary)
        services['isStreamingMovies'] = services['isStreamingMovies'].map(yesNoDictionary)
        services['hasUnlimitedData'] = services['hasUnlimitedData'].map(yesNoDictionary)
        services['hasReferred'] = services['hasReferred'].map(trueFalseDictionary)
        services['hasMultipleLines'] = services['hasMultipleLines'].map(trueFalseDictionary)
        services['hasInternet'] = services['hasInternet'].map(trueFalseDictionary)
        services['hasPaperlessBilling'] = services['hasPaperlessBilling'].map(trueFalseDictionary)

        self.Services = services.drop(
                ['count', 'quarter'],
                axis = 1
            )

    def readLocation(self, filename):
        headerNames = [
            'id',
            'count',
            'country',
            'state',
            'city',
            'zip',
            'latlong',
            'latitude',
            'longitude'
        ]

        location = pd.read_csv(
                filename,
                header = 0,
                names = headerNames
            )
        self.Location = location.drop(
                [
                    'count',
                    'country',
                    'state',
                    'city',
                    'latlong',
                    'latitude',
                    'longitude'
                ],
                axis = 1
            )



    def processData(self, isTraining):
        def mergeDataFrames(left, right):
            mergedDataFrame = pd.merge(
                    left,
                    right,
                    on = ['id'],
                    how = 'right'
                )
            return mergedDataFrame
        
        if (isTraining):
            firstData = mergeDataFrames(
                    self.IDS,
                    self.statusData
                )
        else:
            firstData = self.IDS

        mergedSLDemographics = mergeDataFrames(
                self.Demographics,
                firstData
            )

        mergedSLDSatisfaction = mergeDataFrames(
                self.Satisfaction,
                mergedSLDemographics
            )

        mergedSLDSServices = mergeDataFrames(
                self.Services,
                mergedSLDSatisfaction
            )

        mergedSLDSSLocation = mergeDataFrames(
                self.Location,
                mergedSLDSServices
            )

        if (isTraining):
            self.finalLabels = mergedSLDSSLocation[['churn']]
            self.finalData = mergedSLDSSLocation.drop(
                    ['churn', 'id'],
                    axis = 1
                )
        else:
            self.finalData = mergedSLDSSLocation.drop(['id'], axis = 1)
            mergedSLDemographics.to_csv('./merge1.csv')

        fd.fillTypeInternet(self.finalData)
        fd.fillAvgMonthlyDownload(self.finalData)
        fd.fillHasUnlimitedData(self.finalData)
        fd.fillMonthlyCharge(self.finalData)
        fd.fillMonthsTenure(self.finalData)
        fd.fillTotalCharges(self.finalData)
        fd.fillNumReferrals(self.finalData)
        fd.fillAvgMonthlyLongDistance(self.finalData)
        fd.fillIsFields(self.finalData)
        fd.fillTotalRefunds(self.finalData)
        fd.fillExtraData(self.finalData)
        fd.fillLongDistance(self.finalData)
        fd.fillTotalRevenue(self.finalData)
        fd.fillAge(self.finalData)
        fd.fillDependents(self.finalData)
        fd.fillOthers(self.finalData)

        self.finalData = self.finalData.drop(
                [
                    'hasOnlineSecurity',
                    'hasOnlineBackup',
                    'hasDeviceProtectionPlan',
                    'hasPremiumSupport',
                    'isStreamingMovies',
                    'isStreamingMusic',
                    'hasUnlimitedData',
                    'hasReferred',
                    'hasPhoneService',
                    'hasInternet',
                    'u30',
                    'senior',
                    'dependents'
                ],
                axis = 1
            )
        self.finalData = self.finalData.reset_index()
        