##    Instructions: In the data inputs, input the most recent day, the number of days
##    forward that you want predicted (i.e. 30 days into the future, 60 days, etc.),
##    and the ticker that you want predicted. Down at the bottom, comment IN the
##    algorithm that you want and all associated methods (i.e. comment IN methods
##    relating to clf1 and accuracy1 if you want the RandomForest, or clf2 and
##    accuracy2 if you want KNN, etc.). CLF1 - CLF7 are listed in order of accuracy.
##    The final section (CLF1 - CLF3 and ECLF) is an ensemble method that uses a
##    Voter Classification. You can play around with the weights, but it seems like
##    the RandomForest is just more accurate. Comment OUT everything after X_train,
##    X_test etc. if you want to use it.

##    IMPORTANT NOTE: Sometimes the yahoo finance API call doesn't work (you get
##    a query error). If this happens, just hit run again and it should work. If
##    that doesn't work, close the shell and then run again. Just running it
##    again should resolve the error. Currently trying to find a more consistent
##    data source.

##Bollinger Bands, MACD, SAR, Kaufman Adaptive Moving Average, Moving Average,
##Exponential Moving Average, Simple Moving Average, Weighted Moving Average,
##Triangular Moving Average, Triple Exponential Moving Average,
##Average Directional Movement Index,
##Hilbert Transform - Instantaneous Trendline,
##Average Directional Movement Index Rating, Chande Momentum Oscillator,
##Minus Directional Indicator, Plus Directional Indicator, Momentum,
##Relative Strength Index,  Normalized Average True Range, Beta, VIX, CBOE Skew, 


import quandl
import numpy as np
import scipy
import matplotlib.pyplot as plt
import xlsxwriter
import pandas as pd
import yahoo_finance
from yahoo_finance import Share
import datetime
from datetime import datetime
import statistics
import pandas_datareader.data as data
import talib
import xlrd
import sklearn
from sklearn import preprocessing, cross_validation, neighbors
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import AdaBoostClassifier

#DATA INPUTS
day = "2018-01-05"
days = 10
symbol = 'BTC-USD'

def removeDifferentDates(df1, df2):
    dates1 = list(df1.index)
    dates2 = list(df2.index)
    
    i = 0

    while i < len(dates2):
        dates2[i] = dates2[i]
        i = i + 1

    index = 0

    while index < len(dates1):
        dates1[index] = dates1[index]
        index = index + 1

    index = 0
    
    df1.index = dates1
    df2.index = dates2
    
    overallIndexes = list(set(dates1).intersection(set(dates2)))

    checkIndex = 0

    df1 = df1[df1.index.isin(overallIndexes)]
    df2 = df2[df2.index.isin(overallIndexes)]

    return [df1, df2]

#------------------------------------------------------------------------
start = datetime(2006, 12, 14) 
end = day

#Unfortunately the google version of the following only returns 1 year:
stock_data = data.get_data_yahoo(symbol, start, end)

stock = stock_data["Adj Close"]

listUpDown = []
DateList = []

listOfDates = stock.index.tolist()
listOfPrices = stock.tolist()

index = 0

later = days - 1

while later < len(listOfPrices):
    if listOfPrices[later] > listOfPrices[index]:
        listUpDown.append(1)
        DateList.append(listOfDates[index])
    else:
        listUpDown.append(0)
        DateList.append(listOfDates[index])
    index = index + 1
    later = later + 1

UpDownDF = pd.DataFrame({'Date': DateList, 'Up/Down': listUpDown})

UpDownDF = UpDownDF.set_index('Date')
#------------------------------------------------------------------------

skewdf = quandl.get("CBOE/SKEW", authtoken="HMbMEXCjrykoHnyB6PVC", start_date = "2006-12-13", end_date = day)

MyWalletTransactionsDF = quandl.get("BCHAIN/MWNTD")

MyWalletVolumeDF = quandl.get("BCHAIN/MWTRV")

MinerRevDF = quandl.get("BCHAIN/MIREV")

TransactionVolumeDF = quandl.get("BCHAIN/ETRVU")

NumberofTransactionsDF = quandl.get("BCHAIN/NTRAN")

BitcoinDaysDestroyedDF = quandl.get("BCHAIN/BCDDE")

TradeTransactionRatioDF = quandl.get("BCHAIN/TVTVR")

#print(skewdf)

VIXurl = 'http://www.cboe.com/publish/scheduledtask/mktdata/datahouse/vixcurrent.csv'
VIX_df = pd.DataFrame(pd.read_csv(VIXurl))

VIX_values = VIX_df["Unnamed: 4"].tolist()

VIX_values.remove(VIX_values[0])

series = (VIX_df['Cboe data is compiled for the convenience of site visitors and is furnished without responsibility for accuracy and is accepted by the site visitor on the condition that transmission or omissions shall not be made the basis for any claim demand or cause for action.  Your use of Cboe data is subject to the Terms and Conditions of Cboe Websites.'])

VIX_list = series.values.tolist()

label = VIX_list[0]

VIX_list.remove(VIX_list[0])

index = 0

while index < len(VIX_list):
    VIX_list[index] = datetime.strptime(VIX_list[index], "%m/%d/%Y").strftime("%Y-%m-%d")
    VIX_list[index] = datetime.strptime(VIX_list[index], "%Y-%m-%d")
    index = index + 1

VIX_df = pd.DataFrame({label: VIX_list, "VIX": VIX_values})

VIX_df = VIX_df.set_index('Date')

VIX_df = VIX_df.iloc[30:]

#------------------------------------------------------------------------

Low_df = stock_data['Low']
High_df = stock_data['High']
Price_df = stock_data['Adj Close']


#Data File split into separate dataframes

LowList = Low_df.values.tolist()
HighList = High_df.values.tolist()
PriceList = Price_df.values.tolist()
DateList = stock_data.index.tolist()

#converted to list

z = 0

while z < len(LowList):
    if isinstance(LowList[z], str):
        LowList[z] = float(LowList[z].replace(',', ''))
    z = z + 1

y = 0

while y < len(HighList):
    if isinstance(HighList[y], str):
        HighList[y] = float(HighList[y].replace(',', ''))
    y = y + 1

x = 0

while x < len(PriceList):
    if isinstance(PriceList[x], str):
        PriceList[x] = float(PriceList[x].replace(',', ''))
    x = x + 1

#type conversions complete, string --> float

Low = np.array(LowList)
High = np.array(HighList)
Close = np.array(PriceList)

#Low, High, and Close converted to Array format (TA-Lib calls require Array)

SARtoList = (talib.SAR(High, Low, acceleration = 0.2, maximum = 0.20))
BBandsArray = (talib.BBANDS(Close, timeperiod = 5, nbdevup = 2, nbdevdn = 2, matype = 0))
EMAList = talib.EMA(Close, timeperiod=30)
KAMAList = talib.KAMA(Close, timeperiod=30)
MAList = talib.MA(Close, timeperiod=30, matype=0)
WMAList = talib.WMA(Close, timeperiod=30)
TRIMAList = talib.TRIMA(Close, timeperiod=30)
TEMAList = talib.TEMA(Close, timeperiod=30)
HTList = talib.HT_TRENDLINE(Close)
ADXList = talib.ADX(High, Low, Close, timeperiod=14)
ADXRList = talib.ADXR(High, Low, Close, timeperiod=14)
CMOList = talib.CMO(Close, timeperiod=14)
DXList = talib.DX(High, Low, Close, timeperiod=14)
MACDArray = talib.MACDFIX(Close, signalperiod=9)
MINUS_DI_List = talib.MINUS_DI(High, Low, Close, timeperiod=14)
PLUS_DI_List = talib.PLUS_DI(High, Low, Close, timeperiod=14)
MOMList = talib.MOM(Close, timeperiod=10)
RSIList = talib.RSI(Close, timeperiod=14)
NATRList = talib.NATR(High, Low, Close, timeperiod=14)
BETAList = talib.BETA(High, Low, timeperiod=5)

#method calls to TA-Lib complete, results stored in SARtoList (list) and BBandsArray (array)

BBandsUpperDF = pd.DataFrame(BBandsArray[0], columns = ['Upper Band',])
BBandsMiddleDF = pd.DataFrame(BBandsArray[1], columns = ['Middle Band',])
BBandsLowerDF = pd.DataFrame(BBandsArray[2], columns = ['Lower Band',])

MACD_df = pd.DataFrame(MACDArray[0], columns = ['MACD',])
MACD_Hist_df = pd.DataFrame(MACDArray[1], columns = ['MACD_Hist',])
MACD_Sig_df = pd.DataFrame(MACDArray[2], columns = ['MACD_Sig',])

DateDF = pd.DataFrame({'Date': DateList,})
SARdf = pd.DataFrame({'SAR': SARtoList,})
EMAdf = pd.DataFrame({'EMA': EMAList,})
KAMAdf = pd.DataFrame({'KAMA': KAMAList,})
MAdf = pd.DataFrame({'MA': MAList,})
WMAdf = pd.DataFrame({'WMA': WMAList,})
TRIMAdf = pd.DataFrame({'TRIMA': TRIMAList,})
TEMAdf = pd.DataFrame({'TEMA': TEMAList,})
HTdf = pd.DataFrame({'HT Trendline': HTList,})
ADXdf = pd.DataFrame({'ADX': ADXList,})
ADXRdf = pd.DataFrame({'ADXR': ADXRList,})
CMOdf = pd.DataFrame({'CMO': CMOList,})
MINUSDI_df = pd.DataFrame({'MINUSDI': MINUS_DI_List,})
PLUSDI_df = pd.DataFrame({'PLUSDI': PLUS_DI_List,})
MOMdf = pd.DataFrame({'MOM': MOMList,})
RSIdf = pd.DataFrame({'RSI': RSIList,})
NATRdf = pd.DataFrame({'NATR': NATRList,})
BETAdf = pd.DataFrame({'BETA': BETAList,})

#All data converted to DataFrame type

toCombine = [DateDF, SARdf, BBandsUpperDF, BBandsMiddleDF, BBandsLowerDF, SARdf,
             EMAdf, KAMAdf, MAdf, WMAdf, TRIMAdf, TEMAdf, HTdf, ADXdf, ADXRdf,
             CMOdf, MINUSDI_df, PLUSDI_df, MOMdf, RSIdf, NATRdf, BETAdf]

TA_df = pd.concat(toCombine, axis = 1,)

TA_df = TA_df.set_index('Date')

listOfDFs = [TA_df, VIX_df, skewdf, MyWalletTransactionsDF, MyWalletVolumeDF,
             MinerRevDF, TransactionVolumeDF, NumberofTransactionsDF,
             BitcoinDaysDestroyedDF, TradeTransactionRatioDF]

index = 0

while index < len(listOfDFs):
    checkIndex = 0
    while checkIndex < len(listOfDFs):
        if checkIndex != index:
            l = removeDifferentDates(listOfDFs[index], listOfDFs[checkIndex])
            listOfDFs[index] = l[0]
            listOfDFs[checkIndex] = l[1]
        checkIndex = checkIndex + 1
    index = index + 1

inter = pd.concat(listOfDFs, axis = 1)

currentData = inter.iloc[len(inter) - 1]
#currentData = currentData.to_frame()

#print(inter) until 10/20

inter = inter[:-1]

removeDifferentDates(TA_df, UpDownDF)
removeDifferentDates(VIX_df, UpDownDF)
removeDifferentDates(skewdf, UpDownDF)
removeDifferentDates(MyWalletTransactionsDF, UpDownDF)
removeDifferentDates(MyWalletVolumeDF, UpDownDF)
removeDifferentDates(MinerRevDF, UpDownDF)
removeDifferentDates(TransactionVolumeDF, UpDownDF)
removeDifferentDates(NumberofTransactionsDF, UpDownDF)
removeDifferentDates(BitcoinDaysDestroyedDF, UpDownDF)
removeDifferentDates(TradeTransactionRatioDF, UpDownDF)

final = inter

#print(final) until 10/19

count = 0

while count < days:
    final = final[:-1]
    count = count + 1

final = removeDifferentDates(final, UpDownDF)[0]

UpDownDF = removeDifferentDates(final, UpDownDF)[1]

testAndTrain = pd.concat([final, UpDownDF], axis = 1)

#print(final) until 7/26

testAndTrain.fillna(-9999, inplace=True)
currentData.fillna(-9999, inplace=True)

X = np.array(testAndTrain.drop(['Up/Down'], 1))
Y = np.array(testAndTrain['Up/Down'])

X = preprocessing.scale(X)

X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=0.3)

clf = RandomForestClassifier(n_estimators=10)
##mlp = MLPClassifier(hidden_layer_sizes=(26,26,26), activation='tanh')
###clf2 = KNeighborsClassifier(n_neighbors = 2, weights='distance',)
###clf3 = KNeighborsClassifier()
##clf4 = DecisionTreeClassifier()
###clf5 = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5)
###clf6 = SVC()
###clf7 = GaussianNB()
##clf8 = AdaBoostClassifier()


clf.fit(X_train, Y_train)
accuracy = clf.score(X_test, Y_test)

##mlp.fit(X_train, Y_train)
##accuracyNN = mlp.score(X_test, Y_test)

##clf2.fit(X_train, Y_train)
##accuracy2 = clf2.score(X_test, Y_test)

##clf3.fit(X_train, Y_train)
##accuracy3 = clf3.score(X_test, Y_test)

##clf4.fit(X_train, Y_train)
##accuracy4 = clf4.score(X_test, Y_test)

##clf5.fit(X_train, Y_train)
##accuracy5 = clf5.score(X_test, Y_test)

##clf6.fit(X_train, Y_train)
##accuracy6 = clf6.score(X_test, Y_test)

##clf7.fit(X_train, Y_train)
##accuracy7 = clf7.score(X_test, Y_test)

##clf8.fit(X_train, Y_train)
##accuracy8 = clf8.score(X_test, Y_test)

print(accuracy)

##print(accuracyNN)

##print(accuracy2)

##print(accuracy3)

##print(accuracy4)

##print(accuracy5)

##print(accuracy6)

##print(accuracy7)

##print(accuracy8)

currentData = currentData.values.reshape(1, -1)

print(clf.predict(currentData))

print(clf.predict_proba(currentData))

##print(mlp.predict(currentData))
##
##print(mlp.predict_proba(currentData))

##print(clf2.predict(currentData))
##
##print(clf2.predict_proba(currentData))

##print(clf3.predict(currentData))
##
##print(clf3.predict_proba(currentData))

##print(clf4.predict(currentData))
##
##print(clf4.predict_proba(currentData))

##print(clf5.predict(currentData))
##
##print(clf5.predict_proba(currentData))

##print(clf6.predict(currentData))
##
##print(clf6.predict_proba(currentData))

##print(clf7.predict(currentData))
##
##print(clf7.predict_proba(currentData))

##print(clf8.predict(currentData))
##
##print(clf8.predict_proba(currentData))

#----------------------------------------------------------------------------

##clf1 = RandomForestClassifier(n_estimators=10)
##mlp = MLPClassifier(hidden_layer_sizes=(100,100,100), activation='tanh')
##clf2 = DecisionTreeClassifier()
##clf3 = KNeighborsClassifier(n_neighbors = 2, weights='distance',)
##clf1 = clf1.fit(X_train, Y_train)
##clf2 = clf2.fit(X_train, Y_train)
##clf3 = clf3.fit(X_train, Y_train)
##mlp = mlp.fit(X_train, Y_train)
##eclf = VotingClassifier(estimators=[('rf', clf1), ('dt', clf2), ('nn', mlp)], voting='soft', weights=[2,1,1.5])
##eclf = eclf.fit(X_train, Y_train)
##
##accuracy = eclf.score(X_test, Y_test)
##
##print(accuracy)
##
##print(eclf.predict(currentData))
##
##print(eclf.predict_proba(currentData))

#---------------------------------------------------------------------------
