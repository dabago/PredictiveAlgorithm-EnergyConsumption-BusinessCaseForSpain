#%%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator
from sklearn.neighbors import LocalOutlierFactor
from sklearn.feature_selection import VarianceThreshold

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn import linear_model
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold

from sklearn.metrics import mean_squared_error

from sklearn.pipeline import make_pipeline

from datetime import datetime
from astral import Astral


#%%
# Functions to get the sunrise and sunset for a given date
def calcLightHours(date):
    date = datetime.strptime(date, '%Y-%m-%d').date()
    sun = city.sun(date=date, local=True)
    return (sun['sunset'].hour + sun['sunset'].minute / 60) - (sun['sunrise'].hour + sun['sunrise'].minute / 60)
#%%
def calcSunrise(date):
    date = datetime.strptime(date, '%Y-%m-%d').date()
    sun = city.sun(date=date, local=True)
    return sun['sunrise'].hour
#%%
def calcSunset(date):
    date = datetime.strptime(date, '%Y-%m-%d').date()
    sun = city.sun(date=date, local=True)
    return sun['sunset'].hour
#%%

# Better visualization in console
pd.set_option('display.max_columns', None)  
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)

# Make the graphs a bit prettier, and bigger
plt.rcParams['figure.figsize'] = (15, 5)


#%%

# =============================================================================
# CREATE cos dataframe
# =============================================================================

# Initialize Astral
a = Astral()
city = a['Madrid']
#%%
# Read the data
co = pd.read_csv("/Users/David/Documents/CloudStation/2018_MCSBT/02_Term/DATA_SCIENCE_ANALYTICS/LAB01/PY/Consumptions.csv", sep=';')
tmp = pd.read_csv('/Users/David/Documents/CloudStation/2018_MCSBT/02_Term/DATA_SCIENCE_ANALYTICS/LAB01/PY/Temperatures.csv', sep=';')
#%%
# I'll work with the consumption summary per day and hour
cos = co[['Date', 'Hour', 'Value']].groupby(['Date', 'Hour']).agg(sum).reset_index()

# add CUPS column
cos2 = co[['Date', 'CUPS']].groupby(['Date'])["CUPS"].nunique().reset_index()
cos = pd.merge(cos, cos2, how='outer', on=['Date'])

# Merge the temperatures file.
cos = pd.merge(cos, tmp, how='inner', on=['Date'])

# We add a column with the day of the week
cos['Weekday'] = pd.DatetimeIndex(cos['Date']).weekday
cos['Day'] = pd.DatetimeIndex(cos['Date']).day
cos['Month'] = pd.DatetimeIndex(cos['Date']).month
cos

# Add column with weekend.
cos['Weekend'] = cos['Weekday'].map(lambda x: 1 if x == 5 or x == 6 else 0)

# Add a column with number of light hours for a given date
cos['LightHours'] = cos['Date'].map(calcLightHours)

# Add a column with sunrise hour
cos['SunriseHour'] = cos['Date'].map(calcSunrise)

# Add a column with sunset hour
cos['SunsetHour'] = cos['Date'].map(calcSunset)

# Add column indicating whether there is daylight or not
cos["DaylightBoolean"] = ((cos["Hour"] > cos["SunriseHour"]) & (cos["Hour"] < cos["SunsetHour"]))
cos['Daylight'] = cos['DaylightBoolean'].map(lambda x: 1 if x == True else 0)

# sort columns
cos = cos[['Date', 'CUPS', 'tMax', 'tMin', 'tMean', 'Hour', 'Weekday', 'Day', 'Month', 'Weekend', 'LightHours', 'SunriseHour', 'SunsetHour', 'Daylight', 'Value']]

#%%
#ohe hour
#
#ohe_hour = pd.get_dummies(cos.Hour)
#ohe_hour.columns = ['1AM', '2AM', '3AM', '4AM', '5AM', '6AM', '7AM', '8AM', '9AM', '10AM', '11AM', '12PM', '13PM', '14PM', '15PM', '16PM', '17PM', '18PM', '19PM', '20PM', '21PM', '22PM', '23PM', '24PM']
#cosx=pd.concat([cos, ohe_hour], axis=1)
#cos=cosx.drop("Hour",axis=1)

#%%
cos.to_csv('/Users/David/Documents/CloudStation/2018_MCSBT/02_Term/DATA_SCIENCE_ANALYTICS/LAB01/CSV/COS.csv', sep=';')


#%%

# =============================================================================
#  CREATE fut dataframe
# =============================================================================

# Read the data
ncust = pd.read_csv('/Users/David/Documents/CloudStation/2018_MCSBT/02_Term/DATA_SCIENCE_ANALYTICS/LAB01/PY/nCustomers.csv', sep=';')
ncust.rename(columns={'datetime': 'Date', "nCUPS": "CUPS"}, inplace=True)
tmp2 = pd.read_csv("/Users/David/Documents/CloudStation/2018_MCSBT/02_Term/DATA_SCIENCE_ANALYTICS/LAB01/PY/Temperatures2.csv", sep=';')

fut = pd.merge(ncust, tmp2, how='outer', on=['Date'])
fut = fut[:243] # drop 1st day of March

# create column Hour
hours = pd.Series(pd.date_range('7/1/2017', freq='H', periods=5856))
hours
df = pd.DataFrame(hours,columns = ['Date'])
df
df['Hour'] = df['Date'].dt.hour
type(df)
df['Date'] = df['Date'].map(lambda x: str(x)[:-9]) # remove hour from "Date" column
df
#%%
# Rearrange hours 
def rearrange_hours():
    n=1
    for i, row in df.iterrows():
        df.at[i,"Hour"] = n
        n += 1
        if n == 25:
            n = 1
#%%
rearrange_hours()
df
#%%
df = df[:-24] # drop 1st day of March

# Add "Hour" column
fut = pd.merge(fut, df, how='inner', on=['Date'])

# We add a column with the day of the week
fut['Weekday'] = pd.DatetimeIndex(fut['Date']).weekday
fut['Day'] = pd.DatetimeIndex(fut['Date']).day 
fut['Month'] = pd.DatetimeIndex(fut['Date']).month

# Add column with weekend. 1 means this day is weekend
fut['Weekend'] = fut['Weekday'].map(lambda x: 1 if x == 5 or x == 6 else 0)

# Add a columns with number of light hours for a given date
fut['LightHours'] = fut['Date'].map(calcLightHours)

# Add a column with sunrise hour
fut['SunriseHour'] = fut['Date'].map(calcSunrise)

# Add a column with sunset hour
fut['SunsetHour'] = fut['Date'].map(calcSunset)
fut
# Add column indicating whether there is daylight or not
fut["DaylightBoolean"] = ((fut["Hour"] > fut["SunriseHour"]) & (fut["Hour"] < fut["SunsetHour"]))
fut['Daylight'] = fut['DaylightBoolean'].map(lambda x: 1 if x == True else 0)

fut = fut[['Date', 'CUPS', 'tMax', 'tMin', 'tMean', 'Hour', 'Weekday', 'Day', 'Month', 'Weekend', 'LightHours', 'SunriseHour', 'SunsetHour', 'Daylight']]
fut
#%%
#ohe hour
#
#ohe_hour_fut = pd.get_dummies(fut.Hour)
#ohe_hour_fut
#ohe_hour_fut.columns = ['1AM', '2AM', '3AM', '4AM', '5AM', '6AM', '7AM', '8AM', '9AM', '10AM', '11AM', '12PM', '13PM', '14PM', '15PM', '16PM', '17PM', '18PM', '19PM', '20PM', '21PM', '22PM', '23PM', '24PM']
#futx=pd.concat([fut, ohe_hour_fut], axis=1)
#fut=futx.drop("Hour",axis=1)
#fut

#%%
fut.to_csv('/Users/David/Documents/CloudStation/2018_MCSBT/02_Term/DATA_SCIENCE_ANALYTICS/LAB01/CSV/FUT.csv', sep=';')


#%%

cos.head()
#%%
fut.head()

#%%

def getX(): # get X to fit in model
 
    si = ('si',  SimpleImputer(missing_values=np.NaN, strategy='median'))
    ohe = ('ohe', OneHotEncoder(sparse=False, handle_unknown='ignore'))    
    pipec = Pipeline([si, ohe])
    
    Xc = cos[["Hour","Weekday"]] # no day
    Xct = pd.DataFrame(pipec.fit_transform(Xc))
    Xct
    
    imp = ('imp', SimpleImputer(missing_values= np.NaN, strategy='median'))
    #    scl = ('scl', StandardScaler())
     
    pipen = Pipeline([imp])
    
    Xn = cos[["CUPS","tMean","Weekend", "Daylight"]] # no tMax no tMin no LightHours
    Xn
    Xnt = pd.DataFrame(pipen.fit_transform(Xn))
    Xnt
    
    X = pd.concat([Xct, Xnt], axis=1, sort=False)
    
    return X






#%%

def get_new_X(): # get new X from dataset to predict
    
    si = ('si',  SimpleImputer(missing_values=np.NaN, strategy='median'))
    ohe = ('ohe', OneHotEncoder(sparse=False, handle_unknown='ignore'))    
    pipec = Pipeline([si, ohe])
    
    Xc1 = fut[["Hour","Weekday"]] # no month no day
    Xct1 = pd.DataFrame(pipec.fit_transform(Xc1))
 
    imp = ('imp', SimpleImputer(missing_values=np.NaN, strategy='median'))
#    scl = ('scl', StandardScaler())

    pipen = Pipeline([imp])
    
    Xn1 = fut[["CUPS","tMean","Weekend","Daylight"]] # no tMax no tMin no LightHours
    Xnt1 = pd.DataFrame(pipen.fit_transform(Xn1))
    
    Xnew = pd.concat([Xct1, Xnt1], axis=1, sort=False)
    
    return Xnew
    
#%%
    
# =============================================================================
# PREDICTOR
# =============================================================================
    
X = getX()
#X.shape
y = cos['Value']
#type(y)

##y = np.array(y)
#m = np.shape(X)
#y = y.reshape(m)


model = linear_model.LinearRegression() 
#model = svm.SVR(kernel='poly', degree=2, C=100, gamma='auto')

#model = linear_model.Ridge(alpha=1) 
#model = RandomForestRegressor(n_estimators=100, n_jobs=-1)

model.fit(X, y)
# new instances where we do not know the answer
Xnew = get_new_X()
Xnew.shape
# make a prediction
ynew = model.predict(Xnew)
ynew.shape

# show the inputs and predicted outputs

pred_column = []

for i in range(len(Xnew)):
#    print(ynew[i])
    pred_column.append(ynew[i])

predictions = fut
predictions['Value'] = pred_column

tmp2.head()
type(ynew)

#fut.to_csv('/Users/David/Documents/CloudStation/2018_MCSBT/02_Term/DATA_SCIENCE_ANALYTICS/LAB01/CSV/Predictions.csv', sep=';')


#%%
# =========
# DELIVERY
# =========
#
predictions.keys()

submit = predictions[['Date','Hour','Value']]

# Set the Index to be the Date
submit['Date'] = pd.to_datetime(submit['Date'], format='%Y/%m/%d')
submit.set_index('Date', inplace=True)


# Slice the Data
From = '2017-08-10'
To   = '2017-08-20'
submit1 = submit.loc[From:To,:]

From = '2017-09-10'
To   = '2017-09-20'
submit2 = submit.loc[From:To,:]

From = '2017-11-10'
To   = '2017-11-20'
submit3 = submit.loc[From:To,:]

From = '2018-02-10'
To   = '2018-02-20'
submit4 = submit.loc[From:To,:]

submit_final = pd.concat([submit1,submit2,submit3,submit4],axis=0,sort=False)

#submit_final.to_csv('/Users/David/Documents/CloudStation/2018_MCSBT/02_Term/DATA_SCIENCE_ANALYTICS/LAB01/CSV/Submit_Final_David_Barrero_Gonzalez.csv', sep=';')


#%%

# =============================================================================
# Cross-validation   
# =============================================================================

X = getX()
X.shape
y = cos['Value']
y.shape

X = np.array(X)
rs=0

kf = KFold(n_splits=5, shuffle=True, random_state=rs)

scores = []

for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
#    r = linear_model.Ridge(alpha=1) 
    r = svm.SVR(kernel='linear', degree=2, C=100, gamma='auto')
#    r = linear_model.LinearRegression() 
#    r = RandomForestRegressor(n_estimators=100, n_jobs=-1)

    r.fit(X_train, y_train)
    y_predicted = r.predict(X_test)
    y_test = y_test.values.flatten()
    score = np.average(np.abs(y_test - y_predicted) / y_test)
    print(score)
    scores.append(score)
    
print(scores)   
print(np.average(scores))

#%%

# Some plots 

print(cos.describe())
print(cos.info())
    
# Distribution of output
sns.distplot(cos['Value'])

# Correlation of numeric variables
corr = cos.select_dtypes(include=[np.number]).corr()
print(corr['Value'].sort_values(ascending=False))

# Plotting pairs of variables
sns.jointplot(x=cos['CUPS'], y=cos['Value'])
sns.jointplot(x=cos['tMean'], y=cos['Value'])
sns.jointplot(x=cos['LightHours'], y=cos['Value'])
sns.jointplot(x=cos['Month'], y=cos['Value'])
sns.jointplot(x=cos['Weekend'], y=cos['Value'])
sns.jointplot(x=cos['Weekday'], y=cos['Value'])
sns.jointplot(x=cos['Day'], y=cos['Value'])


#%%

      # Calculate error with formulas from Jesus example

# The learning part. X is the input and y the output 
# The output has to be removed from the input matrix
X = getX()

# And the output is just the Value column
y = cos['Value'] 

# Simple training and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=25)
       
# r is the model (a regressor)
#r = linear_model.LinearRegression()
r = svm.SVR(kernel='poly', degree=2, C=100, gamma='auto')
#r = RandomForestRegressor(n_estimators=100, n_jobs=-1)

# We train the model using the train data set
r.fit(X_train, y_train)

# Then, we make a prediction with the test dataset input matrix
y_predicted = r.predict(X_test)
# At this point, y_predicted contains the predicted consumption for the test set
y_predicted

y_test = y_test.values.flatten()
# finally we calculate the error of the prediction comparing with y_test
# remember that numpy can deal with element wise operations with vectors    
print(np.average(np.abs(y_test - y_predicted) / y_test))

# If you want to cross validate, (and you should) you just put lines from 78 to 96 in a loop
# and it will be equivalent to a Shuffle Split cross validation. Maybe this is
# easier than using standard scikit-learn cross validation methods