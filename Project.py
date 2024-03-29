#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
import statsmodels.api as sm 
from statsmodels.formula.api import glm
from datetime import datetime
from sklearn.linear_model import LinearRegression
from matplotlib.ticker import AutoMinorLocator

#plt.style.use('classic')
#https://datahub.io/cryptocurrency/bitcoin#data-cli
#https://datahub.io/cryptocurrency/bitcoin#python
#%% [markdown]
import os
dirpath = os.getcwd() 
print("current directory is : " + dirpath)
import xlrd
#read the data
fp = os.path.join( dirpath, 'bitcoin_csv.csv')
btc = pd.read_csv(fp)

#%%
#Data cleaning

b = btc.dropna()
#add price change (percent) columns
l = [True]
percl = [0]
p_arr = b.iloc[:,5].values
for i in range(1,len(b)):
    l.append(p_arr[i] > p_arr[i-1])
    percl.append((p_arr[i]-p_arr[i-1])/p_arr[i-1])
b['pc'] = l
b['pcp'] = percl

b= b[b['exchangeVolume(USD)'] != 0]
#Add a year column
b['year'] = pd.DatetimeIndex(b['date']).year
#Add a month column
b['month'] = pd.DatetimeIndex(b['date']).month
b.columns
b.head()

#%%
#_________________________________Market cap VS daily transaction vol___________________________
abnormal = b[b['marketcap(USD)']<b['txVolume(USD)']]

fig, ax = plt.subplots(figsize=(12,10))
plt.grid()
#plot the daily transaction volumn and total marketcap
plt.plot(b['txVolume(USD)'], label='Daily transaction vol')
plt.plot(b['marketcap(USD)'], label='Total marketCap')
#Change the xticks label
locs,labels = plt.xticks()
l = []
for i in locs:
    if i == 3750:
        l.append('na')
    elif i == 4000:
        l.append('na')
    else:
        l.append(btc['date'][i])
ax.set_xticklabels(l)

for i in abnormal.index:
    plt.plot(i,abnormal['txVolume(USD)'][i],'o',ms = 7)
plt.xlabel('date')
plt.ylabel('hundred billion(USD)  1e^11')
plt.legend()
filepath = os.path.join( dirpath,'MarketCap_DailyVol.png')
plt.savefig(filepath)
plt.show()


#%%
#_________________________________MultiGraph_________________________________
fig, axs = plt.subplots(2, 3) 
plt.figure(figsize=(12,10))
axs[0, 0].plot(b['price(USD)'])
axs[0, 0].set_title('price')

axs[0, 1].plot(b['txVolume(USD)'])
axs[0, 1].set_title('txvolume')

axs[1, 0].plot(b['averageDifficulty'])
axs[1, 0].set_title('averageDifficulty')

axs[0, 2].plot(b['activeAddresses'])
axs[0, 2].set_title('activeAddresses')

axs[1, 1].plot(b['fees'])
axs[1, 1].set_title('fees')

axs[1, 2].plot(b['blockSize'])
axs[1, 2].set_title('blockSize')

fig.subplots_adjust(left=0.08, right=0.98, bottom=0.05, top=0.9, hspace=0.8, wspace=0.5)

filepath = os.path.join( dirpath,'Multi_comp.png')
plt.savefig(filepath)
plt.show()
#%% _________________________________Correlation_________________________________
import seaborn as sns
plt.figure(figsize=(12,10))
cor = b.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Blues)
filepath = os.path.join( dirpath ,'corr_b.png')
plt.savefig(filepath)
plt.show()

#%%________________________________GeneratedCoins_________________________________???
start = b.index[0]
end = b.index[-1]
lo = 2738
ro = 2739
b.loc[ro]
#2016-7-10
Cycle1 = b[pd.DatetimeIndex(b['date']) < '2016-07-10']
Cycle2 = b[pd.DatetimeIndex(b['date']) >= '2016-07-10']

x1 = Cycle1.index.values
x2 = Cycle2.index.values
regC1 = LinearRegression().fit(pd.DataFrame(Cycle1.index),Cycle1['generatedCoins'])
regC2 = LinearRegression().fit(pd.DataFrame(Cycle2.index),Cycle2['generatedCoins'])
#y_pred = reg.predict(pd.DataFrame(Cycle1.index))
#reg.score(Cycle1[['generatedCoins']].values,y_pred)
#plt.scatter(b.index,b['generatedCoins'],c = 'black',s = 10)
fig, ax = plt.subplots(figsize=(12,10))
#plt.scatter()
plt.scatter(b.index,b['generatedCoins'].values,c ='deepskyblue')
locs,labels = plt.xticks()
plt.plot(x1,(regC1.coef_*x1 + regC1.intercept_),c = 'r',linewidth=4)
plt.plot(x2,(regC2.coef_*x2 + regC2.intercept_),c = 'r',linewidth=4)
l = []
for i in locs:
    if i == 3750:
        l.append('na')
    elif i == 4000:
        l.append('na')
    else:
        l.append(btc['date'][i])

filepath = os.path.join( dirpath ,'Generated_Coins.png')
ax.set_xticklabels(l)
plt.xlabel('date')
plt.ylabel('generated coins')
ax.set_facecolor('k')
ax.xaxis.label.set_color('w')
ax.yaxis.label.set_color('w')
fig.patch.set_facecolor('steelblue')
plt.grid()

plt.savefig(filepath)
plt.show()



#%%________________________________Category in YEAR _________________________________???
Y14 = b[b['year'] == 2014]
Y15 = b[b['year'] == 2015]
Y16 = b[b['year'] == 2016]
Y17 = b[b['year'] == 2017]
Y18 = b[b['year'] == 2018]

plt.figure(figsize=(12,10))
plt.grid()
ax = plt.subplot(111)

for year in np.delete(b.year.unique(),0):
    ax.boxplot(b[b['year'] == year]['activeAddresses'],positions = [year])
    

ax.set_xlim(2013,2019)
plt.xlabel('year')
plt.ylabel('price')
filepath = os.path.join( dirpath ,'box_year.png')
plt.savefig(filepath)
plt.show()

fig, ax = plt.subplots(figsize=(12,10))
plt.plot(Y16['price(USD)'])
locs,labels = plt.xticks()
l = []
for i in locs:
    if i == 3750:
        l.append('na')
    elif i == 4000:
        l.append('na')
    else:
        l.append(btc['date'][i])
ax.set_xticklabels(l)
plt.xlabel('date')
plt.ylabel('price')
filepath = os.path.join( dirpath ,'2016.png')
#plot the brexit dot
UK_index = Y16[Y16['date'] == '2016-06-23'].index[0]
plt.plot(UK_index,Y16['price(USD)'][UK_index],'^',ms = 15, label = 'Breit')
#plot the BTC coin dot
Loss = Y16[Y16['date'] == '2016-08-01'].index[0]
plt.plot(Loss,Y16['price(USD)'][Loss],'v',ms = 15, label = 'The Bitfinex Bitcoin Hack')
#plot the US president election dot
USP = Y16[Y16['date'] == '2016-11-08'].index[0]
plt.plot(USP,Y16['price(USD)'][USP],'o',ms = 15, label = 'U.S. presidential election')

plt.legend()
plt.savefig(filepath)
plt.show()


#%%________________________________Category in MONTH_________________________________???

plt.figure(figsize=(12,10))
plt.grid()
ax = plt.subplot(111)
#make boxplots by months
for month in b.month.unique():
    ax.boxplot(b[b['month'] == month]['price(USD)'],positions = [month]) 
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.set_xlim(0,13)
ax.text(1,-1,1)
plt.xlabel('month')
plt.ylabel('price')
filepath = os.path.join( dirpath ,'box_month.png')
plt.savefig(filepath)
plt.show()

#%%________________________________Category in SEASON_________________________________???
#split the dataset by season
Spring = b[b['month'].isin([3,4,5])]
Summer = b[b['month'].isin([6,7,8])]
Fall = b[b['month'].isin([9,10,11])]
Winter = b[b['month'].isin([12,1,2])]

plt.figure(figsize=(12,10))
ax = plt.subplot(111)
ax.boxplot(Spring.activeAddresses,positions = [1])
ax.boxplot(Summer.activeAddresses,positions = [2])
ax.boxplot(Fall.activeAddresses,positions = [3])
ax.boxplot(Winter.activeAddresses,positions = [4])
ax.set_xlim(0,5)
plt.show()

plt.figure(figsize=(12,10))
ax = plt.subplot(111)
ax.boxplot(Spring['price(USD)'],positions = [1])
ax.boxplot(Summer['price(USD)'],positions = [2])
ax.boxplot(Fall['price(USD)'],positions = [3])
ax.boxplot(Winter['price(USD)'],positions = [4])
ax.set_xlim(0,5)
plt.xlabel('season')
plt.ylabel('price')
filepath = os.path.join( dirpath ,'box_season.png')
plt.savefig(filepath)
plt.show()


#%%
#_________________________________Blocksize & activeaddresses or txCount_________________________________
from statsmodels.formula.api import ols
modelBS_AA = ols(formula='blockSize ~ activeAddresses', data=b).fit()
print( modelBS_AA.summary())
modelBS_tC = ols(formula='blockSize ~ txCount', data=b).fit()
print( modelBS_tC.summary())

plt.style.use('classic')
plt.scatter(b['activeAddresses'],b['blockSize'],c = 'black',s = 10)
x = np.linspace(100000,1300000,200000)
k = modelBS_AA.params[0]
t = modelBS_AA.params[1]
plt.xlabel('activeAddresses')
plt.ylabel('bloackSize')
#plt.plot(x,1.15*(k*x + t)/100000,c = 'r',linewidth=4)
predictions = modelBS_AA.predict(b['activeAddresses'])
plt.plot(b['activeAddresses'], predictions, '-',c  = 'r',linewidth = 4)
filepath = os.path.join( dirpath ,'add_blockSize.png')
plt.savefig(filepath)
plt.show()

plt.style.use('classic')
plt.scatter(b['txCount'],b['blockSize'],c = 'black',s = 10)
x = np.linspace(50000,500000,100000)
kC = modelBS_tC.params[0]
tC = modelBS_tC.params[1]
plt.xlabel('txCount')
plt.ylabel('bloackSize')
#plt.plot(x,2*(kC*x + tC)/50000,c = 'r',linewidth=4)
predictions = modelBS_tC.predict(b['txCount'])
plt.plot(b['txCount'], predictions, '-',c  = 'r',linewidth = 4)
filepath = os.path.join( dirpath ,'txCount_blockSize.png')
plt.savefig(filepath)
plt.show()



#%%
#____________________________21 milllion coins____________________________
#create a method to caculate number of coins generated in total by cycle and reward
def generator(cycle,reward):
    c = 210000
    if cycle is not 1:
        return c*reward + generator(cycle-1,reward/2)
    else:
        return c*reward
generator(40,50)

#set the x label
x = np.arange(1,33,1).tolist()
l = []
for i in x:
    l.append(generator(i,50))
    
plt.figure(figsize=(12,10))
plt.axhline(y=21000000, color='r', linestyle='-',label = '21 million line')
plt.axvline(x=33, color='green', linestyle=':',label = 'cycle 33')
plt.plot(x,l,label = '#Coins generated')
plt.plot(33,generator(33,50),'v',ms=15,label = 'point')
plt.xlabel('Cycle')
plt.ylabel('#Coins')
plt.legend(loc='center')
filepath = os.path.join( dirpath ,'21m.png')
plt.savefig(filepath)
plt.show()

# %%
#Get X and y variable. Y is one day late
XX = b[b.columns.difference(['date', 'marketcap(USD)','price(USD)','txVolume(USD)','adjustedTxVolume(USD)','exchangeVolume(USD)','medianTxValue(USD)'])].iloc[1:,:]
X = XX.values
y = b.iloc[:-1, 5:6].values
yp = b.iloc[:-1, -2:-1].values

# %%
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
Xp_train, Xp_test, yp_train, yp_test = train_test_split(X, yp, test_size = 0.2, random_state = 0)

# %%

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)
y_test = sc_y.fit_transform(y_test)
#%%
# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression

reg = LinearRegression().fit(X_train, y_train)
y_pred = reg.predict(X_test)
plt.plot(y_pred,label = 'pred')
plt.plot(y_test,label = 'actual')
plt.legend()
plt.show()
y_pred = reg.predict(X_test)
R2 = 1 - (np.sum((y_test-y_pred)**2)/np.sum((y_test-np.mean(y_test))**2))

# %%
from sklearn.ensemble import RandomForestRegressor
RFreg = RandomForestRegressor(n_estimators = 100, random_state = 0)
RFreg.fit(X_train, y_train)
plt.plot(RFreg.predict(X_test),label = 'pred')
plt.plot(y_test,label = 'actual')
plt.legend()
plt.show()

#y_predRF = RFreg.predict(X_test)
#R2_RF = 1 - (np.sum((y_test-y_predRF)**2)/np.sum((y_test-np.mean(y_test))**2))


# %%
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(Xp_train, yp_train)
# Predicting the Test set results
yp_pred = classifier.predict(Xp_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(yp_test, yp_pred)

cm

# %%
# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(Xp_train, yp_train)

# Predicting the Test set results
yp_pred = classifier.predict(Xp_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(yp_test, yp_pred)
cm

# %% Wenyu's Part

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#plt.style.use('classic')
#https://datahub.io/cryptocurrency/bitcoin#data-cli
#%% [markdown]
import os
dirpath = os.getcwd() 
print("current directory is : " + dirpath)
filepath = os.path.join( dirpath, 'bitcoin_csv.csv')
print(filepath)

import xlrd
 
bitcoin = pd.read_csv(filepath)
fp = os.path.join( dirpath, 'bitcoin_csv.csv')
btc = pd.read_csv(fp)

#%%
#Data cleaning
btc = bitcoin.dropna()
btc = btc.dropna()
#add price change (percent) columns
l = [True]
percl = [0]
p_arr = btc.iloc[:,5].values
for i in range(1,len(btc)):
    l.append(p_arr[i] > p_arr[i-1])
    percl.append((p_arr[i]-p_arr[i-1])/p_arr[i-1])
btc['pc'] = l
btc['pcp'] = percl

btc= btc[btc['exchangeVolume(USD)'] != 0]

btc.columns

# %%
#Get X and y variable. Y is one day late
btc.columns=btc.columns.str.strip().str.lower().str.replace('(','').str.replace('U','').str.replace('S','').str.replace('D','').str.replace(')','')

XX = btc[btc.columns.difference(['date', 'marketcapusd','priceusd','txVolumeusd','adjustedTxVolumeusd','exchangeVolumeusd','medianTxValueusd'])].iloc[1:,:]
X = XX.values
y = btc.iloc[:-1, 5:6].values
yp = btc.iloc[:-1, -2:-1].values

# %%
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)
Xp_train, Xp_test, yp_train, yp_test = train_test_split(X, yp, test_size = 0.2, random_state = 123)

# %%

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
#y_train = sc_y.fit_transform(y_train)
#y_test = sc_y.fit_transform(y_test)

btc.head()
# xbtc = btc[['txvolumeusd', 'adjustedtxvolumeusd', 'txcount', 'activeaddresses']]
# # print(xbtc.head())
# ybtc = btc['priceusd']
# xbtcs = pd.DataFrame( scale(xbtc), columns=xbtc.columns )
# priceusd = pd.DataFrame(btc['priceusd'])
# priceusd

#%%
from sklearn import linear_model

full_split = linear_model.LinearRegression() # new instancew
full_split.fit(X_train, y_train)
y_pred = full_split.predict(X_test)
full_split.score(X_test, y_test)

#print(y_pred[0:5])

print('score:', full_split.score(X_test, y_test)) # 0.9425689619048208
print('intercept:', full_split.intercept_) # [2535.11953212]
print('coef_:', full_split.coef_)  # [ 1399.60277109  1081.09477569   840.17730788   172.53603703
                                   #  -196.81423833  1122.51188693  -456.76237218  -333.06092781
                                   #  150.98440138   318.11769235    37.67772706     2.21087506
                                   #  -72.91491501 -1075.72250299   244.68387439]

#%%
plt.scatter(y_test, y_pred)
plt.xlabel('True Values')
plt.ylabel('Predictions')
# %%
# VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Get variables for which to compute VIF and add intercept term
Xvif = btc[['txvolumeusd', 'adjustedtxvolumeusd', 'txcount', 'generatedcoins', 'fees', 'activeaddresses', 'averagedifficulty','mediantxvalueusd', 'blocksize']]
Xvif['Intercept'] = 1

# 'adjustedtxvolumeusd', 'txcount', 'generatedcoins', 'fees', 'activeaddresses', 'averagedifficulty','mediantxvalueusd', 'blocksize'

# Compute and view VIF
vif = pd.DataFrame()
vif["variables"] = Xvif.columns
vif["VIF"] = [ variance_inflation_factor(Xvif.values, i) for i in range(Xvif.shape[1]) ] # list comprehension

# View results using print
print(vif)

# Therefore, txvolume, adjustedtxvolumn, txcount and activeaddresses and blocksize are highly correlated with price

#%%
## Adjusted Linear Model
xbtc = btc[['txvolumeusd', 'adjustedtxvolumeusd', 'txcount', 'activeaddresses', 'blocksize']]
ybtc = btc['priceusd']

Xad_train, Xad_test, yad_train, yad_test = train_test_split(xbtc, ybtc, test_size = 0.2, random_state = 1)

scad_X = StandardScaler()
Xad_train = scad_X.fit_transform(Xad_train)
Xad_test = scad_X.transform(Xad_test)


ad_split = linear_model.LinearRegression() # new instancew
ad_split.fit(Xad_train, yad_train)
yad_pred = ad_split.predict(Xad_test)
ad_split.score(Xad_test, yad_test)

print('score:', ad_split.score(Xad_test, yad_test)) # 0.8906185214311232
print('intercept:', ad_split.intercept_) # 204.34242221557088
print('coef_:', ad_split.coef_)  # [-2.00364500e-07  1.67363753e-06 -2.28343962e-02  1.05130901e-02 -4.24595526e-07]
print(yad_pred[0:5])
plt.scatter(yad_test, yad_pred)
plt.xlabel('True Values')
plt.ylabel('Predictions')

#%% 
# Super Confused!!!!!!!!!!!!!!!!!!!!!Cross-Validation
from sklearn import linear_model
from sklearn.model_selection import cross_val_score, cross_val_predict

full_cv = linear_model.LinearRegression()
cv_results = cross_val_score(full_cv, Xad_train, yad_train, cv=10)
print(cv_results) 
np.mean(cv_results) 
print("Accuracy: %0.2f (+/- %0.2f)" % (cv_results.mean(), cv_results.std() * 2))

ycv_pred = cross_val_predict(full_cv, Xad_train, yad_train, cv=10)
plt.scatter(yad_train, ycv_pred)
plt.xlabel('True Values')
plt.ylabel('Predictions')

from sklearn import metrics
accuracy = metrics.r2_score(yad_train, ycv_pred)
print( 'Cross-Predicted Accuracy:', accuracy)




#%%
# Regression Tree
# import seaborn as sns
# sns.set()
# sns.pairplot(xbtc)

#%%
from sklearn.tree import DecisionTreeRegressor  # Import DecisionTreeRegressor
from sklearn.model_selection import train_test_split  # Import train_test_split
from sklearn.metrics import mean_squared_error as MSE  # Import mean_squared_error as MSE
# Split data into 80% train and 20% test
#X_train, X_test, y_train, y_test= train_test_split(xbtcs, ybtcs, test_size=0.2,random_state=1)
# Instantiate a DecisionTreeRegressor 'regtree0'
regtree0 = DecisionTreeRegressor(max_depth=5, min_samples_leaf=1,random_state=1) # set minimum leaf to contain at least 10% of data points
# DecisionTreeRegressor(criterion='mse', max_depth=8, max_features=None,
#     max_leaf_nodes=None, min_impurity_decrease=0.0,
#     min_impurity_split=None, min_samples_leaf=0.13,
#     min_samples_split=2, min_weight_fraction_leaf=0.0,
#     presort=False, random_state=3, splitter='best')


regtree0.fit(Xad_train, yad_train)  # Fit regtree0 to the training set
# Import mean_squared_error from sklearn.metrics as MSE
from sklearn.metrics import mean_squared_error as MSE

# evaluation
yad_pred = regtree0.predict(Xad_test)  # Compute y_pred
mse_regtree0 = MSE(yad_test, yad_pred)  # Compute mse_regtree0
rmse_regtree0 = mse_regtree0 ** (.5) # Compute rmse_regtree0
print("Test set RMSE of regtree0: {:.2f}".format(rmse_regtree0)) 

#%%
# Let us compare the performance with OLS
from sklearn import linear_model
olsbtc = linear_model.LinearRegression() 
olsbtc.fit( Xad_train, yad_train )

y_pred_ols = olsbtc.predict(Xad_test)  # Predict test set labels/values

mse_ols = MSE(yad_test, y_pred_ols)  # Compute mse_ols
rmse_ols = mse_ols**(0.5)  # Compute rmse_ols

print('Linear Regression test set RMSE: {:.2f}'.format(rmse_ols))
print('Regression Tree test set RMSE: {:.2f}'.format(rmse_regtree0))

# %%

# Compare the tree with CV
regtree1 = DecisionTreeRegressor(max_depth=5, min_samples_leaf=1, random_state=1)

# Evaluate the list of MSE ontained by 10-fold CV

from sklearn.model_selection import cross_val_score
# Set n_jobs to -1 in order to exploit all CPU cores in computation
MSE_CV = - cross_val_score(regtree1, Xad_train, yad_train, cv= 10, scoring='neg_mean_squared_error', n_jobs = -1)
regtree1.fit(Xad_train, yad_train)  # Fit 'regtree1' to the training set
y_predict_train = regtree1.predict(Xad_train)  # Predict the labels of training set
y_predict_test = regtree1.predict(Xad_test)  # Predict the labels of test set

print('CV RMSE:', MSE_CV.mean()**(0.5) )  #CV MSE 
print('Training set RMSE:', MSE(yad_train, y_predict_train)**(0.5) )   # Training set MSE
print('Test set RMSE:', MSE(yad_test, y_predict_test)**(0.5) )   # Test set MSE 

regtree1.score(Xad_test, yad_test)
#%% Try prediction
forecast_out = 15
#Create another column (the target ) shifted 'n' units up
df = btc[['priceusd']]
df['Prediction'] = df[['priceusd']].shift(-forecast_out)
#print the new data set
print(df.tail())

X = np.array(df.drop(['Prediction'],1))

#Remove the last '30' rows
X = X[:-forecast_out]
print(X)

y = np.array(df['Prediction'])
# Get all of the y values except the last '30' rows
y = y[:-forecast_out]
print(y)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 1)

lr = LinearRegression()
lr_predict = lr.fit(x_train, y_train)
lr_confidence = lr.score(x_test, y_test)
print("lr confidence: ", lr_confidence)

x_forecast = np.array(df.drop(['Prediction'],1))[-forecast_out:]
print(x_forecast)
lr_prediction = lr.predict(x_forecast)
print(lr_prediction)
#%% [markdown]
#
# #  Bias-variance tradeoff  
# high bias: underfitting  
# high variance: overfitting, too much complexity  
# Generalization Error = (bias)^2 + Variance + irreducible error  
# 
# Solution: Use CV  
# 
# 1. If CV error (average of 10- or 5-fold) > training set error  
#   - high variance
#   - overfitting the training set
#   - try to decrease model complexity
#   - decrease max depth
#   - increase min samples per leaf
#   - get more data
# 2. If CV error approximates the training set error, and greater than desired error
#   - high bias
#   - underfitting the training set
#   - increase max depth
#   - decrease min samples per leaf
#   - use or gather more relevant features

#%%
# Graphing the tree
from sklearn.tree import export_graphviz  
  
# export the decision tree to a tree.dot file 
# for visualizing the plot easily anywhere 
# import os
# dirpath = os.getcwd() # print("current directory is : " + dirpath)
path2add = 'd:/Download/George Washington University/Fall 2019/6103/Intro-to-data-mining-project'
filepath = os.path.join( dirpath, path2add ,'tree1')
export_graphviz(regtree1, out_file = filepath+'.dot' , feature_names =['txvolumeusd', 'adjustedtxvolumeusd', 'txcount', 'activeaddresses', 'blocksize']) 

# import pydot

# (graph,) = pydot.graph_from_dot_file(filepath)
# graph.write_png(filepath + '.png')

# %% Jane's part


#%% 
# 
######################### import the bitcoin data ############################
import os
dirpath = os.getcwd() # print("current directory is : " + dirpath)
path2add = 'C:\\Users\\Admin\\Documents\\GitHub\\Intro-to-data-mining-project'
filepath = os.path.join( dirpath, path2add ,'BTC_USD.csv')
import numpy as np
import pandas as pd
import csv

bitcoin = pd.read_csv(filepath,index_col=1)
# bitcoin = bitcoin.loc[bitcoin['Closing Price (USD)'] > 3000]


print(bitcoin.head())

print(bitcoin.describe())

print(bitcoin.dtypes)
# %% make histogram plot
# import matplotlib
# import matplotlib.pyplot as plt

# #bitcoin['Closing Price (USD)','Date'].plot(legend=True, figsize=(10, 5), title='Bitcoin Price', label='Closing Price (USD)')

# bitcoin.plot( y = 'Closing Price (USD)', legend=True, figsize=(10, 5), title='Bitcoin Daily Price', label = 'Bitcoin')
# plt.xlabel('Date')
# plt.ylabel('Closing Price (USD)')
# plt.savefig('bitcoinprice.png')


# %% 
######################## import S&P500 stock price ##########################3

path2add = 'C:\\Users\\Admin\\Documents\\GitHub\\Intro-to-data-mining-project'
filepath = os.path.join( dirpath, path2add ,'GSPC.csv')
stockindex = pd.read_csv(filepath,index_col=0)

print(stockindex.head(10))

print(stockindex.dtypes)

# %% 

############################ import gold index ###########################
path2add = 'C:\\Users\\Admin\\Documents\\GitHub\\Intro-to-data-mining-project'
filepath = os.path.join( dirpath, path2add ,'WGC-GOLD_DAILY_USD.csv')
goldindex= pd.read_csv(filepath, index_col=0)

goldindex.head()
goldindex.dtypes

goldindex['change'] = goldindex.pct_change()


# %% 

# Subtseting the date with price and calculate the change %
BTC = bitcoin[['Closing Price (USD)']]
BTC['change'] = BTC.pct_change()

# Subtseting the date with price and calculate the change %
SP500 = stockindex[['Close']]
SP500['change'] = SP500.pct_change()


# %% 
##################### combine bitcoin and stock price into one plot ####################333

import matplotlib
import matplotlib.pyplot as plt


BTC['change'].hist( bins = 50, label = 'BTC', figsize = (10,8), alpha = 0.5)
SP500['change'].hist( bins = 50, label = 'S&P 500', alpha = 0.5)
goldindex['change'].hist( bins = 50, label = 'Gold', alpha = 0.5)
plt.legend()


#%%

# ax = plt.gca()

# bitcoin.plot(kind='line', y = 'Closing Price (USD)', legend=True, figsize=(10, 5), label = 'Bitcoin', ax = ax)
# stockindex.plot(kind='line', y = 'Close', legend=True, figsize=(10, 5), label = 'S&P 500', ax = ax, color = 'red')
# plt.xlabel('Date')
# plt.ylabel('Closing Price (USD)')
# plt.show()


# # %%
# plt.figure(figsize=(10,8))
# top = plt.subplot2grid((4,4), (0, 0), rowspan=3, colspan=4)
# bottom = plt.subplot2grid((4,4), (3,0), rowspan=3, colspan=4)
# top.plot(BTC.index, BTC['Closing Price (USD)']) #CMT.index gives the dates
# bottom.plot(SP500.index, SP500['Close']) 
 
# # set the labels
# top.axes.get_xaxis().set_visible(False)
# top.set_title('Bitcoin Data')
# top.set_ylabel('Closing Price (USD)')
# bottom.set_title('S&P 500 Data')
# bottom.set_ylabel('Close')
# ax = plt.gca()
# ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())
# ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y'))
#ax.tick_params(pad=20)

# fig = plt.figure()
# # Divide the figure into a 2x1 grid, and give me the first section
# ax1 = fig.add_subplot()

# # Divide the figure into a 2x1 grid, and give me the second section
# ax2 = ax1.twinx()

# BTC.plot( y='Closing Price (USD)', ax=ax1, figsize=(10, 8), legend=True, label = 'BTC')
# ax1.xaxis.set_label_text("")
# ax1.set_title("Bitcoin Index vs S&P 500")
# ax1.set_ylabel('Closing Price (USD)')
# ax1.legend(loc=1)

# SP500.plot( y='Close', kind='line', ax=ax2, figsize=(10, 8), label = 'S&P 500', color = 'red')
# ax2.yaxis.set_label_text("")
# ax2.set_ylabel('Closing Price (USD)')
# ax2.legend(loc=2)

#%%

fig = plt.figure(figsize=(15, 10))
# Divide the figure into a 2x1 grid, and give me the first section
ax1=fig.add_subplot(111, label="1")
ax2=fig.add_subplot(111, label="2", frame_on=False)

BTC.plot( y='Closing Price (USD)', ax=ax1, legend=True, label = 'BTC')
# ax1.xaxis.set_label_text("")
# ax1.set_title("Bitcoin Index vs S&P 500")
ax1.set_ylabel('Closing Price (USD)')
ax1.legend(loc=1)
ax1.tick_params(axis='x')
ax1.tick_params(axis='y')

SP500.plot( y='Close', kind='line', ax=ax2, label = 'S&P 500', color = 'red')
ax2.xaxis.tick_top()
ax2.yaxis.tick_right()
ax2.xaxis.set_label_position('top') 
ax2.yaxis.set_label_position('right')
ax2.set_xlabel('Date', color="red") 
ax2.set_ylabel('Closing Price (USD)', color='red')  
ax2.legend(loc=2)
ax2.tick_params(axis='x', colors="red")
ax2.tick_params(axis='y', colors="red")


#%%
################ BTC and S&P 500 daily change percentage #######################


# ax = plt.gca()
# BTCchange.plot(kind='line', y = 'Closing Price (USD)', legend=True, figsize=(10, 5), label = 'Bitcoin', ax = ax)
# SPchange.plot(kind='line', y = 'Close', legend=True, figsize=(10, 5), label = 'S&P 500', ax = ax, color = 'red')
# plt.xlabel('Date')
# plt.ylabel('Percentage(%)')
# plt.show()

# fig = plt.figure()
# # Divide the figure into a 2x1 grid, and give me the first section
# ax1 = fig.add_subplot(211)

# # Divide the figure into a 2x1 grid, and give me the second section
# ax2 = fig.add_subplot(212)

# BTCchange.plot( y='Closing Price (USD)', ax=ax1, legend=False, figsize=(15, 8))
# ax1.xaxis.set_label_text("")
# ax1.set_title("Bitcoin Data")
# ax1.set_ylabel('Price Change %')

# SPchange.plot( y='Close', kind='line', ax=ax2, figsize=(15, 8))
# ax2.yaxis.set_label_text("")
# ax2.set_title("S&P 500 Data")
# ax2.set_ylabel('Price Change %')
# fig.subplots_adjust(hspace=0.3)


#%%

# ############### BTC and Gold daily Price #######################

# import matplotlib
# import matplotlib.pyplot as plt
# ax = plt.gca()

# bitcoin.plot(kind='line', y = 'Closing Price (USD)', legend=True, figsize=(10, 5), label = 'Bitcoin', ax = ax)
# goldindex.plot(kind='line', y = 'Value', legend=True, figsize=(10, 5), label = 'Gold', ax = ax, color = 'red')
# plt.xlabel('Date')
# plt.ylabel('Closing Price (USD)')
# plt.show()

# ax = plt.gca()
# BTCchange.plot(kind='line', y = 'Closing Price (USD)', legend=True, figsize=(10, 5), label = 'Bitcoin', ax = ax)
# goldchange.plot(kind='line', y = 'Value', legend=True, figsize=(10, 5), label = 'Gold', ax = ax, color = 'red')
# plt.xlabel('Date')
# plt.ylabel('Percentage(%)')
# plt.show()

# fig = plt.figure()
# # Divide the figure into a 2x1 grid, and give me the first section
# ax1 = fig.add_subplot(211)

# # Divide the figure into a 2x1 grid, and give me the second section
# ax2 = fig.add_subplot(212)

# BTC.plot( y='Closing Price (USD)', ax=ax1, legend=False, figsize=(10, 8))
# ax1.xaxis.set_label_text("")
# ax1.set_title("Bitcoin Data")
# ax1.set_ylabel('Closing Price (USD)')

# goldindex.plot( y='Value', kind='line', ax=ax2, figsize=(10, 8))
# ax2.yaxis.set_label_text("")
# ax2.set_title("Gold Data")
# ax2.set_ylabel('Value (USD)')
# fig.subplots_adjust(hspace=0.3)

# #%%

# fig = plt.figure()
# # Divide the figure into a 2x1 grid, and give me the first section
# ax1 = fig.add_subplot()

# # Divide the figure into a 2x1 grid, and give me the second section
# ax2 = ax1.twinx()

# BTC.plot( y='Closing Price (USD)', ax=ax1, figsize=(10, 8), legend=True, label = 'BTC')
# ax1.xaxis.set_label_text("")
# ax1.set_title("Bitcoin Index vs Gold Index")
# ax1.set_ylabel('Closing Price (USD)')
# ax1.legend(loc=1)

# goldindex.plot( y='Value', kind='line', ax=ax2, figsize=(10, 8), label = 'Gold', color = 'red')
# ax2.yaxis.set_label_text("")
# ax2.set_ylabel('Value (USD)')
# ax2.legend(loc=2)

#%%

fig=plt.figure(figsize=(15, 10))
ax1=fig.add_subplot(111, label="1")
ax2=fig.add_subplot(111, label="2", frame_on=False)

BTC.plot( y='Closing Price (USD)', ax=ax1, legend=True, label = 'BTC')
ax1.xaxis.set_label_text("")
ax1.set_xlabel("Date")
ax1.set_ylabel("Closing Price (USD)")
ax1.tick_params(axis='x')
ax1.tick_params(axis='y')
ax1.legend(loc=1)

goldindex.plot( y='Value', kind='line', ax=ax2, label = 'Gold', color = 'red')
ax2.xaxis.tick_top()
ax2.yaxis.tick_right()
ax2.yaxis.set_label_text("")
ax2.set_xlabel('Date', color="red") 
ax2.set_ylabel('Closing Price (USD)', color='red')  
ax2.xaxis.set_label_position('top') 
ax2.yaxis.set_label_position('right') 
ax2.tick_params(axis='x', colors="red")
ax2.tick_params(axis='y', colors="red")
ax2.legend(loc=2)

#%%
################ BTC and Gold daily change percentage #######################

# fig = plt.figure()
# # Divide the figure into a 2x1 grid, and give me the first section
# ax1 = fig.add_subplot(211)

# # Divide the figure into a 2x1 grid, and give me the second section
# ax2 = fig.add_subplot(212)

# BTCchange.plot( y='Closing Price (USD)', ax=ax1, legend=False, figsize=(15, 8))
# ax1.xaxis.set_label_text("")
# ax1.set_title("Bitcoin Data")
# ax1.set_ylabel('Price Change %')

# goldchange.plot( y='Value', kind='line', ax=ax2, figsize=(15, 8))
# ax2.yaxis.set_label_text("")
# ax2.set_title("Gold Data")
# ax2.set_ylabel('Value Change %')
# fig.subplots_adjust(hspace=0.3)

# #%%
# from statsmodels.graphics.tsaplots import plot_acf
# from matplotlib import pyplot

# plot_acf(BTC)
# pyplot.show()

# #%%

# from statsmodels.graphics.tsaplots import plot_pacf
# plot_pacf(BTC, lags=100)
# pyplot.show()
#%%
################ Try to find correlation#######################

from statsmodels.formula.api import ols

BTCSP = pd.merge(BTC, SP500, on='Date')
BTCSP = BTCSP.rename(columns={"Closing Price (USD)": "btc", "Close": "sp500"}).dropna()

modelBTCSP = ols(formula='btc ~ sp500', data=BTCSP).fit()
print( modelBTCSP.summary() )


##### gold
BTCGOLD = pd.merge(BTC, goldindex, on='Date')
BTCGOLD = BTCGOLD.rename(columns={"Closing Price (USD)": "btc", "Value": "gold"}).dropna()

modelBTCGOLD = ols(formula='btc ~ gold', data=BTCGOLD).fit()
print( modelBTCGOLD.summary() )


# %%
################ Try to find correlation of change %#######################

BTC1 = bitcoin[['Closing Price (USD)']]
BTCchange = BTC1.pct_change()

# Subtseting the date with price and calculate the change %
SP5001 = stockindex[['Close']]
SPchange = SP5001.pct_change()

goldchange1 = goldindex.pct_change()
#%%
#####sp500
from statsmodels.formula.api import ols
BTCSPchange = pd.merge(BTCchange, SPchange, on='Date')
BTCSPchange = BTCSPchange.rename(columns={"Closing Price (USD)": "btc", "Close": "sp500"}).dropna()

modelBTCSP = ols(formula='btc ~ sp500', data=BTCSPchange).fit()
print( modelBTCSP.summary() )

##### gold
BTCGOLDchange = pd.merge(BTCchange, goldchange1, on='Date')
BTCGOLDchange = BTCGOLDchange.rename(columns={"Closing Price (USD)": "btc", "Value": "gold"}).dropna()

modelBTCGOLD = ols(formula='btc ~ gold', data=BTCGOLDchange).fit()
print( modelBTCGOLD.summary() )

#%%

##### gold and sp500

GSP = pd.merge(SPchange, goldchange1, on='Date')
GSP= GSP.rename(columns={"Close": "SP500", "Value": "gold"}).dropna()

modelGSP = ols(formula='SP500 ~ gold', data=GSP).fit()
print( modelGSP.summary() )

#%%

# import os
# dirpath = os.getcwd() # print("current directory is : " + dirpath)
# path2add = 'C:\\Users\\Admin\\Documents\\GitHub\\DM_FinalProject'
# filepath = os.path.join( dirpath, path2add ,'BTC_USD.csv')
# import numpy as np
# import pandas as pd
# import csv

# bitcoin = pd.read_csv(filepath)


# print(bitcoin.head())

# print(bitcoin.describe())

# print(bitcoin.dtypes)



#%%
forecast_out = 10 #'n=30' days
BTC1['Prediction'] = BTC1[['Closing Price (USD)']].shift(-forecast_out)
print(BTC1.tail(10))

#%%

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
### Create the independent data set (X)  #######
# Convert the dataframe to a numpy array
X = np.array(BTC1.drop(['Prediction'],1))

#Remove the last '30' rows
X = X[:-forecast_out]
print(X)


y = np.array(BTC1['Prediction'])
# Get all of the y values except the last '30' rows
y = y[:-forecast_out]
print(y)



x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1) 
# svr_rbf.fit(x_train, y_train)


# # Testing Model: Score returns the coefficient of determination R^2 of the prediction. 
# # The best possible score is 1.0
# svm_confidence = svr_rbf.score(x_test, y_test)
# print("svm confidence: ", svm_confidence)

# %%
lr = LinearRegression()
# Train the model
lr.fit(x_train, y_train)

lr_confidence = lr.score(x_test, y_test)
print("lr confidence: ", lr_confidence)

#%%

x_forecast = np.array(BTC1.drop(['Prediction'],1))[-forecast_out:]
print(x_forecast)# %%


# %%

# Print linear regression model predictions for the next '30' days
lr_prediction = lr.predict(x_forecast)
arr = lr_prediction[::-1] 

print(np.asarray(arr).reshape(10,1))
# Print support vector regressor model predictions for the next '30' days
# svm_prediction = svr_rbf.predict(x_forecast)
# print(np.asarray(svm_prediction).reshape(5,1))

# %%


# %%

