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

# %%
