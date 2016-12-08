import statsmodels.api as sm
from statsmodels.regression.linear_model import RegressionResults
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

# this is where i stored my data
mydata = pd.read_excel("C:/Users/Victor/Documents/mwangi/empirical finance/EMP FINANCE DATA.xlsx",
                       sheetname="DEMEANED",
                       index_col="DATE")
# the following were the headers in my columns
# u can adjust your data to have similar headers before running this code
'''Index(['LOG DIVYIELD', 'LOGDIVIDEND GROWTH', 'LED LOG DIVYIELD',
 'LED LOG DIV GROWTH', 'LED LOGRETURN'])'''

print(mydata.dtypes.index)
length_of_each_dataset = len(mydata.index)
numberofdatasets = 2000

# regress yield(t+1) on yield
regress_model = sm.OLS(mydata['LED LOG DIVYIELD'], mydata["LOG DIVYIELD"])
results = regress_model.fit()
parameter = results.params
phi = parameter[0]
print(phi)
er = np.array(RegressionResults.resid(self=results))
std_dev1 = np.std(er)[0]
var1 = np.var(er)

#regress growth(t+1) on yield
regress_model2 = sm.OLS(mydata['LED LOG DIV GROWTH'], mydata["LOG DIVYIELD"])
results2 = regress_model2.fit()
bd = results2.params[0]
print(bd)
er2 = np.array(RegressionResults.resid(self=results))
std_dev2 = np.std(er)
var2 = np.var(er2)

# regress returns(t+1) on yield
regr3 = sm.OLS(mydata['LED LOGRETURN'], mydata["LOG DIVYIELD"])
results3 = regr3.fit()
br = results3.params[0]

# to get rho
div_yield = np.array(mydata["LOG DIVYIELD"])
logpd = div_yield * -1
itsmean = np.mean(logpd)
numerator = math.exp(itsmean)
denominator = 1 + numerator
rho = numerator / denominator

# simulate d0-p0

for i in range(0, 10):
    if phi < 1:
        std_dev_of_yield = math.sqrt(var1 / (1 - phi ** 2))
        d0p0 = np.random.normal(0, std_dev_of_yield, (1, numberofdatasets))
    elif phi > 1:
        d0p0 = np.zeros((1, numberofdatasets))

# simulate errors for divyield equation
er1sim = np.random.normal(0, std_dev1, (length_of_each_dataset, numberofdatasets))

# put phi in an array
totalphisused = numberofdatasets * length_of_each_dataset
emptyarray = np.empty(totalphisused)
emptyarray.fill(phi)
phiarray = emptyarray.reshape(length_of_each_dataset, numberofdatasets)
# print(phiarray)
# print(phiarray.shape[0])



# get simulations of yield
simulated_divyields = np.array(d0p0)
for j in range(0, length_of_each_dataset):
    nextrow = phiarray[j, :] * simulated_divyields[j, :] + er1sim[j, :]
    simulated_divyields = np.vstack((simulated_divyields, nextrow))

# divyieldsim=np.delete(divyieldsim,(0), axis=0)
# print(divyieldsim)



# get array of rho
totalrhosused = numberofdatasets * length_of_each_dataset
emptyarray3 = np.empty(totalrhosused)
emptyarray3.fill(rho)
rhos_array = emptyarray3.reshape(length_of_each_dataset, numberofdatasets)

# coefficient for dividend growth regresion against dividend yield
len_ofdivyieldsim = length_of_each_dataset + 1
array1 = np.ones((length_of_each_dataset, numberofdatasets))
growthcoeff = phiarray * rhos_array - array1

# simulate errors for dividend growth equation
er2sim = np.random.normal(0, std_dev2, (len_ofdivyieldsim, numberofdatasets))

# simulate dividend growth
divempty = np.empty((1, numberofdatasets))
divgrowtharray = np.vstack(divempty)
for i in range(0, len_ofdivyieldsim - 1):
    nextrow2 = growthcoeff[i, :] * simulated_divyields[i, :] + er2sim[i, :]
    divgrowtharray = np.vstack((divgrowtharray, nextrow2))

# remove last row of divgrwth errors
er2sim = er2sim[0:2712, :]
print(er2sim.shape)
# remove the extra simulation in yield and growth
# all this is because of the first dividend yield that we had to simulate.....
# we are making all the series have equal length
simulated_divyields = simulated_divyields[0:2712, :]
divgrowtharray = divgrowtharray[0:2712, :]

# get simulated returns
simulated_returns = er1sim + rhos_array * er2sim

# make sure all simulations are of same length/shape
print(simulated_divyields.shape)
print(divgrowtharray.shape)
print(simulated_returns.shape)

# lead the div yield forward by one period
led_dividendyields = simulated_divyields[1:2712, :]

# now remove the last row from each of the other series so that regressions will have equal series
# the other series are already led because thats how they're simulated
divgrowtharray = divgrowtharray[0:2711, :]
simulated_returns = simulated_returns[0:2711, :]
simulated_divyields = simulated_divyields[0:2711, :]

# confirm shapes of all series
print(simulated_divyields.shape)
print(divgrowtharray.shape)
print(simulated_returns.shape)
print(led_dividendyields.shape)
print(led_dividendyields[:, 1])
length_ofmy_series = simulated_divyields.shape[0]


# now do the regressions
# this function will do the regression and return coefficients and t-statistics
def simregression(y, x):
    coefflist = []
    tstats = []
    # list of coeffecients
    for z in range(0, y.shape[1]):
        model1 = sm.OLS(y[:, z], x[:, z])
        regresults1 = model1.fit()
        coeff1 = regresults1.params
        coefflist.append(coeff1[0])
        tstats.append(regresults1.tvalues[0])
    return coefflist, tstats

# bd and br stand for dividend yield and dividend growth coefficients
# am just putting them in a list
bd_list, bd_tstat_list = simregression(divgrowtharray, simulated_divyields)
br_list, br_tstat_list = simregression(simulated_returns, simulated_divyields)
plt.scatter(br_list,bd_list)
plt.axhline(y=bd)
plt.axvline(x=br)
# the following coordinates are specific to the results i got
# for your data it might be different
plt.text(-0.05,0.05,"1")
plt.text(-0.05,-0.2,"2")
plt.text(0.05,-0.2,"3")
plt.text(0.05,0.05,"4")
plt.show()
