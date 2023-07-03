import pandas as pd
import numpy as np
import math
import datetime
import matplotlib.pyplot as plt
from matplotlib import style
import detrendPrice
import WhiteRealityCheckFor1

# from Data_ingestion_Rot_Momentum import *
import yfinance as yf

yf.pdr_override()


def getDate(dt):
    if type(dt) != str:
        return dt
    try:
        datetime_object = datetime.datetime.strptime(dt, "%Y-%m-%d")
    except Exception:
        datetime_object = datetime.datetime.strptime(dt, "%m/%d/%Y")
        return datetime_object
    else:
        return datetime_object


# All COLUMN_NAMES are capitalized
# IMPORTANT: ALL ETF LISTS  MUST END WITH SHY (or equivalent) ETF (=rightmost column of the dataframe)
# SHY equivalent = a bond ETF
# dfP are the prices, used for the calculation of trading signals
# dfAP are the adjusted prices, used for the calculation of portfolio returns

dfP = pd.read_csv(
    "VSC.TO.VSB.TO.ZIC.TO.HBB.TO.TDB.TO.XBB.TO.XGB.TO.csv", parse_dates=["Date"]
)
dfAP = pd.read_csv(
    "VSC.TO.VSB.TO.ZIC.TO.HBB.TO.TDB.TO.XBB.TO.XGB.TO_AP.csv", parse_dates=["Date"]
)

dfP = dfP.sort_values(by="Date")
dfAP = dfAP.sort_values(by="Date")
dfP.set_index("Date", inplace=True)
dfAP.set_index("Date", inplace=True)

# logReturns = 1 means log returns will be used in the calculation of portfolio returns, 0 means pct_changes
# Aperiods = lookback, determines what counts as the short term period, 20 days is the Default setting
# the selection of the ETF is based on maximum weighted score of:

# A returns, or returns in the short term period

# B returns, or returns in the longer term period (here set to 3 times the short term period)

# volatility, or volatility in the short term period.

# Frequency= holding period, how long to wait to rebalance, 18W-FRI is the Default setting for SPY-TLT combination.
# The frequency of a given ETF list can be determined by generating a Sharpe surface, assuming Fridays is the best trading day.

# Frequency="B" every business day, "D" every calendar day; "10B" means every 10 business days ("2B" is the minimum)
# Frequency="W" every weeek, "2W" for every 2 weeks, "3W" every 3 weeks etc
# Frequency="W-TUE" every Tuesday, "2W-TUE" for every 2 Tuesdays, "3W-TUE" every 3 Tuesdays etc
# Frequency= "BM" every month, "2BM" for every 2 months, "3BM" every 3 months etc; B relates to business days; 31 or previous business day if necessary
# Frequency="SM" on the middle (15) and end (31) of the month, or previous business day if necessary

# Delay = 1 if the trade occurs instantaneously with the signal, 2 if the trade occurs 1 day after the signal

# ShortTermWeight = the weight of short term returns, .3 by default
# LongTermWeight = the weight of long term returns, .4 by default
# ShortTermVolatilityWeight = the weight of short term volatility, .4
# ShortTermWeight+LongTermWeight+ShortTermVolatilityWeight should add up to 1.

# other parameters (experimental)
# Cash Filter = 0 (or 1) forces the portfolio to invest in whatever ETF in the list "StandsForCash"
# MAperiods = moving average, usually 200, that prices have to be above of to allow trading, if the cash filter is set to 1.
# StandsForCash = "TLT" #what the system invests in when the CashFilter is triggered, the rightmost ETF in the dataframe
# if you set cash filter to 1, you need to identify what ETF StandsForCash (should be the rightmost ETF in the dataframe)
# Zboundary = -1.5 #alternative cash filter (but need understand and to uncomment line: alternative cash filter)
# Zperiods = 200  #alternative cash filter  (but need understand and to uncomment line: alternative cash filter)
# momentum = 1 means A and B returns are ranked in increasing order (momentum), 0 in decreasing order (reversion to the mean)
# volmomentum = 1 volatility ranked in increasing order (momentum), 0 in decreasing order (reversion to the mean)

logReturns = 0
Aperiods = 8 * 5  # Lookback 20 Default
Frequency = "18W-FRI"  # fridays and thursdays are the best trading days
ShortTermWeight = 0.3
LongTermWeight = 0.4
ShortTermVolatilityWeight = 0.4
Delay = 1


# experimental cash filter
CashFilter = 0
MAperiods = 200
StandsForCash = "SHY"
Zboundary = -1.5
Zperiods = 200

# experimental mean reversion trading
momentum = 1
volmomentum = 0  # do not change

# lookbacks for longer term period and volatily
Bperiods = 3 * Aperiods + ((3 * Aperiods) // 20) * 2  # 66 Default
Speriods = Aperiods  # 20 Default

# dfA contains a short moving average of the daily percent changes, calculated for each ETF
# dfB contains a long moving average of the daily percent changes, calculated for each ETF
# dfS contains the annualized volatility, calculated for each ETF
# dfMA contains 200 MA of price
# dfDetrend contains the detrended AP prices (for White's reality test)

dfA = dfP.drop(labels=None, axis=1, columns=dfP.columns)
dfB = dfP.drop(labels=None, axis=1, columns=dfP.columns)
dfS = dfP.drop(labels=None, axis=1, columns=dfP.columns)
dfZ = dfP.drop(labels=None, axis=1, columns=dfP.columns)
dfMA = dfP.drop(labels=None, axis=1, columns=dfP.columns)
dfDetrend = dfP.drop(labels=None, axis=1, columns=dfP.columns)

# calculating the three performance measures in accordance with their windows

dfA = dfP.pct_change(
    periods=Aperiods - 1, fill_method="pad", limit=None, freq=None
)  # is counting window from 0
dfB = dfP.pct_change(
    periods=Bperiods - 1, fill_method="pad", limit=None, freq=None
)  # is counting window from 0
dfR = dfP.pct_change(
    periods=1, fill_method="pad", limit=None, freq=None
)  # is counting window from 0

columns = dfP.shape[1]
for column in range(columns):
    dfS[dfP.columns[column]] = (
        dfR[dfP.columns[column]].rolling(window=Speriods).std()
    ) * math.sqrt(252)
    dfZ[dfP.columns[column]] = (
        dfP[dfP.columns[column]]
        - dfP[dfP.columns[column]].rolling(window=Zperiods).mean()
    ) / dfP[dfP.columns[column]].rolling(window=Zperiods).std()
    dfMA[dfP.columns[column]] = (
        dfP[dfP.columns[column]].rolling(window=MAperiods).mean()
    )
    dfDetrend[dfAP.columns[column]] = detrendPrice.detrendPrice(
        dfAP[dfAP.columns[column]]
    ).values

# Ranking each ETF w.r.t. short moving average of returns
dfA_ranks = dfP.copy(deep=True)
dfA_ranks[:] = 0

columns = dfA_ranks.shape[1]
rows = dfA_ranks.shape[0]

# this loop takes each row of the A dataframe, puts the row into an array,
# within the array the contents are ranked,
# then the ranks are placed into the A_ranks dataframe one by one
for row in range(rows):
    arr_row = dfA.iloc[row].values
    if momentum == 1:
        temp = arr_row.argsort()  # sort momentum, best is ETF with largest return
    else:
        temp = (-arr_row).argsort()[
            : arr_row.size
        ]  # sort reversion, best is ETF with lowest return
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(1, len(arr_row) + 1)
    for column in range(columns):
        dfA_ranks.iat[row, column] = ranks[column]

dfB_ranks = dfP.copy(deep=True)
dfB_ranks[:] = 0

columns = dfB_ranks.shape[1]
rows = dfB_ranks.shape[0]

# this loop takes each row of the B dataframe, puts the row into an array,
# within the array the contents are ranked,
# then the ranks are placed into the B_ranks dataframe one by one
for row in range(rows):
    arr_row = dfB.iloc[row].values
    if momentum == 1:
        temp = arr_row.argsort()  # sort momentum
    else:
        temp = (-arr_row).argsort()[: arr_row.size]  # sort reversion
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(1, len(arr_row) + 1)
    for column in range(columns):
        dfB_ranks.iat[row, column] = ranks[column]

dfS_ranks = dfP.copy(deep=True)
dfS_ranks[:] = 0

columns = dfS_ranks.shape[1]
rows = dfS_ranks.shape[0]

# this loop takes each row of the dfS dataframe, puts the row into an array,
# within the array the contents are ranked,
# then the ranks are placed into the dfS_ranks dataframe one by one
for row in range(rows):
    arr_row = dfS.iloc[row].values
    if volmomentum == 1:
        temp = arr_row.argsort()  # sort momentum, best is highest volatility
    else:
        temp = (-arr_row).argsort()[
            : arr_row.size
        ]  # sort reversion, best is lowest volatility
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(1, len(arr_row) + 1)
    for column in range(columns):
        dfS_ranks.iat[row, column] = ranks[column]

#################################################################################################################
# Weights of the varous ranks ####################################################################################
dfA_ranks = dfA_ranks.multiply(ShortTermWeight)  # .3 default
dfB_ranks = dfB_ranks.multiply(LongTermWeight)  # .4 default
dfS_ranks = dfS_ranks.multiply(ShortTermVolatilityWeight)  # .3 default
dfAll_ranks = dfA_ranks.add(dfB_ranks, fill_value=0)
dfAll_ranks = dfAll_ranks.add(dfS_ranks, fill_value=0)
#################################################################################################################
#################################################################################################################

# Choice is the dataframe where the ETF with the maximum score is identified
dfChoice = dfP.copy(deep=True)
dfChoice[:] = 0
rows = dfChoice.shape[0]

# this loop takes each row of the All-ranks dataframe, puts the row into an array,
# within the array the contents scanned for the maximum element
# then the maximum element is placed into the Choice dataframe
for row in range(rows):
    arr_row = dfAll_ranks.iloc[row].values
    if momentum == 0 or CashFilter == 1:
        arr_row = arr_row[
            0 : len(arr_row) - 1
        ]  # don't rank SHY (the rightmost column) if doing reversion to mean trading or if requested to ignore it
    max_arr_column = np.argmax(arr_row, axis=0)  # gets the INDEX of the max
    if CashFilter == 1:
        if (
            dfP[dfP.columns[max_arr_column]][row]
            >= dfMA[dfMA.columns[max_arr_column]][row]
        ):  # "200MA" condition for cash filter
            # if (dfZ[dfZ.columns[max_arr_column]][row] > Zboundary): #alternative cash filter
            dfChoice.iat[row, max_arr_column] = 1
        else:
            dfChoice.iat[row, dfP.columns.get_loc(StandsForCash)] = 1
    else:
        dfChoice.iat[row, max_arr_column] = 1

# dfPRR is the dataframe containing the log or pct_change returns of the ETFs
# will be based on adjusted prices rather than straight prices

if logReturns == 1:
    dfPLog = dfAP.apply(np.log)
    dfPLogShift = dfPLog.shift(1)
    dfPRR = dfPLog.subtract(dfPLogShift, fill_value=0)
    # repeat with detrended prices
    dfPLog = dfDetrend.apply(np.log)
    dfPLogShift = dfPLog.shift(1)
    dfDetrendRR = dfPLog.subtract(dfPLogShift, fill_value=0)

else:
    dfPRR = dfAP.pct_change()
    dfDetrendRR = dfDetrend.pct_change()

# T is the dataframe where the trading day is calculated.

dfT = dfP.drop(labels=None, axis=1, columns=dfP.columns)
columns = dfP.shape[1]
for column in range(columns):
    new = dfP.columns[column] + "_CHOICE"
    dfPRR[new] = pd.Series(np.zeros(rows), index=dfPRR.index)
    dfPRR[new] = dfChoice[dfChoice.columns[column]]

dfT["DateCopy"] = dfT.index
dfT1 = dfT.asfreq(freq=Frequency, method="pad")
dfT1.set_index("DateCopy", inplace=True)
dfTJoin = pd.merge(
    dfT, dfT1, left_index=True, right_index=True, how="outer", indicator=True
)

dfTJoin = dfTJoin.loc[
    ~dfTJoin.index.duplicated(keep="first")
]  # eliminates a row with a duplicate index which arises when using kibot data
dfPRR = pd.merge(dfPRR, dfTJoin, left_index=True, right_index=True, how="inner")
dfPRR.rename(columns={"_merge": Frequency + "_FREQ"}, inplace=True)
# dfPRR[Frequency+"_FREQ"] = dfTJoin["_merge"] #better to do this with merge as above

# _LEN means Long entry for that ETF
# _NUL means number units long of that ETF
# _LEX means long exit for that ETF
# _R means returns of that ETF (traded ETF)
# _ALL_R means returns of all ETFs traded, i.e. portfolio returns
# CUM_R means commulative returns of all ETFs, i.e. portfolio cummulative returns

columns = dfP.shape[1]
for column in range(columns):
    new = dfP.columns[column] + "_LEN"
    dfPRR[new] = (dfPRR[Frequency + "_FREQ"] == "both") & (
        dfPRR[dfP.columns[column] + "_CHOICE"] == 1
    )
    new = dfP.columns[column] + "_LEX"
    dfPRR[new] = (dfPRR[Frequency + "_FREQ"] == "both") & (
        dfPRR[dfP.columns[column] + "_CHOICE"] != 1
    )
    new = dfP.columns[column] + "_NUL"
    dfPRR[new] = np.nan
    dfPRR.loc[
        dfPRR[dfP.columns[column] + "_LEX"] == True, dfP.columns[column] + "_NUL"
    ] = 0
    dfPRR.loc[
        dfPRR[dfP.columns[column] + "_LEN"] == True, dfP.columns[column] + "_NUL"
    ] = 1  # this order is important
    dfPRR.iat[0, dfPRR.columns.get_loc(dfP.columns[column] + "_NUL")] = 0
    dfPRR[dfP.columns[column] + "_NUL"] = dfPRR[dfP.columns[column] + "_NUL"].fillna(
        method="pad"
    )
    new = dfP.columns[column] + "_R"
    dfPRR[new] = dfPRR[dfP.columns[column]] * dfPRR[dfP.columns[column] + "_NUL"].shift(
        Delay
    )
    # repeat for detrended returns
    dfDetrendRR[new] = dfDetrendRR[dfP.columns[column]] * dfPRR[
        dfP.columns[column] + "_NUL"
    ].shift(Delay)


# calculating all returns
dfPRR = dfPRR.assign(ALL_R=pd.Series(np.zeros(rows)).values)
# repeat for detrended returns
dfDetrendRR = dfDetrendRR.assign(ALL_R=pd.Series(np.zeros(rows)).values)


# the return of the portfolio is a sequence of returns made
# by appending sequences of returns of traded ETFs
# Since non traded returns are multiplied by zero, we only need to add the columns
# of the returns of each ETF, traded or not
columns = dfP.shape[1]
for column in range(columns):
    dfPRR["ALL_R"] = dfPRR["ALL_R"] + dfPRR[dfP.columns[column] + "_R"]
    # repeat for detrended returns
    dfDetrendRR["ALL_R"] = (
        dfDetrendRR["ALL_R"] + dfDetrendRR[dfP.columns[column] + "_R"]
    )

dfPRR = dfPRR.assign(DETREND_ALL_R=dfDetrendRR["ALL_R"])

# dfPRR['CUM_R'] = dfPRR['ALL_R'].cumsum()  #this is good only for log returns
# dfPRR['CUM_R'] = dfPRR['CUM_R'] + 1 #this is good only for log returns

# calculating portfolio investment column in a separate dataframe, using 'ALL_R' = portfolio returns

dfPRR = dfPRR.assign(I=np.cumprod(1 + dfPRR["ALL_R"]))  # this is good for pct return
dfPRR.iat[0, dfPRR.columns.get_loc("I")] = 1
# repeat for detrended returns
dfDetrendRR = dfDetrendRR.assign(
    I=np.cumprod(1 + dfDetrendRR["ALL_R"])
)  # this is good for pct return
dfDetrendRR.iat[0, dfDetrendRR.columns.get_loc("I")] = 1

dfPRR = dfPRR.assign(DETREND_I=dfDetrendRR["I"])

try:
    sharpe = (dfPRR["ALL_R"].mean() / dfPRR["ALL_R"].std()) * math.sqrt(252)
except ZeroDivisionError:
    sharpe = 0.0

style.use("fivethirtyeight")
dfPRR["I"].plot()
plt.legend()
plt.show()
# plt.savefig(r'Results\%s.png' %(title))
# plt.close()

start = 1
start_val = start
end_val = dfPRR["I"].iat[-1]


start_date = getDate(dfPRR.iloc[0].name)
end_date = getDate(dfPRR.iloc[-1].name)
days = (end_date - start_date).days

TotaAnnReturn = (end_val - start_val) / start_val / (days / 360)
TotaAnnReturn_trading = (end_val - start_val) / start_val / (days / 252)

CAGR_trading = round(
    ((float(end_val) / float(start_val)) ** (1 / (days / 252.0))).real - 1, 4
)  # when raised to an exponent I am getting a complex number, I need only the real part
CAGR = round(
    ((float(end_val) / float(start_val)) ** (1 / (days / 350.0))).real - 1, 4
)  # when raised to an exponent I am getting a complex number, I need only the real part

print("TotaAnnReturn = %f" % (TotaAnnReturn * 100))
print("CAGR = %f" % (CAGR * 100))
print("Sharpe Ratio = %f" % (round(sharpe, 2)))


# Detrending Prices and Returns
WhiteRealityCheckFor1.bootstrap(dfPRR["DETREND_ALL_R"])

dfPRR.to_csv(r"Results\dfPRR.csv", header=True, index=True, encoding="utf-8")
