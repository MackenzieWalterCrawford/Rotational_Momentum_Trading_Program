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


class ETFRanker:
    def __init__(
        self, close_price_df: pd.DataFrame, adj_close_df: pd.DataFrame, param_dict: dict
    ):
        self.logReturns = param_dict["logReturns"]
        self.Aperiods = param_dict["Aperiods"]  # Lookback 20 Default
        self.Frequency = param_dict["Frequency"]
        self.momentum = param_dict["momentum"]  # experimental mean reversion trading
        self.volmomentum = 0  # do not change, experimental mean reversion trading
        self.CashFilter = param_dict["CashFilter"]
        self.MAperiods = param_dict["MAperiods"]
        self.Delay = param_dict["Delay"]
        self.ShortTermWeight = param_dict["ShortTermWeight"]
        self.LongTermWeight = param_dict["LongTermWeight"]
        self.ShortTermVolatilityWeight = param_dict["ShortTermVolatilityWeight"]
        self.StandsForCash = param_dict["StandsForCash"]
        self.Zboundary = param_dict["Zboundary"]
        self.Zperiods = param_dict["Zperiods"]
        self.max_holding = param_dict["max_holding"]
        self.step_holding = param_dict["step_holding"]
        self.max_lookback = param_dict["max_lookback"]
        self.step_lookback = param_dict["step_lookback"]

        self.close_price_df = close_price_df.sort_values(by="Date")
        self.adj_close_df = adj_close_df.sort_values(by="Date")
        self.sharpe_surface_df = pd.DataFrame(columns=["lookback", "holding", "sharpe"])
        self.sharpe_matrix = np.zeros(shape=(self.max_lookback, self.max_holding))

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

    def rank_etfs(self):
        # lookbacks for longer term period and volatily
        Bperiods = 3 * self.Aperiods + ((3 * self.Aperiods) // 20) * 2  # 66 Default
        Speriods = self.Aperiods  # 20 Default

        # dfA contains a short moving average of the daily percent changes, calculated for each ETF
        # dfB contains a long moving average of the daily percent changes, calculated for each ETF
        # dfS contains the annualized volatility, calculated for each ETF
        # dfMA contains 200 MA of price
        # dfDetrend contains the detrended AP prices (for White's reality test)

        dfA = self.close_price_df.drop(
            labels=None, axis=1, columns=self.close_price_df.columns
        )
        dfB = self.close_price_df.drop(
            labels=None, axis=1, columns=self.close_price_df.columns
        )
        dfS = self.close_price_df.drop(
            labels=None, axis=1, columns=self.close_price_df.columns
        )
        dfZ = self.close_price_df.drop(
            labels=None, axis=1, columns=self.close_price_df.columns
        )
        dfMA = self.close_price_df.drop(
            labels=None, axis=1, columns=self.close_price_df.columns
        )
        dfDetrend = self.close_price_df.drop(
            labels=None, axis=1, columns=self.close_price_df.columns
        )

        # calculating the three performance measures in accordance with their windows

        dfA = self.close_price_df.pct_change(
            periods=self.Aperiods - 1, fill_method="pad", limit=None, freq=None
        )  # is counting window from 0
        dfB = self.close_price_df.pct_change(
            periods=Bperiods - 1, fill_method="pad", limit=None, freq=None
        )  # is counting window from 0
        dfR = self.close_price_df.pct_change(
            periods=1, fill_method="pad", limit=None, freq=None
        )  # is counting window from 0

        columns = self.close_price_df.shape[1]
        for column in range(columns):
            dfS[self.close_price_df.columns[column]] = (
                dfR[self.close_price_df.columns[column]].rolling(window=Speriods).std()
            ) * math.sqrt(252)
            dfZ[self.close_price_df.columns[column]] = (
                self.close_price_df[self.close_price_df.columns[column]]
                - self.close_price_df[self.close_price_df.columns[column]]
                .rolling(window=self.Zperiods)
                .mean()
            ) / self.close_price_df[self.close_price_df.columns[column]].rolling(
                window=self.Zperiods
            ).std()
            dfMA[self.close_price_df.columns[column]] = (
                self.close_price_df[self.close_price_df.columns[column]]
                .rolling(window=self.MAperiods)
                .mean()
            )
            dfDetrend[self.adj_close_df.columns[column]] = detrendPrice.detrendPrice(
                self.adj_close_df[self.adj_close_df.columns[column]]
            ).values

        # Ranking each ETF w.r.t. short moving average of returns
        dfA_ranks = self.close_price_df.copy(deep=True)
        dfA_ranks[:] = 0

        columns = dfA_ranks.shape[1]

        # this loop takes each row of the A dataframe, puts the row into an array,
        # within the array the contents are ranked,
        # then the ranks are placed into the A_ranks dataframe one by one

        arr_row = dfA.iloc[-1].values
        if self.momentum == 1:
            temp = arr_row.argsort()  # sort momentum, best is ETF with largest return
        else:
            temp = (-arr_row).argsort()[
                : arr_row.size
            ]  # sort reversion, best is ETF with lowest return
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(1, len(arr_row) + 1)
        for column in range(columns):
            dfA_ranks.iat[-1, column] = ranks[column]

        dfB_ranks = self.close_price_df.copy(deep=True)
        dfB_ranks[:] = 0

        columns = dfB_ranks.shape[1]
        rows = dfB_ranks.shape[0]

        # this loop takes each row of the B dataframe, puts the row into an array,
        # Take the most recent row (i.e. today) and rank contents
        # then the ranks are placed into the B_ranks dataframe, which is only one row (today)

        arr_row = dfB.iloc[-1].values
        if self.momentum == 1:
            temp = arr_row.argsort()  # sort momentum
        else:
            temp = (-arr_row).argsort()[: arr_row.size]  # sort reversion
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(1, len(arr_row) + 1)
        for column in range(columns):
            dfB_ranks.iat[-1, column] = ranks[column]

        dfS_ranks = self.close_price_df.copy(deep=True)
        dfS_ranks[:] = 0

        columns = dfS_ranks.shape[1]
        rows = dfS_ranks.shape[0]

        # this loop takes each row of the dfS dataframe, puts the row into an array,
        # within the array the contents are ranked,
        # then the ranks are placed into the dfS_ranks dataframe one by one
        for row in range(rows):
            arr_row = dfS.iloc[row].values
            if self.volmomentum == 1:
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
        dfA_ranks = dfA_ranks.multiply(self.ShortTermWeight)  # .3 default
        dfB_ranks = dfB_ranks.multiply(self.LongTermWeight)  # .4 default
        dfS_ranks = dfS_ranks.multiply(self.ShortTermVolatilityWeight)  # .3 default
        dfAll_ranks = dfA_ranks.add(dfB_ranks, fill_value=0)
        dfAll_ranks = dfAll_ranks.add(dfS_ranks, fill_value=0)
        #################################################################################################################
        #################################################################################################################

        # Choice is the dataframe where the ETF with the maximum score is identified
        dfChoice = self.close_price_df.copy(deep=True)
        dfChoice[:] = 0
        rows = dfChoice.shape[0]

        # this loop takes each row of the All-ranks dataframe, puts the row into an array,
        # within the array the contents scanned for the maximum element
        # then the maximum element is placed into the Choice dataframe

        arr_row = dfAll_ranks.iloc[-1].values
        if self.momentum == 0 or self.CashFilter == 1:
            arr_row = arr_row[
                0 : len(arr_row) - 1
            ]  # don't rank SHY (the rightmost column) if doing reversion to mean trading or if requested to ignore it
        max_arr_column = np.argmax(arr_row, axis=0)  # gets the INDEX of the max
        if self.CashFilter == 1:
            if (
                self.close_price_df[self.close_price_df.columns[max_arr_column]][-1]
                >= dfMA[dfMA.columns[max_arr_column]][-1]
            ):  # "200MA" condition for cash filter
                # if (dfZ[dfZ.columns[max_arr_column]][row] > Zboundary): #alternative cash filter
                dfChoice.iat[-1, max_arr_column] = 1
            else:
                dfChoice.iat[-1, self.close_price_df.columns.get_loc(StandsForCash)] = 1
        else:
            dfChoice.iat[-1, max_arr_column] = 1

        combine_df = dfAll_ranks.join(
            dfChoice, lsuffix="_rank", rsuffix="_choice", on="Date", how="left"
        )

        final_df = combine_df[-1:]

        # nprint(final_df.head())
        return final_df.head()
