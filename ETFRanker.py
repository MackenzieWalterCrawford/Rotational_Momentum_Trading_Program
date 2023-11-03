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

    def rank_dataframe(self, df_to_rank: pd.DataFrame):
        """
        args:
            df_to_rank (pd.DataFrame):

        Returns:
            ranks (pd.DataFrame): ranked DF

        """
        # Ranking each ETF w.r.t. short moving average of returns
        rank_df = self.close_price_df.copy(deep=True)
        rank_df[:] = 0

        columns = rank_df.shape[1]

        # this loop takes each row of the A dataframe, puts the row into an array,
        # within the array the contents are ranked,
        # then the ranks are placed into the A_ranks dataframe one by one

        arr_row = df_to_rank.iloc[-1].values
        if self.momentum == 1:
            temp = arr_row.argsort()  # sort momentum, best is ETF with largest return
        else:
            temp = (-arr_row).argsort()[
                : arr_row.size
            ]  # sort reversion, best is ETF with lowest return
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(1, len(arr_row) + 1)
        for column in range(columns):
            rank_df.iat[-1, column] = ranks[column]

        return rank_df

    def rank_etfs(self):
        # lookbacks for longer term period and volatily
        Bperiods = 3 * self.Aperiods + ((3 * self.Aperiods) // 20) * 2  # 66 Default
        Speriods = self.Aperiods  # 20 Default

        # short_MA contains a short moving average of the daily percent changes, calculated for each ETF
        # long_MA contains a long moving average of the daily percent changes, calculated for each ETF
        # ann_volatility contains the annualized volatility, calculated for each ETF
        # dfMA contains 200 MA of price
        # dfDetrend contains the detrended AP prices (for White's reality test)

        short_MA = self.close_price_df.drop(
            labels=None, axis=1, columns=self.close_price_df.columns
        )
        long_MA = self.close_price_df.drop(
            labels=None, axis=1, columns=self.close_price_df.columns
        )
        ann_volatility = self.close_price_df.drop(
            labels=None, axis=1, columns=self.close_price_df.columns
        )
        dfZ = self.close_price_df.drop(
            labels=None, axis=1, columns=self.close_price_df.columns
        )
        dfMA = self.close_price_df.drop(
            labels=None, axis=1, columns=self.close_price_df.columns
        )

        # calculating the three performance measures in accordance with their windows

        short_MA = self.close_price_df.pct_change(
            periods=self.Aperiods - 1, fill_method="pad", limit=None, freq=None
        )  # is counting window from 0
        long_MA = self.close_price_df.pct_change(
            periods=Bperiods - 1, fill_method="pad", limit=None, freq=None
        )  # is counting window from 0
        dfR = self.close_price_df.pct_change(
            periods=1, fill_method="pad", limit=None, freq=None
        )  # is counting window from 0

        columns = self.close_price_df.shape[1]
        for column in range(columns):
            ann_volatility[self.close_price_df.columns[column]] = (
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

        # Ranking each ETF w.r.t. short moving average of returns
        short_MA_ranks = self.rank_dataframe(short_MA)

        long_MA_ranks = self.rank_dataframe(long_MA)

        ann_volatility_ranks = self.rank_dataframe(ann_volatility)

        #################################################################################################################
        # Weights of the varous ranks ####################################################################################
        short_MA_ranks = short_MA_ranks.multiply(self.ShortTermWeight)  # .3 default
        long_MA_ranks = long_MA_ranks.multiply(self.LongTermWeight)  # .4 default
        ann_volatility_ranks = ann_volatility_ranks.multiply(
            self.ShortTermVolatilityWeight
        )  # .3 default
        dfAll_ranks = short_MA_ranks.add(long_MA_ranks, fill_value=0)
        dfAll_ranks = dfAll_ranks.add(ann_volatility_ranks, fill_value=0)
        #################################################################################################################
        #################################################################################################################

        # Choice is the dataframe where the ETF with the maximum score is identified
        dfChoice = self.close_price_df.copy(deep=True)
        dfChoice[:] = 0

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
                dfChoice.iat[
                    -1, self.close_price_df.columns.get_loc(self.StandsForCash)
                ] = 1
        else:
            dfChoice.iat[-1, max_arr_column] = 1

        combine_df = dfAll_ranks.join(
            dfChoice, lsuffix="_rank", rsuffix="_choice", on="Date", how="left"
        )

        final_df = combine_df[-1:]

        return final_df.head()
