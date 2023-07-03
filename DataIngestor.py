import pandas as pd

# pd.core.common.is_list_like = pd.api.types.is_list_like #datareader problem probably fixed in next version of datareader
from pandas_datareader import data as pdr
from datetime import datetime
from util import *
import yfinance as yf

yf.pdr_override()


class DataIngestor:
    def __init__(self, stock_list: list, start_date: datetime, end_date: datetime):
        self.stock_list = stock_list
        self.start_date = start_date
        self.end_date = end_date
        self.filename_list = []
        self.df_adj_close = pd.DataFrame()
        self.df_close = pd.DataFrame()

        for close_adj in [True, False]:
            self._df = pd.DataFrame()
            self.load_yahoo_data(close_adj)

    def load_yahoo_data(self, adjusted_close: bool):
        """
        Grab stock data from yahoo finance, filter to keep on close prices. Can be adjusted close price, or non-adjusted close

            Args:
                stock_list [list]: list of stock tickers (upper case) matching those listed on yahoo finance
                adjusted_close [bool]: if True, will take adjusted close prices. If False will take non-adjusted close price

            Returns:
        """

        if adjusted_close:
            close = "Adj Close"
            columns_to_drop = ["Close", "High", "Low", "Open", "Volume"]
        else:
            close = "Close"
            columns_to_drop = ["Adj Close", "High", "Low", "Open", "Volume"]

        for stock in range(len(self.stock_list)):
            temp_df = pdr.get_data_yahoo(
                self.stock_list[stock], start=self.start_date, end=self.end_date
            )
            temp_df.drop(columns_to_drop, axis=1, inplace=True)
            temp_df.rename(columns={close: self.stock_list[stock]}, inplace=True)
            print(temp_df.columns)
            if self._df.empty:
                self._df = temp_df
            else:
                self._df = self._df.join(temp_df, on="Date")

        self.write_to_csv(adjusted_close)

        if adjusted_close:
            self.df_adj_close = self._df.copy(deep=True)
        else:
            self.df_close = self._df.copy(deep=True)

    def write_to_csv(self, adjusted_close: bool):
        file_name = ".".join(self.stock_list)
        self.filename_list.append(file_name)
        if adjusted_close:
            file_name = file_name + "_AP.csv"
        else:
            file_name = file_name + ".csv"

        self._df.to_csv(file_name)

    def standardize_data(self, df: pd.DataFrame):
        for col in df.columns:
            df[col] = (df[col] - df[col].mean()) / df[col].std()

        return df
