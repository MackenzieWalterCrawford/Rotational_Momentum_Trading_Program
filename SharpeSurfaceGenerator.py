import pandas as pd
import numpy as np
import math
import datetime
import matplotlib.pyplot as plt
from matplotlib import style
import detrendPrice
import WhiteRealityCheckFor1

# import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import Axes3D


class SharpeSurfaceGenerator:
    def __init__(
        self,
        close_price_df: pd.DataFrame,
        adj_close_df: pd.DataFrame,
    ):
        self.logReturns = 0
        self.momentum = 1
        self.volmomentum = 0  # do not change
        self.CashFilter = 0
        self.MAperiods = 200  # for the cash filter
        self.Delay = 1
        self.ShortTermWeight = 0.3
        self.LongTermWeight = 0.4
        self.ShortTermVolatilityWeight = 0.4
        self.StandsForCash = "SHY"
        self.max_holding = 20
        self.step_holding = 1
        self.max_lookback = 12
        self.step_lookback = 5

        self.close_price_df = close_price_df.sort_values(by="Date").set_index("Date")
        self.adj_close_df = adj_close_df.sort_values(by="Date").set_index("Date")
        self.sharpe_surface_df = pd.DataFrame(columns=["lookback", "holding", "sharpe"])
        self.sharpe_matrix = np.zeros(shape=(self.max_lookback, self.max_holding))

    def getDate(self, dt):
        if type(dt) != str:
            return dt
        try:
            datetime_object = datetime.datetime.strptime(dt, "%Y-%m-%d")
        except Exception:
            datetime_object = datetime.datetime.strptime(dt, "%m/%d/%Y")
            return datetime_object
        else:
            return datetime_object

    def trade(self, lookback_period, hold_period):
        Aperiods = lookback_period  # short period
        Bperiods = 3 * Aperiods + ((3 * Aperiods) // 20) * 2  # long period
        Frequency = hold_period
        freq = Frequency + "_FREQ"

        std_dev_df = self.close_price_df.drop(columns=self.close_price_df.columns)
        mov_avg_df = self.close_price_df.drop(columns=self.close_price_df.columns)
        detrend_df = self.close_price_df.drop(columns=self.close_price_df.columns)

        short_returns_df = self.close_price_df.pct_change(
            periods=Aperiods - 1, fill_method="pad", limit=None, freq=None
        )  # is counting window from 0
        long_returns_df = self.close_price_df.pct_change(
            periods=Bperiods - 1, fill_method="pad", limit=None, freq=None
        )  # is counting window from 0
        returns_df = self.close_price_df.pct_change(
            periods=1, fill_method="pad", limit=None, freq=None
        )  # is counting window from 0

        # columns = self.close_price_df.shape[1]
        # for column in range(columns):
        for col in self.close_price_df.columns:
            std_dev_df[col] = (
                returns_df[col].rolling(window=Aperiods).std()
            ) * math.sqrt(252)
            mov_avg_df[col] = (
                self.close_price_df[col].rolling(window=self.MAperiods).mean()
            )
            detrend_df[col] = detrendPrice.detrendPrice(self.adj_close_df[col]).values

        short_ranks_df = self.close_price_df.copy(deep=True)
        short_ranks_df[:] = 0

        columns = short_ranks_df.shape[1]
        rows = short_ranks_df.shape[0]

        for row in range(rows):
            arr_row = short_returns_df.iloc[row].values
            if self.momentum == 1:
                temp = (
                    arr_row.argsort()
                )  # sort momentum, best is ETF with largest return
            else:
                temp = (-arr_row).argsort()[
                    : arr_row.size
                ]  # sort reversion, best is ETF with lowest return
            ranks = np.empty_like(temp)  # empty array
            ranks[temp] = np.arange(1, len(arr_row) + 1)
            for column in range(columns):
                short_ranks_df.iat[row, column] = ranks[column]

        long_ranks_df = self.close_price_df.copy(deep=True)
        long_ranks_df[:] = 0

        columns = long_ranks_df.shape[1]
        rows = long_ranks_df.shape[0]

        for row in range(rows):
            arr_row = long_returns_df.iloc[row].values
            if self.momentum == 1:
                temp = arr_row.argsort()  # sort momentum
            else:
                temp = (-arr_row).argsort()[: arr_row.size]  # sort reversion
            ranks = np.empty_like(temp)
            ranks[temp] = np.arange(1, len(arr_row) + 1)
            for column in range(columns):
                long_ranks_df.iat[row, column] = ranks[column]

        dfS_ranks = self.close_price_df.copy(deep=True)
        dfS_ranks[:] = 0

        columns = dfS_ranks.shape[1]
        rows = dfS_ranks.shape[0]

        for row in range(rows):
            arr_row = std_dev_df.iloc[row].values
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
        short_ranks_df = short_ranks_df.multiply(self.ShortTermWeight)  # .3 default
        long_ranks_df = long_ranks_df.multiply(self.LongTermWeight)  # .4 default
        dfS_ranks = dfS_ranks.multiply(self.ShortTermVolatilityWeight)  # .3 default
        dfAll_ranks = short_ranks_df.add(long_ranks_df, fill_value=0)
        dfAll_ranks = dfAll_ranks.add(dfS_ranks, fill_value=0)
        #################################################################################################################
        #################################################################################################################

        dfChoice = self.close_price_df.copy(deep=True)
        dfChoice[:] = 0
        rows = dfChoice.shape[0]

        for row in range(rows):
            arr_row = dfAll_ranks.iloc[row].values
            if self.momentum == 0 or self.CashFilter == 1:
                arr_row = arr_row[
                    0 : len(arr_row) - 1
                ]  # don't rank SHY (the last column) if doing reversion to mean trading
            max_arr_column = np.argmax(arr_row, axis=0)  # gets the INDEX of the max

            if self.CashFilter == 1:
                if (
                    self.close_price_df[self.close_price_df.columns[max_arr_column]][
                        row
                    ]
                    >= mov_avg_df[mov_avg_df.columns[max_arr_column]][row]
                ):  # 200MA condition for cash filter
                    dfChoice.iat[row, max_arr_column] = 1
                else:
                    dfChoice.iat[
                        row, self.close_price_df.columns.get_loc(self.StandsForCash)
                    ] = 1
            else:
                dfChoice.iat[row, max_arr_column] = 1

        if self.logReturns == 1:
            dfPLog = self.adj_close_df.apply(np.log)
            dfPLogShift = dfPLog.shift(1)
            results_df = dfPLog.subtract(dfPLogShift, fill_value=0)
            # repeat with detrended prices
            dfPLog = detrend_df.apply(np.log)
            dfPLogShift = dfPLog.shift(1)
            dfDetrendRR = dfPLog.subtract(dfPLogShift, fill_value=0)

        else:
            results_df = self.adj_close_df.pct_change()
            dfDetrendRR = detrend_df.pct_change()

        # T is the dataframe where the trading day is calculated.

        dfT = self.close_price_df.drop(
            labels=None, axis=1, columns=self.close_price_df.columns
        )

        for col in self.close_price_df.columns:
            new = col + "_CHOICE"
            # results_df[new] = pd.Series(np.zeros(rows), index=results_df.index)
            results_df[new] = dfChoice[col]

        dfT["DateCopy"] = dfT.index
        dfT1 = dfT.asfreq(freq=Frequency, method="pad").set_index("DateCopy")
        dfTJoin = pd.merge(
            dfT, dfT1, left_index=True, right_index=True, how="outer", indicator=True
        )
        results_df[freq] = dfTJoin["_merge"]

        # _LEN means Long entry for that ETF
        # _NUL means number units long of that ETF
        # _LEX means long exit for that ETF
        # _R means returns of that ETF (traded ETF)
        # _ALL_R means returns of all ETFs traded, i.e. portfolio returns
        # CUM_R means commulative returns of all ETFs, i.e. portfolio cummulative returns

        # columns = self.close_price_df.shape[1]
        # for column in range(columns):
        for col in self.close_price_df.columns:
            """
            new_column = pd.DataFrame({column + "_LEN":   } ) column + "_LEN"
            """
            long_enter = col + "_LEN"
            long_exit = col + "_LEX"
            num_long = col + "_NUL"
            returns = col + "_R"

            results_df[long_enter] = (results_df[freq] == "both") & (
                results_df[col + "_CHOICE"] == 1
            )

            results_df[long_exit] = (results_df[freq] == "both") & (
                results_df[col + "_CHOICE"] != 1
            )

            results_df[num_long] = np.nan
            results_df.loc[results_df[long_exit] == True, num_long] = 0
            results_df.loc[
                results_df[long_enter] == True, num_long
            ] = 1  # this order is important
            results_df.iat[0, results_df.columns.get_loc(num_long)] = 0
            results_df[num_long] = results_df[num_long].fillna(method="pad")

            results_df[returns] = results_df[col] * results_df[num_long].shift(
                self.Delay
            )

            dfDetrendRR[returns] = dfDetrendRR[col] * results_df[num_long].shift(
                self.Delay
            )

        results_df = results_df.assign(ALL_R=pd.Series(np.zeros(rows)).values)

        dfDetrendRR = dfDetrendRR.assign(ALL_R=pd.Series(np.zeros(rows)).values)

        # columns = self.close_price_df.shape[1]
        for col in self.close_price_df.columns:
            results_df["ALL_R"] = results_df["ALL_R"] + results_df[col + "_R"]
            # repeat for detrended returns
            dfDetrendRR["ALL_R"] = dfDetrendRR["ALL_R"] + dfDetrendRR[col + "_R"]

        results_df = results_df.assign(DETREND_ALL_R=dfDetrendRR["ALL_R"])

        results_df = results_df.assign(
            I=np.cumprod(1 + results_df["ALL_R"])
        )  # this is good for pct return or log return
        results_df.iat[0, results_df.columns.get_loc("I")] = 1
        # repeat for detrended returns
        dfDetrendRR = dfDetrendRR.assign(
            I=np.cumprod(1 + dfDetrendRR["ALL_R"])
        )  # this is good for pct return or log return
        dfDetrendRR.iat[0, dfDetrendRR.columns.get_loc("I")] = 1

        results_df = results_df.assign(DETREND_I=dfDetrendRR["I"])

        try:
            sharpe = (
                results_df["ALL_R"].mean() / results_df["ALL_R"].std()
            ) * math.sqrt(252)
        except ZeroDivisionError:
            sharpe = 0.0

        style.use("fivethirtyeight")
        results_df["I"].plot()
        plt.legend()
        plt.show()

        start = 1
        start_val = start
        end_val = results_df["I"].iat[-1]

        start_date = self.getDate(results_df.iloc[0].name)
        end_date = self.getDate(results_df.iloc[-1].name)
        days = (end_date - start_date).days

        TotaAnnReturn = (end_val - start_val) / start_val / (days / 360)
        # TotaAnnReturn_trading = (end_val - start_val) / start_val / (days / 252)

        # CAGR_trading = round(
        #     ((float(end_val) / float(start_val)) ** (1 / (days / 252.0))).real - 1, 4
        # )  # when raised to an exponent I am getting a complex number, I need only the real part
        CAGR = round(
            ((float(end_val) / float(start_val)) ** (1 / (days / 350.0))).real - 1, 4
        )  # when raised to an exponent I am getting a complex number, I need only the real part

        print("TotaAnnReturn = %f" % (TotaAnnReturn * 100))
        print("CAGR = %f" % (CAGR * 100))
        print("Sharpe Ratio = %f" % (round(sharpe, 2)))

        # Detrending Prices and Returns
        WhiteRealityCheckFor1.bootstrap(results_df["DETREND_ALL_R"])

        results_df.to_csv(
            r"Results/results_df.csv", header=True, index=True, encoding="utf-8"
        )

        return sharpe

    def generate_sharpe_matrix(self):
        i = 1
        for holding in range(
            2, self.max_holding, 1
        ):  # the step in range has to be 1 because I do not want empty rows or columns in MyMatrix
            hold_period = (
                str(holding * self.step_holding) + "W-FRI"
            )  # trading on fridays, day of the week matters.
            for lookback in range(
                2, self.max_lookback, 1
            ):  # the step in range has to be 1 because I do not want empty rows or columns in MyMatrix
                lookback_period = lookback * self.step_lookback
                self.sharpe_matrix[lookback, holding] = self.trade(
                    lookback_period, hold_period
                )
                sharpe = self.trade(lookback_period, hold_period)
                self.sharpe_surface_df.loc[i] = [lookback, holding, sharpe]
                i += 1

    def plot_sharpe_surface(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.view_init(90, 0)

        # want lookback to show as rows (y axis) and holding as columns (x axis) like in MyMatrix
        surf = ax.plot_trisurf(
            self.sharpe_surface_df["holding"],
            self.sharpe_surface_df["lookback"],
            self.sharpe_surface_df["sharpe"],
            cmap=plt.cm.viridis,
            linewidth=0.1,
        )
        fig.colorbar(surf, shrink=0.5, aspect=5)

        # The fix  for tight layout
        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.set_xlabel("Holding")
        ax.set_ylabel("Lookback")
        ax.set_zlabel("Sharpe Ratio")

        plt.tight_layout()

        plt.show()

        for angle in range(0, 360, 60):
            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")
            ax.view_init(60, angle)
            surf = ax.plot_trisurf(
                self.sharpe_surface_df["holding"],
                self.sharpe_surface_df["lookback"],
                self.sharpe_surface_df["sharpe"],
                cmap=plt.cm.viridis,
                linewidth=0.1,
            )
            fig.colorbar(surf, shrink=0.5, aspect=5)
            ax.set_xlabel("Holding")
            ax.set_ylabel("Lookback")
            ax.set_zlabel("Sharpe Ratio")
            plt.savefig(r"Results/%s.png" % (angle))
            # plt.draw()

        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.view_init(90, 0)
        surf = ax.plot_trisurf(
            self.sharpe_surface_df["holding"],
            self.sharpe_surface_df["lookback"],
            self.sharpe_surface_df["sharpe"],
            cmap=plt.cm.viridis,
            linewidth=0.1,
        )
        fig.colorbar(surf, shrink=0.5, aspect=5)
        ax.set_xlabel("Holding")
        ax.set_ylabel("Lookback")
        ax.set_zlabel("Sharpe Ratio")
        plt.savefig(r"Results/%s.png" % ("From above"))
