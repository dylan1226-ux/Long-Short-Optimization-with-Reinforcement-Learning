from typing import List, Union, Optional, Tuple, Dict
from enum import IntEnum
import pandas as pd
import torch
from joblib import load
from numpy import dtype



class FeatureType(IntEnum):
    buy_trades_act = 0
    close_moneyflow_pct_value = 1
    close_moneyflow_pct_volume = 2
    close_net_inflow_rate_value = 3
    close_net_inflow_rate_volume = 4
    cancell_buy_order_value = 5
    large_trades_return = 6
    large_buy_value = 7
    large_sell_value = 8
    factor_bolldown20d = 9
    factor_obv1d = 10
    factor_pvt1d = 11
    factor_willr14d = 12
    factor_adxr14d = 13
    factor_apbma5d = 14
    factor_bias10d = 15
    factor_cci5d = 16
    factor_ema5d = 17
    factor_adtm = 18
    factor_stom = 19
    factor_dmi5d = 20
    factor_variance20d = 21
    factor_atr14d = 22
    factor_roc6d = 23
    factor_kdjk9d = 24

class StockData2:
    def __init__(self,
                 csv_path: str,
                 start_time: str,
                 end_time: str,
                 max_backtrack_days: int = 10,
                 max_future_days: int = 30,
                 out_sample = False,
                 features: Optional[List[FeatureType]] = None,
                 device: torch.device = torch.device('cuda:1')) -> None:


        self.date_column = "trade_date"
        self.stock_id_column = "code"

        self.csv_path = csv_path
        self.max_backtrack_days = max_backtrack_days
        self.max_future_days = max_future_days
        self._start_time = pd.to_datetime(start_time)
        self._end_time = pd.to_datetime(end_time)
        self._features = features if features is not None else list(FeatureType)
        self.if_out_sample = out_sample
        self.device = device
        self.data, self._dates, self._stock_ids = self._get_data()



    def _get_data(self) -> Tuple[torch.tensor, pd.Index, pd.Index]:
        """
        Handles two types of input:
        1. Normal Dataframe where (trade_date,code,features)
        2. Multi-column Dataframe (1 level: feather; 2 level: code; index: trade_date)
        """
        if self.if_out_sample:
            # process data
            df = load(self.csv_path)
            df[self.date_column] = pd.to_datetime(df[self.date_column], errors='coerce')
            # select days we are interested in
            df = df[(df[self.date_column] >= self._start_time) & (df[self.date_column] <= self._end_time)]
            # feature name [$close,$high....]
            feature_columns = [f.name for f in FeatureType]
            # Create Pivot (Row: date ;Column: stock id)
            df = df.pivot(index=self.date_column, columns=self.stock_id_column, values=feature_columns)
            # reindex as pivot creates alphabetically!!!!
            df = df.reindex(columns=feature_columns, level=0)
        else:
            # a direct process multi column
            feature_columns = [f.name for f in FeatureType]
            df = load(self.csv_path)
            df = df.loc[self._start_time:self._end_time]

        dates = df.index.get_level_values(self.date_column).unique()
        stock_ids = df.columns.get_level_values(self.stock_id_column).unique()
        values = df.values
        # 3-d tensor (days,features,stocks)
        num_days = len(dates)
        num_features = len(feature_columns)
        num_stocks = len(stock_ids)
        values = values.reshape((num_days, num_features,num_stocks))
        return torch.tensor(values, dtype=torch.float, device=self.device), dates, stock_ids


    @property
    def n_features(self) -> int:
        return len(self._features)

    @property
    def n_stocks(self) -> int:
        return self.data.shape[-1]

    @property
    def n_days(self) -> int:
        return self.data.shape[0] - self.max_backtrack_days - self.max_future_days

    @property
    def start_time(self):
        return str(self._start_time.date())

    @property
    def end_time(self):
        return str(self._end_time.date())

