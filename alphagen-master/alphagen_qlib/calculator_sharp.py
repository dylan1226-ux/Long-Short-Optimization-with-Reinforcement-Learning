from alphagen.utils.pytorch_utils import normalize_by_day
from alphagen_qlib.my_data import *
from alphagen.data.expression import *
from scipy.stats import norm


class SharpeCalculator:

    def __init__(self, data: StockData2, device=None):
        self.data = data
        self.device = device if device else torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        self.dummy_tensor = None
        self.ret_f1d_tensor = None
        self._initialize_tensors()

    def _initialize_tensors(self):
        """
        Initialize dummy and return data
        """
        #  dummy data
        dummy = pd.read_csv('/tmp/pycharm_project_877/company_data/processed_data/dummy_v2.csv').set_index('trade_date')
        dummy = dummy[sorted(dummy.columns, key=lambda x: int(str(x)))]
        dummy = dummy.loc[self.data.start_time:self.data.end_time] # filter to contain the correct date
        dummy_tensor = torch.tensor(dummy.values, dtype=torch.float32).to(self.device)

        # ret_f1d data
        ret_f1d = pd.read_csv('/tmp/pycharm_project_877/company_data/processed_data/ret_v2.csv').set_index('trade_date')
        ret_f1d = ret_f1d[sorted(ret_f1d.columns, key=lambda x: int(str(x)))]
        ret_f1d = ret_f1d.loc[self.data.start_time:self.data.end_time] # filter to contain the correct date
        ret_f1d_tensor = torch.tensor(ret_f1d.values, dtype=torch.float32).to(self.device)

        # Adjust tensors based on data_test parameters
        start = self.data.max_backtrack_days
        stop = self.data.max_backtrack_days + self.data.n_days
        self.dummy_tensor = dummy_tensor[start:stop, :]
        self.ret_f1d_tensor = ret_f1d_tensor[start:stop, :]

    def print_tensors(self):  # printing for checking
        print("Dummy Tensor:\n", self.dummy_tensor)
        print("Ret_f1d Tensor:\n", self.ret_f1d_tensor)

    def calc_alpha(self, expr: Expression) -> Tensor:
        """
        Compute alpha value for a given expression
        """
        alpha_value = normalize_by_day(expr.evaluate(self.data))
        return alpha_value

    def calc_ensemble_alpha(self, alpha_value:Optional[List[Tensor]], expr: Optional[List[Expression]],weights: List[float]) :
        """
        Compute alpha value for the alpha pool
        """
        if alpha_value is None:
            n = len(expr)
            factors : List[Tensor] = [self.calc_alpha(expr[i]) * weights[i] for i in range(n)]
        else: # if input is alpha value
            factors: List[Tensor] = [alpha_value[i]* weights[i] for i in range(len(alpha_value))]
        return sum(factors)


    def factor_score_sig_torch(self, invar: Tensor) -> Tensor:
        """
        Perform ranking
        """
        invar_rank = torch.argsort(torch.argsort(invar, dim=1, descending=False), dim=1).float() + 1
        count = (~torch.isnan(invar)).sum(dim=1, keepdim=True).float()
        invar_rank = (invar_rank - 3.0 / 8.0) / (count + 1.0 / 4.0)
        invar_score = torch.from_numpy(norm.ppf(invar_rank.cpu().numpy())).to(invar.device)
        invar_score[torch.isnan(invar)] = float('nan')
        return invar_score

    def factorToWeightTorch(self, invar_score: Tensor):
        """
        Calculate weight for long, short position given a ranking from factor_score_sig_torch()
        """
        weighT = invar_score.clone()
        weighT[weighT <= 0] = float('nan')
        weighT = weighT / torch.nansum(weighT, dim=1, keepdim=True)

        weighB = invar_score.clone()
        weighB[weighB >= 0] = float('nan')
        weighB = weighB / torch.nansum(weighB, dim=1, keepdim=True)

        weight = torch.nan_to_num(weighT, nan=0.0) - torch.nan_to_num(weighB, nan=0.0)
        weight[weight == 0] = float('nan')

        stockNum_T = (~torch.isnan(weighT)).sum(dim=1)
        stockNum_B = (~torch.isnan(weighB)).sum(dim=1)
        stockNum = torch.stack([stockNum_T, stockNum_B], dim=1)
        return weighT, weighB, weight, stockNum



    def factor_pnl_sharpe(self, precomputed_list=None, expr=None, expr_list=None, weights=None) -> float:
        """
        Compute Sharpe
        Three types of input :
        1. a list of expression AND a list of weight
        2. a expression
        3. a list of alpha values AND a list of weight
        """
        if expr_list is not None and weights is not None:
            alpha_value = self.calc_ensemble_alpha(None, expr_list, weights) # computation for pool
        elif expr is not None:
            alpha_value = self.calc_alpha(expr) # single calculation
        elif precomputed_list is not None and weights is not None:
            factor = [precomputed_list[i] * weights[i] for i in range(len(precomputed_list))]
            alpha_value = sum(factor)
        else:
            raise ValueError("Check input")

        alpha_value = alpha_value * self.dummy_tensor # mask non-tradable stock
        alpha_value = torch.where(torch.isnan(alpha_value), torch.tensor(float('nan'), device=self.device), alpha_value)

        invar_score = self.factor_score_sig_torch(alpha_value) #compute factor scores
        invar_score = invar_score * self.dummy_tensor
        _, _, weight, _ = self.factorToWeightTorch(invar_score)

        ret_f1d_tensor = torch.nan_to_num(self.ret_f1d_tensor, nan=0.0) * torch.nan_to_num(self.dummy_tensor, nan=0.0)
        ret_f1d_tensor[ret_f1d_tensor == 0] = float('nan')
        ret_f1d_tensor = torch.exp(ret_f1d_tensor) - 1
        ret_mkt_fnd = torch.nanmean(ret_f1d_tensor, dim=1).unsqueeze(1)
        er_fnd = torch.where(torch.isnan(ret_f1d_tensor), torch.tensor(float('nan'), device=ret_f1d_tensor.device),
                             ret_f1d_tensor - ret_mkt_fnd)

        weighted = er_fnd * weight
        weighted = torch.where(torch.isnan(weighted), torch.tensor(float('nan'), device=weighted.device), weighted)
        total = torch.nan_to_num(weighted, nan=0.0).sum(dim=1).unsqueeze(dim=1)
        outpnl = torch.log(total + 1)
        sharpe = outpnl.mean() * 252 / (outpnl.std() * torch.sqrt(torch.tensor(252.0, device=outpnl.device)))
        return sharpe.item()



if __name__ == '__main__':
    csv_path_file = '/tmp/pycharm_project_877/company_data/processed_data/in_sample_18_22.joblib'
    data_test = StockData2(csv_path=csv_path_file,
                            start_time='2020-01-02',
                            end_time='2022-12-31')

    sharpe_calculator = SharpeCalculator(data_test)

    expression = TS_Ref(Feature(FeatureType.buy_trades_act), -20)
    sharpe = sharpe_calculator.factor_pnl_sharpe(expr=expression)
    print(sharpe)

    expression2 = Add(Feature(FeatureType.close_moneyflow_pct_value),Constant(5))
    sharpe2 = sharpe_calculator.factor_pnl_sharpe(expr_list=[expression, expression2],weights=[1.0,1.0])
    print(sharpe2)
