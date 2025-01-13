import numpy as np
from itertools import count
from alphagen_qlib.calculator_sharp import SharpeCalculator
from alphagen.data.expression import *
from typing import List, Optional, Tuple
from abc import ABCMeta, abstractmethod

class SharpePoolBase(metaclass=ABCMeta): # calculator base class
    def __init__(
        self,
        capacity: int,
        calculator: SharpeCalculator,
        device: torch.device = torch.device('cuda:1')
    ):
        self.capacity = capacity
        self.calculator = calculator
        self.device = device

    @abstractmethod
    def try_new_expr(self, expr: Expression) -> float: ...


class AlphaPoolSharpe(SharpePoolBase):
    def __init__(
        self,
        capacity: int,
        calculator: SharpeCalculator,
        l1_alpha: float = 5e-3,
        device: torch.device = torch.device('cuda:1'),
    ):
        super().__init__(capacity, calculator,device)

        self.size : int = 0
        self.exprs:  List[Optional[Expression]] = [None for _ in range(capacity + 1)]
        self.single_sharps: np.ndarray = np.zeros(capacity +1)
        self.weights: np.ndarray = np.zeros(capacity + 1)
        self.best_sharpe_ret: float = -1
        self.l1_alpha = l1_alpha
        self.eval_cnt = 0


    def try_new_expr(self, expr: Expression) -> float:
        """
        Input:  a new expression from AlphaGenerator
        'If a expression is valid, AlphaGenerator passes the expression to here;
        Output: a maximized sharpe
        'this output is used as the reward signal of the AlphaGenerator'
        """
        sharpe = self._calc_sharpe(expr)

        # add factor to the pool
        self._add_factor(expr,sharpe)

        if self.size >1: # starts optimizing when the pool size is greater than 1
            new_weights = self._optimize(alpha=self.l1_alpha, lr=5e-4,n_iter=100) # adjust n_iter here
            worst_idx = np.argmin(np.abs(new_weights)) # worst weight index
            if worst_idx != self.capacity:
                self.weights[:self.size] = new_weights # assign new weights
            self._pop()
        new_sharpe_ret = self._calc_sharpe(expr_list=self.exprs[:self.size],weights=self.weights[:self.size]) # only effective alpha!
        increment = new_sharpe_ret - self.best_sharpe_ret
        if increment > 0 :
            self.best_sharpe_ret = new_sharpe_ret
        self.eval_cnt +=1
        return new_sharpe_ret

    def _calc_sharpe(self, expr: Expression = None, expr_list: List[Expression] = None,
                     weights: List[float] = None, precomputed_list=None) -> float:
        """
        Three types of input :
        1. a list of expression AND a list of weight
        2. a expression
        3. a list of alpha values AND a list of weight
        Output: a sharpe calculated from the SharpeCalculator Class
        """
        if expr_list is not None and weights is not None:
            sharpe = self.calculator.factor_pnl_sharpe(expr_list=expr_list, weights=weights)
        elif expr is not None:
            sharpe = self.calculator.factor_pnl_sharpe(expr=expr)
        elif precomputed_list is not None and weights is not None:
            sharpe = self.calculator.factor_pnl_sharpe(precomputed_list=precomputed_list, weights=weights)
        else:
            raise ValueError("Check Input")

        return sharpe

    def _add_factor(self,
                    expr: Expression,
                    sharpe: float):
        """
        Add a factor to the pool
        """
        n = self.size # current size
        self.exprs[n] = expr # store expression
        self.single_sharps[n] = sharpe # store sharpe
        self.weights[n] = sharpe * 0.01 # an arbitrary initial weight
        self.size += 1

    def _optimize(self, alpha:float, lr:float, n_iter: int) -> np.ndarray:
        """
        optimize weights with gradient descent
        Loss: -sharpe -> minimizing -sharpe maximizes sharpe
        """
        # initialize sharpe and weights (gradient enabled)
        weights = torch.from_numpy(self.weights[:self.size]).to(self.device).requires_grad_()
        # initialize alpha value as a list of Tensors
        precomputed_list = [self.calculator.calc_alpha(expr) for expr in self.exprs[:self.size]]
        # Adam optimizer
        optim = torch.optim.Adam([weights], lr=lr)
        loss_sharpe_max = float('-inf') # An arbitrary small value
        best_weights = weights.cpu().detach().numpy()
        iter_count = 0
        for it in count():
            # calculate sharpe
            sharpe = self._calc_sharpe(precomputed_list=precomputed_list,weights=weights)
            # define loss with L1 regularization
            loss_sharpe = -sharpe
            loss_l1 = torch.norm(weights,p=1)
            loss = loss_sharpe + alpha * loss_l1

            if sharpe > loss_sharpe_max: # check improvement and update
                # print("calculated sharpe is highest now",sharpe)
                best_weights = weights.cpu().detach().numpy()
                loss_sharpe_max = sharpe

            optim.zero_grad()
            loss.backward()
            optim.step()

            # traceback and early stopping
            if sharpe > loss_sharpe_max + 0.5: # significant improvement in Sharpe
                iter_count = 0 # reset count
            else:
                iter_count += 1

            if iter_count >= n_iter or it >= 2000: # stop conditions
                break
        return best_weights

    def _pop(self) -> None:
        if self.size <= self.capacity:
            return
        idx = np.argmin(np.abs(self.weights))
        self._swap_idx(idx,self.capacity)
        self.size = self.capacity

    def _swap_idx(self, i,j) -> None:
        if i == j:
            return # no need to swap
        self.exprs[i], self.exprs[j] = self.exprs[j], self.exprs[i]
        self.weights[i], self.weights[j] = self.weights[j], self.weights[i]

    def to_dict(self) -> dict:
        return {
            "exprs": [str(expr) for expr in self.exprs[:self.size]],
            "weights": list(self.weights[:self.size]) }

    @property
    def state(self) -> dict:
        return {
            "exprs": list(self.exprs[:self.size]),
            "sharpes_ret": list(self.single_sharps[:self.size]),
            "weights": list(self.weights[:self.size]),
            "best_sharpe_ret": self.best_sharpe_ret}

    def test_ensemble(self,calculator:SharpeCalculator):
        sharpe = calculator.factor_pnl_sharpe(expr_list=self.exprs[:self.size],weights=self.weights[:self.size])
        return sharpe

if __name__ == '__main__':

    data_test = StockData2(csv_path='/tmp/pycharm_project_877/company_data/processed_data/in_sample_18_22.joblib',
                            start_time='2018-01-02',
                            end_time='2022-12-30')

    exp1 = TS_Ref(Feature(FeatureType.factor_adtm), -20)
    exp2 = Sub(Feature(FeatureType.large_sell_value),Feature(FeatureType.close_moneyflow_pct_value))
    exp3 = S_LOG2_LP(Feature(FeatureType.factor_atr14d))
    exp4 = FloorDiv(Log2(Feature(FeatureType.factor_variance20d)),Feature(FeatureType.factor_bias10d))

    sharpe_calculator = SharpeCalculator(data_test)
    sharpe_pool = AlphaPoolSharpe(capacity=2,calculator=sharpe_calculator,device=torch.device('cuda:1'))


    sharpe_pool.try_new_expr(expr=exp1)
    print(sharpe_pool.state)
    sharpe_pool.try_new_expr(expr=exp2)
    print(sharpe_pool.state)
    sharpe_pool.try_new_expr(expr=exp3)
    print(sharpe_pool.state)
    sharpe_pool.try_new_expr(expr=exp4)
    print(sharpe_pool.state)

    print("done!!!!!!")