from typing import List, Optional, Tuple
from torch import Tensor
import torch
from alphagen.data.calculator import AlphaCalculator
from alphagen.data.expression import Expression
from alphagen.utils.correlation import batch_pearsonr, batch_spearmanr
from alphagen.utils.pytorch_utils import normalize_by_day
from alphagen_qlib.my_data import StockData2


class QLibStockDataCalculator(AlphaCalculator):
    def __init__(self, data: StockData2, target: Optional[Expression]):
        self.data = data

        if target is None: # Combination-only mode
            self.target_value = None
        else:
            self.target_value = normalize_by_day(target.evaluate(self.data))

    def _calc_alpha(self, expr: Expression) -> Tensor:
        print('\ncalculating alpha:',expr)
        result = normalize_by_day(expr.evaluate(self.data))
        print(result)
        return result

    def _calc_IC(self, value1: Tensor, value2: Tensor) -> float:
        print('\ncalculating IC')
        result = batch_pearsonr(value1, value2).mean().item()
        print(result)
        return result

    def _calc_rIC(self, value1: Tensor, value2: Tensor) -> float:
        print('\ncalculating RIC')
        print(batch_spearmanr(value1, value2).mean().item())
        return batch_spearmanr(value1, value2).mean().item()

    def make_ensemble_alpha(self, exprs: List[Expression], weights: List[float]) -> Tensor:
        print('\nmaking ensemble alpha')
        n = len(exprs)
        factors: List[Tensor] = [self._calc_alpha(exprs[i]) * weights[i] for i in range(n)]
        return sum(factors)  # type: ignore

    def calc_single_IC_ret(self, expr: Expression) -> float:
        print('\ncalculating single IC')
        value = self._calc_alpha(expr)
        result = self._calc_IC(value, self.target_value)
        print(result)
        return result

    def calc_single_rIC_ret(self, expr: Expression) -> float:
        print('\ncalculating single RIC')
        value = self._calc_alpha(expr)
        result = self._calc_rIC(value, self.target_value)
        print(result)
        return result

    def calc_single_all_ret(self, expr: Expression) -> Tuple[float, float]:
        print('\ncalculating single RET')
        value = self._calc_alpha(expr)
        ic = self._calc_IC(value, self.target_value)
        ric = self._calc_rIC(value, self.target_value)
        return ic, ric

    def calc_mutual_IC(self, expr1: Expression, expr2: Expression) -> float:
        print('\ncalculating mutual IC')
        value1, value2 = self._calc_alpha(expr1), self._calc_alpha(expr2)
        result = self._calc_IC(value1, value2)
        print(result)
        return result


    def calc_pool_IC_ret(self, exprs: List[Expression], weights: List[float]) -> float:
        print('\ncalculating pool IC')
        with torch.no_grad():
            ensemble_value = self.make_ensemble_alpha(exprs, weights)
            result = self._calc_IC(ensemble_value, self.target_value)
            print(result)
            return result

    def calc_pool_rIC_ret(self, exprs: List[Expression], weights: List[float]) -> float:
        print('\ncalculating pool RIC')
        with torch.no_grad():
            ensemble_value = self.make_ensemble_alpha(exprs, weights)
            result = self._calc_rIC(ensemble_value, self.target_value)
            print(result)
            return result

    def calc_pool_all_ret(self, exprs: List[Expression], weights: List[float]) -> Tuple[float, float]:
        print('\ncalculating pool RET')
        with torch.no_grad():
            ensemble_value = self.make_ensemble_alpha(exprs, weights)
            target_value = self.target_value
            ic = self._calc_IC(ensemble_value, target_value)
            ric = self._calc_rIC(ensemble_value,target_value)
            return ic, ric
