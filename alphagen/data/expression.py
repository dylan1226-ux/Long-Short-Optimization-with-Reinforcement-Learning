from abc import ABCMeta, abstractmethod
from typing import List, Type, Union
from xmlrpc.client import Binary

import torch
from sympy import Inverse
from torch import Tensor

from alphagen_qlib.my_data import StockData2, FeatureType


class OutOfDataRangeError(IndexError):
    pass


class Expression(metaclass=ABCMeta):
    @abstractmethod
    def evaluate(self, data: StockData2, period: slice = slice(0, 1)) -> Tensor: ...

    def __repr__(self) -> str: return str(self)

    def __add__(self, other: Union["Expression", float]) -> "Add":
        if isinstance(other, Expression):
            return Add(self, other)
        else:
            return Add(self, Constant(other))

    def __radd__(self, other: float) -> "Add": return Add(Constant(other), self)

    def __sub__(self, other: Union["Expression", float]) -> "Sub":
        if isinstance(other, Expression):
            return Sub(self, other)
        else:
            return Sub(self, Constant(other))

    def __rsub__(self, other: float) -> "Sub": return Sub(Constant(other), self)

    def __mul__(self, other: Union["Expression", float]) -> "Mul":
        if isinstance(other, Expression):
            return Mul(self, other)
        else:
            return Mul(self, Constant(other))

    def __rmul__(self, other: float) -> "Mul": return Mul(Constant(other), self)

    def __truediv__(self, other: Union["Expression", float]) -> "Div":
        if isinstance(other, Expression):
            return Div(self, other)
        else:
            return Div(self, Constant(other))

    def __rtruediv__(self, other: float) -> "Div": return Div(Constant(other), self)

    def __pow__(self, other: Union["Expression", float]) -> "Pow":
        if isinstance(other, Expression):
            return Pow(self, other)
        else:
            return Pow(self, Constant(other))

    def __rpow__(self, other: float) -> "Pow": return Pow(Constant(other), self)

    def __pos__(self) -> "Expression": return self
    def __neg__(self) -> "Sub": return Sub(Constant(0), self)
    def __abs__(self) -> "Abs": return Abs(self)

    @property
    def is_featured(self): raise NotImplementedError


class Feature(Expression):
    def __init__(self, feature: FeatureType) -> None:
        self._feature = feature

    def evaluate(self, data: StockData2, period: slice = slice(0, 1)) -> Tensor:
        assert period.step == 1 or period.step is None
        if (period.start < -data.max_backtrack_days or
                period.stop - 1 > data.max_future_days):
            raise OutOfDataRangeError()
        start = period.start + data.max_backtrack_days
        stop = period.stop + data.max_backtrack_days + data.n_days - 1
        return data.data[start:stop, int(self._feature), :]


    def __str__(self) -> str: return '$' + self._feature.name.lower()

    @property
    def is_featured(self): return True


class Constant(Expression):
    def __init__(self, value: float) -> None:
        self._value = value

    def evaluate(self, data: StockData2, period: slice = slice(0, 1)) -> Tensor:
        assert period.step == 1 or period.step is None
        if (period.start < -data.max_backtrack_days or
                period.stop - 1 > data.max_future_days):
            raise OutOfDataRangeError()
        device = data.data.device
        dtype = data.data.dtype
        days = period.stop - period.start - 1 + data.n_days
        return torch.full(size=(days, data.n_stocks),
                          fill_value=self._value, dtype=dtype, device=device)

    def __str__(self) -> str: return f'Constant({str(self._value)})'

    @property
    def is_featured(self): return False


class DeltaTime(Expression):
    # This is not something that should be in the final expression
    # It is only here for simplicity in the implementation of the tree builder
    def __init__(self, delta_time: int) -> None:
        self._delta_time = delta_time

    def evaluate(self, data: StockData2, period: slice = slice(0, 1)) -> Tensor:
        assert False, "Should not call evaluate on delta time"

    def __str__(self) -> str: return str(self._delta_time)

    @property
    def is_featured(self): return False


# Operator base classes

class Operator(Expression):
    @classmethod
    @abstractmethod
    def n_args(cls) -> int: ...

    @classmethod
    @abstractmethod
    def category_type(cls) -> Type['Operator']: ...


class UnaryOperator(Operator):
    def __init__(self, operand: Union[Expression, float]) -> None:
        self._operand = operand if isinstance(operand, Expression) else Constant(operand)

    @classmethod
    def n_args(cls) -> int: return 1

    @classmethod
    def category_type(cls) -> Type['Operator']: return UnaryOperator

    def evaluate(self, data: StockData2, period: slice = slice(0, 1)) -> Tensor:
        return self._apply(self._operand.evaluate(data, period))

    @abstractmethod
    def _apply(self, operand: Tensor) -> Tensor: ...

    def __str__(self) -> str:
        return f"{type(self).__name__}({self._operand})"

    @property
    def is_featured(self): return self._operand.is_featured


class BinaryOperator(Operator):
    def __init__(self, lhs: Union[Expression, float], rhs: Union[Expression, float]) -> None:
        self._lhs = lhs if isinstance(lhs, Expression) else Constant(lhs)
        self._rhs = rhs if isinstance(rhs, Expression) else Constant(rhs)

    @classmethod
    def n_args(cls) -> int: return 2

    @classmethod
    def category_type(cls) -> Type['Operator']: return BinaryOperator

    def evaluate(self, data: StockData2, period: slice = slice(0, 1)) -> Tensor:
        return self._apply(self._lhs.evaluate(data, period), self._rhs.evaluate(data, period))

    @abstractmethod
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: ...

    def __str__(self) -> str:
        return f"{type(self).__name__}({self._lhs},{self._rhs})"

    @property
    def is_featured(self): return self._lhs.is_featured or self._rhs.is_featured


class RollingOperator(Operator):
    def __init__(self, operand: Union[Expression, float], delta_time: Union[int, DeltaTime]) -> None:
        self._operand = operand if isinstance(operand, Expression) else Constant(operand)
        if isinstance(delta_time, DeltaTime):
            delta_time = delta_time._delta_time
        self._delta_time = delta_time

    @classmethod
    def n_args(cls) -> int: return 2

    @classmethod
    def category_type(cls) -> Type['Operator']: return RollingOperator

    def evaluate(self, data: StockData2, period: slice = slice(0, 1)) -> Tensor:
        start = period.start - self._delta_time + 1
        stop = period.stop
        # L: period length (requested time window length)
        # W: window length (dt for rolling)
        # S: stock count
        values = self._operand.evaluate(data, slice(start, stop))   # (L+W-1, S)
        values = values.unfold(0, self._delta_time, 1)              # (L, S, W)
        return self._apply(values)                                  # (L, S)

    @abstractmethod
    def _apply(self, operand: Tensor) -> Tensor: ...

    def __str__(self) -> str:
        return f"{type(self).__name__}({self._operand},{self._delta_time})"

    @property
    def is_featured(self): return self._operand.is_featured


class PairRollingOperator(Operator):
    def __init__(self,
                 lhs: Expression, rhs: Expression,
                 delta_time: Union[int, DeltaTime]) -> None:
        self._lhs = lhs if isinstance(lhs, Expression) else Constant(lhs)
        self._rhs = rhs if isinstance(rhs, Expression) else Constant(rhs)
        if isinstance(delta_time, DeltaTime):
            delta_time = delta_time._delta_time
        self._delta_time = delta_time

    @classmethod
    def n_args(cls) -> int: return 3

    @classmethod
    def category_type(cls) -> Type['Operator']: return PairRollingOperator

    def _unfold_one(self, expr: Expression,
                    data: StockData2, period: slice = slice(0, 1)) -> Tensor:
        start = period.start - self._delta_time + 1
        stop = period.stop
        # L: period length (requested time window length)
        # W: window length (dt for rolling)
        # S: stock count
        values = expr.evaluate(data, slice(start, stop))            # (L+W-1, S)
        return values.unfold(0, self._delta_time, 1)                # (L, S, W)

    def evaluate(self, data: StockData2, period: slice = slice(0, 1)) -> Tensor:
        lhs = self._unfold_one(self._lhs, data, period)
        rhs = self._unfold_one(self._rhs, data, period)
        return self._apply(lhs, rhs)                                # (L, S)

    @abstractmethod
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: ...

    def __str__(self) -> str:
        return f"{type(self).__name__}({self._lhs},{self._rhs},{self._delta_time})"

    @property
    def is_featured(self): return self._lhs.is_featured or self._rhs.is_featured


# Unary Operators
class Abs(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.abs()


class Sign(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.sign()


class Log(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.log()


class CSRank(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        nan_mask = operand.isnan()
        n = (~nan_mask).sum(dim=1, keepdim=True)
        rank = operand.argsort().argsort() / n
        rank[nan_mask] = torch.nan
        return rank

# new unary operators implementations
class Inverse(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor: return 1 / operand

class Log10(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.log10()

class Log2(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.log2()

class Reverse(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor: return -operand

class Relu(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor: return torch.relu(operand)

class Round(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.round()

class S_LOG_LP(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.sign() * (1+operand.abs()).log()

class S_LOG2_LP(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.sign() * (1+operand.abs()).log2()

class S_LOG10_LP(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.sign() * (1+operand.abs()).log10()

class Sigmoid(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.sigmoid()

class Sin(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.sin()

class Sqrt(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.sqrt()

class TANH(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.tanh()

class Ceil(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.ceil()

class Floor(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.floor()

class Frac(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.sign() * (operand.abs() - operand.abs().floor())

class Demean(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor: return  operand - operand.nanmean()

class Exp(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.exp()


# Binary Operators
class Add(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: return lhs + rhs

class Sub(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: return lhs - rhs

class Mul(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: return lhs * rhs

class Div(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: return lhs / rhs

class Pow(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: return lhs ** rhs

class Greater(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: return lhs.max(rhs)
    @property
    def is_featured(self):
        return self._lhs.is_featured and self._rhs.is_featured

class Less(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: return lhs.min(rhs)
    @property
    def is_featured(self):
        return self._lhs.is_featured and self._rhs.is_featured

# new binary operators
class Signed_Pow(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: return lhs.sign() * lhs ** rhs

class Mod(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: return lhs % rhs

# class Ression_Proj(BinaryOperator):
#     # Performs cross-sectional regression Y = a + bX
#     def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
#         n_days, n_stocks = lhs.shape
#         y_hat = torch.zeros_like(rhs)
#         for t in range(n_days):
#             Xt = lhs[t,:]
#             Xt = torch.nan_to_num(Xt, nan=0.0) # fill NaN with 0
#             Yt = rhs[t,:]
#             Yt = torch.nan_to_num(Yt, nan=0.0) # fill NaN with 0
#             Xt_with_intercept = torch.cat([torch.ones(Xt.shape[0], 1, device=Xt.device), Xt.unsqueeze(1)], dim=1)  # add a column of 1s to X
#             # OLS Regression
#             XtX = Xt_with_intercept.T @ Xt_with_intercept
#             XtY = Xt_with_intercept.T @ Yt
#             beta = torch.linalg.pinv(XtX) @ XtY
#             at = beta[0]
#             bt = beta[1]
#             y_hat[t, :] = at + bt * Xt
#         return y_hat
#
# class Tregresi(BinaryOperator):
#     def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
#         n_days, n_stocks = lhs.shape
#         residuals = torch.zeros_like(rhs)
#         for t in range(n_days):
#             Xt = lhs[t,:]
#             Xt = torch.nan_to_num(Xt, nan=0.0) # fill NaN with 0
#             Yt = rhs[t,:]
#             Yt = torch.nan_to_num(Yt, nan=0.0) # fill NaN with 0
#             Xt_with_intercept = torch.cat([torch.ones(Xt.shape[0], 1, device=Xt.device), Xt.unsqueeze(1)], dim=1)  # add a column of 1s to X
#             # OLS Regression
#             XtX = Xt_with_intercept.T @ Xt_with_intercept
#             XtY = Xt_with_intercept.T @ Yt
#             beta = torch.linalg.pinv(XtX) @ XtY
#             at = beta[0]
#             bt = beta[1]
#             y_hat = at + bt * Xt
#             residuals[t, :] = Yt - y_hat
#         return residualsegr

class FloorDiv(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor, Filter=0) -> Tensor: return lhs//rhs






# rolling operators
class TS_Ref(RollingOperator):
    # Ref is not *really* a rolling operator, in that other rolling operators
    # deal with the values in (-dt, 0], while Ref only deal with the values
    # at -dt. Nonetheless, it should be classified as rolling since it modifies
    # the time window.

    def evaluate(self, data: StockData2, period: slice = slice(0, 1)) -> Tensor:
        start = period.start - self._delta_time
        stop = period.stop - self._delta_time
        return self._operand.evaluate(data, slice(start, stop))

    def _apply(self, operand: Tensor) -> Tensor:
        # This is just for fulfilling the RollingOperator interface
        ...

class TS_Mean(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.mean(dim=-1)


class TS_Sum(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.sum(dim=-1)


class TS_Std(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.std(dim=-1)


class TS_Var(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.var(dim=-1)


class TS_Skew(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        # skew = m3 / m2^(3/2)
        central = operand - operand.mean(dim=-1, keepdim=True)
        m3 = (central ** 3).mean(dim=-1)
        m2 = (central ** 2).mean(dim=-1)
        return m3 / m2 ** 1.5


class TS_Kurtosis_S(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        # kurt = m4 / var^2 - 3
        central = operand - operand.mean(dim=-1, keepdim=True)
        m4 = (central ** 4).mean(dim=-1)
        var = operand.var(dim=-1)
        return m4 / var ** 2 - 3


class TS_Max(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.max(dim=-1)[0]


class TS_Min(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.min(dim=-1)[0]


class TS_Med(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.median(dim=-1)[0]


class TS_Mad(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        central = operand - operand.mean(dim=-1, keepdim=True)
        return central.abs().mean(dim=-1)


class TS_Rank(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        n = operand.shape[-1]
        last = operand[:, :, -1, None]
        left = (last < operand).count_nonzero(dim=-1)
        right = (last <= operand).count_nonzero(dim=-1)
        result = (right + left + (right > left)) / (2 * n)
        return result


class TS_Delta(RollingOperator):
    # Delta is not *really* a rolling operator, in that other rolling operators
    # deal with the values in (-dt, 0], while Delta only deal with the values
    # at -dt and 0. Nonetheless, it should be classified as rolling since it
    # modifies the time window.

    def evaluate(self, data: StockData2, period: slice = slice(0, 1)) -> Tensor:
        start = period.start - self._delta_time
        stop = period.stop
        values = self._operand.evaluate(data, slice(start, stop))
        return values[self._delta_time:] - values[:-self._delta_time]

    def _apply(self, operand: Tensor) -> Tensor:
        # This is just for fulfilling the RollingOperator interface
        ...


class TS_WMA(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        n = operand.shape[-1]
        weights = torch.arange(n, dtype=operand.dtype, device=operand.device)
        weights /= weights.sum()
        return (weights * operand).sum(dim=-1)


class TS_EMA(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        n = operand.shape[-1]
        alpha = 1 - 2 / (1 + n)
        power = torch.arange(n, 0, -1, dtype=operand.dtype, device=operand.device)
        weights = alpha ** power
        weights /= weights.sum()
        return (weights * operand).sum(dim=-1)

# new rolling operators implementations
class TS_Product(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.prod(dim=-1)

class TS_IR(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return operand.mean(dim=-1) / operand.std(dim=-1)

class TS_Delay(RollingOperator):
    def evaluate(self, data: StockData2, period: slice = slice(0, 1)) -> Tensor:
        start = period.start - self._delta_time
        stop = period.stop
        values = self._operand.evaluate(data, slice(start, stop))
        return values[:-self._delta_time]
    def _apply(self, operand: Tensor) -> Tensor:
        ...

class TS_Max_Diff(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return operand[:,:,-1] - operand.max(dim=-1)[0]

class TS_AV_Diff(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return operand[:,:,-1] - operand.mean(dim=-1)

class TS_Kurtosis(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        central = operand - operand.mean(dim=-1, keepdim=True)
        m4 = (central ** 4).mean(dim=-1)
        var = operand.var(dim=-1)
        return m4 / var ** 2

class TS_Min_Diff(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return operand[:,:,-1] - operand.min(dim=-1)[0]

class TS_Scale(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        num= operand[:,:,-1] - operand.min(dim=-1)[0]
        den = operand.max(dim=-1)[0] - operand.min(dim=-1)[0]
        return num / den

class TS_Zscore(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return (operand[:,:,-1] - operand.mean(dim=-1)) / operand.std(dim=-1)



# pairwise rolling operators
class TS_Cov(PairRollingOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        n = lhs.shape[-1]
        clhs = lhs - lhs.mean(dim=-1, keepdim=True)
        crhs = rhs - rhs.mean(dim=-1, keepdim=True)
        return (clhs * crhs).sum(dim=-1) / (n - 1)


class TS_Corr(PairRollingOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        clhs = lhs - lhs.mean(dim=-1, keepdim=True)
        crhs = rhs - rhs.mean(dim=-1, keepdim=True)
        ncov = (clhs * crhs).sum(dim=-1)
        nlvar = (clhs ** 2).sum(dim=-1)
        nrvar = (crhs ** 2).sum(dim=-1)
        stdmul = (nlvar * nrvar).sqrt()
        stdmul[(nlvar < 1e-6) | (nrvar < 1e-6)] = 1
        return ncov / stdmul





# Deprecated!
Operators: List[Type[Expression]] = [
    # Unary
    Abs, Sign, Log, CSRank, Inverse, Log10, Log2, Reverse, Relu, Round, S_LOG_LP, S_LOG2_LP,
    S_LOG10_LP, Sigmoid, Sin, TANH, Sqrt, Ceil, Floor, Frac, Demean, Exp,
    # Binary
    Add, Sub, Mul, Div, Pow, Greater, Less, Signed_Pow, Mod, FloorDiv,
    # Tregresi, Regression_Proj
    # Rolling
    TS_Ref, TS_Mean, TS_Sum, TS_Std, TS_Var, TS_Skew, TS_Kurtosis_S, TS_Max, TS_Min,
    TS_Med, TS_Mad, TS_Rank, TS_Delta, TS_WMA, TS_EMA,
    TS_Product, TS_IR, TS_Delay, TS_Max_Diff, TS_AV_Diff, TS_Kurtosis, TS_Min_Diff, TS_Scale, TS_Zscore,

    # Pair rolling
    TS_Cov, TS_Corr
]


# Seperate calling
UnaryOperators: List[Type[Expression]] = [
    Abs, Sign, Log, CSRank, Inverse, Log10, Log2, Reverse, Relu, Round,
    S_LOG_LP, S_LOG2_LP, S_LOG10_LP, Sigmoid, Sin, TANH, Sqrt, Ceil, Floor,
    Frac, Demean, Exp
]

BinaryOperators:  List[Type[Expression]]  = [
    Add, Sub, Mul, Div, Pow, Greater, Less, Signed_Pow, Mod,
    FloorDiv,
    # Tregresi, Regression_Proj,
]

RollingOperators: List[Type[Expression]] = [
    TS_Ref, TS_Mean, TS_Sum, TS_Std, TS_Var, TS_Skew, TS_Kurtosis_S, TS_Max, TS_Min,
    TS_Med, TS_Mad, TS_Rank, TS_Delta, TS_WMA, TS_EMA,
    TS_Product, TS_IR, TS_Delay, TS_Max_Diff, TS_AV_Diff, TS_Kurtosis, TS_Min_Diff, TS_Scale, TS_Zscore
]

PairRollingOperators: List[Type[Expression]] = [
    TS_Cov, TS_Corr
]