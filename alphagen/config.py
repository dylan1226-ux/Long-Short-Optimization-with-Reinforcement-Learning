from alphagen.data.expression import *


MAX_EXPR_LENGTH = 20
MAX_EPISODE_LENGTH = 256

OPERATORS = [
    # Unary
    Abs, Sign, Log, CSRank, Inverse, Log10, Log2, Reverse, Relu, Round, S_LOG_LP, S_LOG2_LP,
    S_LOG10_LP, Sigmoid, Sin, TANH, Sqrt, Ceil, Floor, Frac, Demean, Exp,
    # Binary
    Add, Sub, Mul, Div, Pow, Greater, Less, Signed_Pow, Mod, FloorDiv,
    # Regression_Proj, Tregresi
    # Rolling
    TS_Ref, TS_Mean, TS_Sum, TS_Std, TS_Var, TS_Skew, TS_Kurtosis_S, TS_Max, TS_Min,
    TS_Med, TS_Mad, TS_Rank, TS_Delta, TS_WMA, TS_EMA,
    TS_Product, TS_IR, TS_Delay, TS_Max_Diff, TS_AV_Diff, TS_Kurtosis, TS_Min_Diff, TS_Scale, TS_Zscore,

    # Pair rolling
    TS_Cov, TS_Corr
]

DELTA_TIMES = [10, 20, 30, 40, 50]

CONSTANTS = [-30., -10., -5., -2., -1., -0.5, -0.01, 0.01, 0.5, 1., 2., 5., 10., 30.]

REWARD_PER_STEP = 0.
