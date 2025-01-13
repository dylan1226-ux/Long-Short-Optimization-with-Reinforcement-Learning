import os
import pandas as pd
from typing import Dict

from torch.onnx.symbolic_opset9 import tensor

from alphagen_qlib.my_data import FeatureType,StockData2
from alphagen.data.expression import *
from alphagen.utils.pytorch_utils import normalize_by_day
import re

class AlphaExpressionParser:
    def __init__(self, operator_map:Dict[str, Type[Expression]],feature_map:Dict[str, FeatureType]) -> None:
        self.operator_map = operator_map
        self.feature_map = feature_map

    def split_arguments(self, inner_expression: str) -> List[str]:
        """
        Split the inner expression into arguments based on comma separation (nested case)
        :param inner_expression
        :return: A list
        """
        arguments = []
        depth = 0
        current_arg = ''

        # Iterate over each character in the inner expression
        for char in inner_expression:
            if char == ',' and depth == 0:
                arguments.append(current_arg)
                current_arg = ''
            else:
                current_arg += char
                if char == '(':
                    depth += 1
                elif char == ')':
                    depth -= 1
        if current_arg:
            arguments.append(current_arg)

        return arguments

    def parse(self, expression: str) -> Union[Expression, float]:
        """
        Recursively parse an alpha expression string into an expression object.
        :param expression: The alpha expression string.
        :return: An Expression object or float value if it’s a constant.
        """

        expression = expression.replace(" ", "")

        # Constant expressions
        if expression.startswith('Constant(') and expression.endswith(')'):
            # Slice to remove 'Constant(' and ')'
            value = float(expression[len('Constant('):-1])
            return Constant(value)

        # Feature expressions (e.g., $open)
        match = re.match(r'^\$(\w+)$', expression)
        if match:
            feature_name = match.group(1).lower()
            if feature_name in self.feature_map:
                return Feature(self.feature_map[feature_name])
            else:
                raise ValueError(f"Unknown feature: {feature_name}")

        for op_name, op_class in self.operator_map.items():
            if expression.startswith(op_name + '(') and expression.endswith(')'):
                inner_expression = expression[len(op_name) + 1:-1]  # Extract inner expressions
                arguments = self.split_arguments(inner_expression)
                parsed_args = []
                # iterate over each argument
                for i, arg in enumerate(arguments):
                    # For rolling operators, if the second argument is a integer
                    if op_class in RollingOperators and i == 1:
                        try:
                            parsed_args.append(int(arg))
                        except ValueError:
                            parsed_args.append(self.parse(arg))
                    # pair rolling, the third argument is a integer
                    elif op_class in PairRollingOperators and i==2:
                        try:
                            parsed_args.append(int(arg))
                        except ValueError:
                            parsed_args.append(self.parse(arg))
                    # Unary and Binary
                    else:
                        parsed_args.append(self.parse(arg))
                return op_class(*parsed_args)

        raise ValueError(f"Could not parse expression: {expression}")

def tensor_to_df1(tenor:torch.Tensor,row_labels=None,column_labels=None) -> pd.DataFrame:
    np_array = tenor.detach().cpu().numpy()
    df = pd.DataFrame(data=np_array, index=row_labels, columns=column_labels)
    df.reset_index(inplace=True)
    df.columns.name = None
    df.rename(columns={'index':'trade_date'}, inplace=True)
    return df

def tensor_to_df2(tensor, row_labels, column_labels):
    # Flatten the tensor to a long format
    flattened_tensor = tensor.detach().cpu().numpy().flatten()

    # Create a DataFrame with three columns: 'trade_date', 'ticker', and 'factor'
    df = pd.DataFrame({
        'trade_date': row_labels.repeat(len(column_labels)),
        'ticker': list(column_labels) * len(row_labels),
        'signal_value': flattened_tensor
    })
    return df

# example usage
operator_map = {
    # Unary
    "Abs":Abs, "Sign":Sign, "Log":Log, "CSRank":CSRank, "Inverse":Inverse, "Log10":Log10, "Log2":Log2,
    "Reverse": Reverse, "Relu":Relu, "Round":Round, "S_LOG_LP":S_LOG_LP, "S_LOG2_LP":S_LOG2_LP,
    "S_LOG10_LP":S_LOG10_LP, 'Sigmoid': Sigmoid, 'Sin':Sin, "TANH":TANH, "Sqrt":Sqrt, "Ceil":Ceil,
    "Floor":Floor, "Frac":Frac, "Demean":Demean, "Exp":Exp,
    # Binary
    "Add":Add, "Sub":Sub,"Mul":Mul,"Div":Div,"Pow":Pow,"Greater":Greater,"Less":Less,
    "Signed_Pow":Signed_Pow, "Mod":Mod,"FloorDiv":FloorDiv,
    # Rolling
    "TS_Ref":TS_Ref, 'TS_Mean':TS_Mean,  "TS_Sum":TS_Sum, "TS_Std":TS_Std, "TS_Var":TS_Var, "TS_Skew":TS_Skew, 'TS_Kurtosis_S':TS_Kurtosis_S, 'TS_Max':TS_Max, 'TS_Min':TS_Min,
    "TS_Med": TS_Med, "TS_Mad": TS_Mad, "TS_Rank":TS_Rank, "TS_Delta":TS_Delta, "TS_WMA":TS_WMA, "TS_EMA":TS_EMA,
    'TS_Product':TS_Product, 'TS_IR':TS_IR, "TS_Delay":TS_Delay, "TS_Max_Diff": TS_Max_Diff, "TS_AV_Diff":TS_AV_Diff, "TS_Kurtosis" : TS_Kurtosis,
    'TS_Min_Diff': TS_Min_Diff, "TS_Scale":TS_Scale, "TS_Zscore":TS_Zscore,
    # Pair Rolling
    "TS_Cov":TS_Cov, "TS_Corr":TS_Corr
}


feature_map ={
"buy_trades_act":FeatureType.buy_trades_act,
"close_moneyflow_pct_value":FeatureType.close_moneyflow_pct_value,
"close_moneyflow_pct_volume":FeatureType.close_moneyflow_pct_volume,
"close_net_inflow_rate_value":FeatureType.close_net_inflow_rate_value,
"close_net_inflow_rate_volume":FeatureType.close_net_inflow_rate_volume,
"cancell_buy_order_value":FeatureType.cancell_buy_order_value,
"large_trades_return":FeatureType.large_trades_return,
"large_buy_value":FeatureType.large_buy_value,
"large_sell_value":FeatureType.large_sell_value,
"factor_bolldown20d":FeatureType.factor_bolldown20d,
"factor_obv1d":FeatureType.factor_obv1d,
"factor_pvt1d":FeatureType.factor_pvt1d,
"factor_willr14d":FeatureType.factor_willr14d,
"factor_adxr14d":FeatureType.factor_adxr14d,
"factor_apbma5d":FeatureType.factor_apbma5d,
"factor_bias10d":FeatureType.factor_bias10d,
"factor_cci5d":FeatureType.factor_cci5d,
"factor_ema5d":FeatureType.factor_ema5d,
"factor_adtm":FeatureType.factor_adtm,
"factor_stom":FeatureType.factor_stom,
"factor_dmi5d":FeatureType.factor_dmi5d,
"factor_variance20d":FeatureType.factor_variance20d,
"factor_atr14d":FeatureType.factor_atr14d,
"factor_roc6d":FeatureType.factor_roc6d,
"factor_kdjk9d":FeatureType.factor_kdjk9d,
}


def single_alpha_value(Expression, data):
    parser = AlphaExpressionParser(operator_map, feature_map)
    parsed_expression = parser.parse(Expression)
    tensor_data = normalize_by_day(parsed_expression.evaluate(data))
    return tensor_data

def alpha_value_combined(alpha_dic, data):
    alpha_value_combined = None
    for expression, weight in alpha_dic.items():
        parser = AlphaExpressionParser(operator_map, feature_map)
        parsed_expression = parser.parse(expression)
        # normalization
        alpha_value_tensor = normalize_by_day(parsed_expression.evaluate(data))
        alpha_value_tensor = alpha_value_tensor * weight
        # add to the combination tensor
        if alpha_value_combined is None:
            alpha_value_combined = alpha_value_tensor
        else:
            alpha_value_combined += alpha_value_tensor
    return alpha_value_combined


def create_expression_dict(input_data):
    exprs = input_data.get("exprs",[])
    weights = input_data.get("weights",[])
    if len(exprs) != len(weights):
        raise ValueError(f"Expecting {len(exprs)} weights, got {len(weights)}")

    alpha_dic = {expr: weight for expr, weight in zip(exprs, weights)}

    return alpha_dic



if __name__ == '__main__':

    from joblib import load



    out_sample_path = '/tmp/pycharm_project_877/company_data/processed_data/out_sample_23_24.joblib'

    out_sample = StockData2(csv_path=out_sample_path,
                            start_time='2023-01-01',
                            end_time='2024-12-29',
                            out_sample=True)


    step_36864 = {"exprs": ["$factor_bias10d", "Reverse(TANH($factor_atr14d))", "$close_moneyflow_pct_volume", "CSRank($close_net_inflow_rate_volume)", "Div(Constant(-30.0),$factor_stom)", "$factor_pvt1d", "$factor_kdjk9d", "Sin($close_net_inflow_rate_volume)", "Pow(Signed_Pow(Constant(30.0),Round($factor_pvt1d)),Constant(-2.0))", "FloorDiv(FloorDiv(Constant(0.01),$factor_atr14d),Signed_Pow($close_net_inflow_rate_volume,Constant(0.5)))", "Signed_Pow(Relu(Div(Constant(-1.0),Frac(Log2(Relu(TANH($factor_pvt1d)))))),$large_sell_value)", "FloorDiv(Constant(-10.0),$factor_dmi5d)", "Log2($factor_obv1d)", "Mod(Abs($factor_kdjk9d),Round(Sign(Pow(Floor($factor_kdjk9d),Less(Sigmoid(Sigmoid($buy_trades_act)),Constant(-2.0))))))", "Sub(Greater(Constant(0.01),Sign(Add($large_trades_return,Constant(-1.0)))),$factor_bias10d)"], "weights": [-5.815041126138843e-05, -3.373562466802735e-05, -8.882739463492746e-05, 5.6846625272928416e-05, -2.612348263767177e-05, 3.058221193128543e-05, 3.095856170337016e-05, 3.0317677454192168e-05, -8.911237113072122e-05, -5.816759091384426e-06, -0.000136758454246194, -3.066903098642874e-05, -3.0140180934235974e-05, 4.338779031206239e-05, 3.0846941233045615e-05]}
    alpha_dic = create_expression_dict(step_36864)
    for expr, weight in alpha_dic.items():
        print(expr,weight)

    tensor_data_out_sample = alpha_value_combined(alpha_dic,out_sample)
    print(tensor_data_out_sample)



    # to data frame
    dates = out_sample._dates
    start = out_sample.max_backtrack_days
    stop = out_sample.max_backtrack_days + out_sample.n_days
    dates = pd.to_datetime(dates[start:stop])
    factor_data = tensor_to_df2(tensor_data_out_sample, row_labels=dates, column_labels=out_sample._stock_ids)



    # sort values
    factor_data = factor_data.sort_values(by=['ticker','trade_date'])
    # shift to avoid using today's data
    factor_data['signal_value'] = factor_data.groupby('ticker')['signal_value'].shift(1)
    factor_data = factor_data.dropna()
    factor_data.reset_index(inplace=True,drop=True)
    factor_data['ticker'] = factor_data['ticker'].astype('int')
    factor_data['signal_value'] = factor_data['signal_value'].astype('float64')

    print(factor_data)

    # to csv file
    directory = '/tmp/pycharm_project_877/alpha_value'
    if not os.path.exists(directory):
        os.makedirs(directory)
    csv_filename = os.path.join(directory, "sharpe_v3.csv")
    factor_data.to_csv(csv_filename,index=False)
