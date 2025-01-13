from alphagen.data.expression import Feature, Ref
from alphagen_qlib.my_data import FeatureType


af_open = Feature(FeatureType.AF_OPEN)
close = Feature(FeatureType.CLOSE)
af_high = Feature(FeatureType.AF_HIGH)
af_low = Feature(FeatureType.AF_LOW)
volume = Feature(FeatureType.VOLUME)
af_vwap = Feature(FeatureType.AF_VWAP)
amount = Feature(FeatureType.AMOUNT)
avg_price = Feature(FeatureType.AVG_PRICE)
main_in_flow = Feature(FeatureType.MAIN_IN_FLOW)
tot_market_cap = Feature(FeatureType.TOT_MARKET_CAP)
turn_rate = Feature(FeatureType.TURN_RATE)
turnover_5d = Feature(FeatureType.TURNOVER_5D)


# net_fund_in_5d = Feature(FeatureType.NET_FUND_IN_5D)
# pb = Feature(FeatureType.PB)
# chgpct = Feature(FeatureType.CHGPCT)

target = TS_Ref(close, -20) / close - 1