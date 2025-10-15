# conversion.py
# on-chain unit -> off chain unit: gas, wei, ether -> swiss francs (CHF)
import sg_config as Config

# convert ether value to dollars, with 3 decimal places for the output value
def _to_dollars(ether):
    return round(ether * Config.PRICE_OF_ETHER, 3)

# convert wei value to dollars
def convert_wei_to_dollars(wei_value):
    cost_ether = (wei_value * 1000) / Config.WEI_IN_ETHER
    return _to_dollars(cost_ether)

# convert gas units to dollars
def convert_gas_to_dollars(gas_cost):
    calc_price = gas_cost * Config.GAS_PRICE            # wei
    cost_ether = calc_price / Config.WEI_IN_ETHER      # ETH
    return _to_dollars(cost_ether)

# Sum wei values, convert to ETH → dollars.
def convert_array_wei_to_dollars(array_wei):
    sum_cost = sum(array_wei)                          # wei
    cost_ether = sum_cost / Config.WEI_IN_ETHER        # ETH
    return _to_dollars(cost_ether)

# Sum gas units, convert to ETH → dollars.
def convert_array_gas_to_dollars(array):
    #   ✔ todo-1 some transaction receipt with gas as none, how?
    # cleaned = []
    # for x in array:
    #     try:
    #         if x is None: 
    #             continue
    #         cleaned.append(int(x))
    #     except (TypeError, ValueError):
    #         continue
    if len(array) == 0:
        return float(0)
    sum_cost = sum(array)
    calc_price = sum_cost * Config.GAS_PRICE            # wei
    cost_ether = calc_price / Config.WEI_IN_ETHER      # ETH
    return _to_dollars(cost_ether)


