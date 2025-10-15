# clearMarketCalculation.py
# Market Cleaning Calculation process
import math
import numpy as np
import sg_config as Config

# calculate the clearing price and quantity
# linear helper, linear regression
def _linear_fit(points):
    if len(points) == 0:
        return 0.0, 0.0
    xs = np.array([p[0] for p in points], dtype=float)
    ys = np.array([p[1] for p in points], dtype=float)
    # If all x are identical, fallback to horizontal line at mean y
    if np.allclose(xs, xs[0]):
        return 0.0, float(np.mean(ys))
    m, b = np.polyfit(xs, ys, 1)  # slope, intercept
    return float(m), float(b)

# calculation logic
def calculate_intersection(array1, array2):
    """
    the logic of calculation is as follows:
    - bids: sort by price desc to formulate cumulative-amount vs price, demand curve
    - asks: sort by price asc to formulate cumulative-amount vs price, supply curve
    - linear regression on two curevs
    - solve for intersection x, then y from bids line
    Returns [x, int(y)] with fallback if y is inf/undefined.
    """
    # defensive copies
    bids = list(array1 or [])
    asks = list(array2 or [])

    # sort (stable)
    bids.sort(key=lambda x: x["price"], reverse=True)   # desc
    asks.sort(key=lambda x: x["price"])                 # asc

    # Build cumulative amount for bids
    cum = 0.0
    bid_points = []   # (cum_amount, price)
    for r in bids:
        cum += float(r["amount"])
        bid_points.append((cum, float(r["price"])))

    # Build cumulative amount for asks; start with (0,0)
    ask_points = [(0.0, 0.0)]
    cum = 0.0
    for r in asks:
        cum += float(r["amount"])
        ask_points.append((cum, float(r["price"])))

    # Linear regressions
    m1, b1 = _linear_fit(bid_points)  # bids: y = m1 x + b1
    m2, b2 = _linear_fit(ask_points)  # asks: y = m2 x + b2

    # Solve for x: m1 x + b1 = m2 x + b2  ->  x = (b2 - b1) / (m1 - m2)
    denom = (m1 - m2)
    if abs(denom) < 1e-12:
        x = 0.0  # nearly parallel; fallback like JS would end up doing
    else:
        x = (b2 - b1) / denom
    # Predict y from bids line, meaning clearing price
    y = m1 * x + b1

    # if y is Infinity/undefined, use a large default
    if (y is None) or (isinstance(y, float) and (math.isinf(y) or math.isnan(y))):
        y = 240000000000000

    return [float(x), int(y)] # [quantity/whatt and price/wei]

# match bids and asks from the energy market
# helper
def _find_agent(agents, address):
    return next((agent for agent in agents if agent.ethereumAddress == address), None)
# match and clear method
async def matchClear(bidIndex, askIndex, bids, asks, agents, clearPrice):
    # matching process is over
    if len(bids) == 0 or len(asks) == 0:
        return bids, asks

    # find match bids and asks
    # Buyer (from current bid)
    buyer_addr = bids[bidIndex]["owner"]
    buyer_amt = bids[bidIndex]["amount"]

    # Seller (from current ask)
    seller_addr = asks[askIndex]["owner"]
    seller_amt = asks[askIndex]["amount"]

    buyer_obj = _find_agent(agents, buyer_addr)

    # Compare bid vs ask amounts
    diff = buyer_amt - seller_amt  # >=0 means bid covers or equals ask

    if diff >= 0:
        # Buyer can take all from this seller
        calc_amount = seller_amt
        # Transfer at market clearing price
        await buyer_obj.sendFunds(clearPrice, calc_amount, seller_addr)
        # Charge battery
        buyer_obj.charge(calc_amount)

        # Mark success for seller
        seller_obj = _find_agent(agents, seller_addr)
        if seller_obj is None:
            raise ValueError(
                f"Seller agent not found for address {seller_addr}")
        seller_obj.addSuccessfulAsk(calc_amount)

        # Update remaining amounts
        bids[bidIndex]["amount"] = diff    # leftover on buyer
        # Remove fully cleared orders
        if diff == 0:
            del bids[bidIndex]
        del asks[askIndex]

        # Recurse from last elements
        return await matchClear(len(bids) - 1, len(asks) - 1, bids, asks, agents, clearPrice)

    else:
        # Seller amount > buyer amount, meaning buyer is fully consumed
        calc_amount = buyer_amt
        await buyer_obj.sendFunds(clearPrice, calc_amount, seller_addr)
        buyer_obj.charge(calc_amount)  # charge

        # seller
        seller_obj = _find_agent(agents, seller_addr)
        seller_obj.addSuccessfulAsk(calc_amount)

        # Update remaining amounts
        asks[askIndex]["amount"] = seller_amt - buyer_amt  # leftover on seller
        del bids[bidIndex]  # buyer fully matched
        if asks[askIndex]["amount"] == 0:
            del asks[askIndex]

        return await matchClear(len(bids) - 1, len(asks) - 1, bids, asks, agents, clearPrice)
    
# match and buy method
async def matchInfo(bidIndex, askIndex, bids, asks, agents, tradePrices):
    # natural stop
    if len(bids) == 0 or len(asks) == 0:
        return str("match finished")
    
    # stop because price gap
    if bids[0]["price"] < asks[askIndex]["price"]:
        return str("no match")
    
    # find match bids and asks
    # Buyer (from current bid)
    buyer_addr = bids[bidIndex]["owner"]
    buyer_amt = bids[bidIndex]["amount"]
    buyer_p = bids[bidIndex]["price"]
    buyer_oId = bids[bidIndex]["orderId"]

    # Seller (from current ask)
    seller_addr = asks[askIndex]["owner"]
    seller_amt = asks[askIndex]["amount"]
    seller_p = asks[askIndex]["price"]
    seller_oId = asks[askIndex]["orderId"]

    if buyer_p >= seller_p:
        # find buys and sellers
        buyer_obj = _find_agent(agents, buyer_addr)
        seller_obj = _find_agent(agents, seller_addr)

        # buying amount
        diff = buyer_amt - seller_amt

        tradePrices.append(buyer_p)

        # buyer's amount is greater than seller's amount
        if diff >= 0:
            # get buying amount
            amt = seller_amt
            await buyer_obj.sendFunds(buyer_p, amt, seller_addr)
            buyer_obj.charge(amt)
            seller_obj.addSuccessfulAsk(amt)
            # leftover for the bid
            bids[bidIndex]["amount"] = diff
            # change on-chain amount
            await buyer_obj.reviseBuy(diff, buyer_oId)
            # same amount
            if diff == 0:
                del bids[bidIndex]
                # delete onchain record
                await buyer_obj.delBuy(buyer_oId)
            del asks[askIndex]
            await seller_obj.delAsk(seller_oId)

            return await matchInfo(len(bids) - 1, len(asks) - 1, bids, asks, agents, tradePrices)
        
        # buyer's amount is smaller than seller's amount
        if diff < 0:
            # get buying amount
            amt = buyer_amt
            await buyer_obj.sendFunds(buyer_p, amt, seller_addr)
            buyer_obj.charge(amt)
            seller_obj.addSuccessfulAsk(amt)
            # leftover for the ask
            asks[askIndex]["amount"] = seller_amt - buyer_amt
            await seller_obj.reviseAsk(seller_amt - buyer_amt, seller_oId)
            del bids[bidIndex]
            await buyer_obj.delBuy(buyer_oId)
            if asks[askIndex]["amount"] == 0:
                del asks[askIndex]
                await seller_obj.delAsk(seller_oId)

            return await matchInfo(len(bids) - 1, len(asks) - 1, bids, asks, agents, tradePrices)
    
    else:
        del bids[bidIndex]
        return await matchInfo(len(bids) - 1, len(asks) - 1, bids, asks, agents, tradePrices)
