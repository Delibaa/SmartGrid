# emSimulation.py
# Energy Market Running Simulation
import pandas as pd
import os
from datetime import datetime
from pathlib import Path
import csv
from web3 import AsyncWeb3, AsyncHTTPProvider
import sg_config as Config
from Endpoints.consumer import AgentConsumer
from Endpoints.prosumer import AgentProsumer
from Endpoints.NG import AgentNationalGrid
from Blockchain.interact import get_energy_market
from . import conversion
from . import clearMarketCalculation


# get blockchain connection
w3 = AsyncWeb3(AsyncHTTPProvider(Config.RPC_URL))
logger = Config.get_logger(os.path.splitext(os.path.basename(__file__))[0])
outputFileName = 'EMoutput.csv'
# make sure path existed
OUTPUTS_DIR: Path = Config.OUTPUTS_DIR
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


# create consumers, prosumers and national grid
async def createAgents():
    agentsConsumer = []
    agentsProsumer = []

    # National grid
    agentNationalGrid = AgentNationalGrid(Config.NATIONAL_GRID_PRICE)
    await agentNationalGrid.init(w3)
    await agentNationalGrid.getAccount()

    # --- consumers ---
    for i in range(0, Config.CONSUMER_NUMBER):
        agentConsumer = AgentConsumer(Config.MAX_BATTERY_CAPACITY, i + 1)
        await agentConsumer.init(w3)
        await agentConsumer.getAccount(i + 1)
        await agentConsumer.loadSmartMeterData()
        agentsConsumer.append(agentConsumer)

    # --- prosumers ---
    # global index for prosumers starts after consumers
    for k in range(Config.CONSUMER_NUMBER, Config.CONSUMER_NUMBER + Config.PROSUMER_NUMBER):
        global_idx = k + 1
        agentProsumer = AgentProsumer(Config.MAX_BATTERY_CAPACITY, global_idx)
        await agentProsumer.init(w3)
        await agentProsumer.getAccount(global_idx)
        await agentProsumer.loadSmartMeterData()
        agentsProsumer.append(agentProsumer)

    return agentsConsumer, agentsProsumer, agentNationalGrid


# get energy market info from blockchain


async def getInfo(contract_em):
    # get bids and asks
    bids = []
    bidsCount = await contract_em.functions.getBidsCount().call()
    asks = []
    asksCount = await contract_em.functions.getAsksCount().call()

    if bidsCount == 0 or asksCount == 0:
        logger.info(
            f"Nothing from the energy market...")
        return bids, asks

    if bidsCount > 0:
        for i in range(0, bidsCount-1):
            bid = await contract_em.functions.getBid(i).call()
            Abid = {
                'owner': bid[0],
                'price': bid[1],
                'amount': bid[2],
                'date': datetime.fromtimestamp(bid[3]),
                'orderId': bid[4]
            }
            bids.append(Abid)

    if asksCount > 0:
        for i in range(0, asksCount-1):
            ask = await contract_em.functions.getAsk(i).call()
            Aask = {
                'owner': ask[0],
                'price': ask[1],
                'amount': ask[2],
                'date': datetime.fromtimestamp(ask[3]),
                'orderId': ask[4]               
            }
            asks.append(Aask)

    logger.info(
        f"Get {len(bids)} bids and {len(asks)} asks from the energy market...")

    return bids, asks

# clear market info
# only delete all info in the blockchain, starting a new session of trading
async def clearMarket(agentNationGrid, contract_em):
    receipt = None
    try:
        tx_hash = await contract_em.functions.clearMarket().transact({
            'from': agentNationGrid.ethereumAddress,
            'gas': 90_000_000
        })
        receipt = await w3.eth.wait_for_transaction_receipt(tx_hash)
    except Exception as err:
        print('Error in clearing market', err)
        logger.error(f"Error in clearing market: {err}")

    return receipt

# main simulation logic


async def energyMarketSimulation():
    unfilled_bids = []  # after clearing market, unfilled bids
    unfilled_asks = []  # after clearing market, unfilled asks
    time_array = []  # output data, time

    logger.info(f"Initializing simulation of energy market...")
    logger.info(f"Initializing contract instances...")
    contract_em = await get_energy_market(w3)

    # settings
    agentsConsumer, agentsProsumer, agentNationalGrid = await createAgents()
    # all agents of consumers and prosumers
    agents = agentsConsumer + agentsProsumer

    logger.info(
        f"Endpoints created, {len(agentsConsumer)} consumers, {len(agentsProsumer)} prosumers and 1 national grid...")

    # show data to return to API
    data_to_return_api = []
    data_to_return_api_national_grid_price = Config.NATIONAL_GRID_PRICE

    try:
        await clearMarket(agentNationalGrid, contract_em)
    except Exception as err:
        logger.error("Error while trying to clear market", err)
    
    try: 
        await agentNationalGrid.clearScoreBillboard()
    except Exception as err:
        logger.error("Error while trying to clear billboard", err)

    logger.info(f"Starting simulation...")
    logger.info(
        f"Picking up days from {Config.TIME_WINDOW_START} to {Config.TIME_WINDOW_END} to simulate...")
    
    # Simulation loop
    for d in range (Config.TIME_WINDOW_START, Config.TIME_WINDOW_END):
        # first predict the next day's consumption
        for agent in agents:
            await agent.predictNextDay(d)
            agent.turnOffAllMachines() # reset all machines to off status at the beginning of each day

        for t in range((d - 1) * 24 + 1, d * 24 + 1):
            # time point
            time_array.append(t)
            logger.info(f"Current time point is day {d}, {t - (d - 1) * 24 - 1}:00 ...")

            # decision making per agent
            for agent in agents:
                agent.setCurrentTime(t, d)
                # set national grid once at window start
                if t == (Config.TIME_WINDOW_START - 1) * 24 + 1:
                    agent.setNationalGrid(
                        Config.NATIONAL_GRID_PRICE, agentNationalGrid.ethereumAddress)

                try:
                    # let each agent decide their status, whether to ask or bid
                    await agent.decisionMaker()
                except Exception as err:
                    logger.error(f"Error from decision maker", err)
            
            # get info from the market and match
            bds, aks = await getInfo(contract_em)

            if len(bds) >= 1 and len(aks) >= 1:
                # smallest price last in list
                bds = sorted(bds, key=lambda x: x["price"], reverse=True)
                aks = sorted(aks, key=lambda x: x["price"], reverse=True)
                logger.info(f"Starting matching information from the energy market...")
                # get trade prices
                tradePrices = []
                info = await clearMarketCalculation.matchInfo(len(bds) - 1, len(aks) - 1, bds, aks, agents, tradePrices)

                if info == "no match":
                    logger.info(f"No match info for this time point's market...")
                    avg_price = 0
                    for agent in agents:
                        agent.historicalTradePrices.append(avg_price)
                elif info == "match finished":
                    logger.info(f"Matching finished...")
                    if tradePrices is None:
                        avg_price = 0
                    else:
                        avg_price = round(sum(tradePrices) / len(tradePrices), 3)
                    for agent in agents:
                        agent.historicalTradePrices.append(avg_price)
            else:
                avg_price = 0
                for agent in agents:
                        agent.historicalTradePrices.append(avg_price)

        # write the real consumption into new files
        for agent in agents:
            agent.storeRealConsumption()
        
        # publish the next day's miner/leader
        await agentNationalGrid.calculateNextLeader(d + 1)

        # get left bids & asks from blockchain (pending orders to be cleared)
        bids, asks = await getInfo(contract_em)

        # todo2-there should be an extra match information to match those transactions from the blockchain
        # and then clearing the market in the frequency of one day, now one day one point, one day one clearing

        if len(bids) >= 1 and len(asks) >= 1:
            # start market clearing
            # [float: quantity/whatt, int: priceWei]
            intersection = clearMarketCalculation.calculate_intersection(
                bids, asks)
            price_dollars = conversion.convert_wei_to_dollars(intersection[1])
            logger.info(
                f"Current market clearing price in Swiss Franc: {price_dollars} CHF/kwh")

            # sort by amount descending (smallest amount last in list => match from end)
            bids = sorted(bids, key=lambda x: x["amount"], reverse=True)
            asks = sorted(asks, key=lambda x: x["amount"], reverse=True)

            # store clearing price to each agent historicalClearPrices at time t
            for agent in agents:
                agent.historicalClearPrices[d] = intersection[1]

            # match left bids and asks
            unfilled_bids, unfilled_asks = await clearMarketCalculation.matchClear(len(bids) - 1, len(asks) - 1, bids, asks, agents, intersection[1])

            # leftovers: buyers -> NG
            if len(unfilled_bids) > 0:
                for b in bids:
                    buyer = next(
                        (agent for agent in agents if agent.ethereumAddress == b["owner"]), None)
                    if buyer is not None:
                        await buyer.buyFromNationalGrid(b["amount"])

            # leftovers: sellers -> charge
            if len(unfilled_asks) > 0:
                for a in asks:
                    seller = next(
                        (agent for agent in agents if agent.ethereumAddress == a["owner"]), None)
                    if seller is not None:
                        seller.charge(a["amount"])
            # delete all data in blockchain for time point t
            try:
                await clearMarket(agentNationalGrid, contract_em)
            except Exception as err:
                logger.error("Error while trying to clear market", err)

        # no intersection (no bids or no asks to match)
        else:
            if len(bids) > 0:
                for b in bids:
                    unfilled_bids.append(b)
                    buyer = next(
                        (agent for agent in agents if agent.ethereumAddress == b["owner"]), None)
                    if buyer is not None:
                        await buyer.buyFromNationalGrid(b["amount"])

            if len(asks) > 0:
                for a in asks:
                    unfilled_asks.append(a)
                    seller = next(
                        (agent for agent in agents if agent.ethereumAddress == a["owner"]), None)
                    if seller is not None:
                        seller.charge(a["amount"])

            # record zero clearing price since no exchange
            for agent in agents:
                agent.historicalClearPrices[d] = 0

            try:
                await clearMarket(agentNationalGrid, contract_em)
            except Exception as err:
                logger.error("Error while trying to clear market", err)

    await agentNationalGrid.clearScoreBillboard()
    logger.info("Simulation ends, starting outputting and analyzing data...")

    # --- Post-processing / stats ---
    # CSV rows accumulator for calculation
    csv_data = []  # storing on-chain data
    aggregated_demand = []  # output data, total demand
    aggregated_supply = []  # output data, total supply

    transaction_cost_bid = []
    transaction_cost_ask = []
    transaction_cost_trade = []
    total_number_transactions = []
    total_number_transactions_EM = []

    # energy price from the energy market
    csv_price_path = os.path.join(Config.OUTPUTS_DIR, "clearPrice.csv")
    file_exists = os.path.isfile(csv_price_path)
    if not file_exists or os.path.getsize(csv_price_path) == 0:
        with open(csv_price_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["day", "price(CHF/kwh)"])

    with open(csv_price_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        t = 0
        for d in range(Config.TIME_WINDOW_START, Config.TIME_WINDOW_END):
            price = conversion.convert_wei_to_dollars(
                agents[0].historicalClearPrices[d]
            )
            writer.writerow([d, price])
            t += 1
    # writing billboard
    # energy price from the energy market
    csv_price_path = os.path.join(Config.OUTPUTS_DIR, "scorebillboard.csv")
    file_exists = os.path.isfile(csv_price_path)
    if not file_exists or os.path.getsize(csv_price_path) == 0:
        with open(csv_price_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["house_Id", "day", "error"])

    with open(csv_price_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        t = 0
        for info in agentNationalGrid.submitLeaderHistory:
            houseid = info["houseId"]
            day = info["day"]
            err = info["err"]
            writer.writerow([houseid, day, err])
            t += 1

    # record per hour
    for t in range((Config.TIME_WINDOW_START - 1) * 24 + 1, (Config.TIME_WINDOW_END - 1) * 24 + 1):
        demand = []  # total demand at the time t
        supply = []  # total supply at the time t
        gas_cost_bids = []  # total gas cost of sending all bids at time t
        gas_cost_asks = []  # total gas cost of sending all asks at time t
        successful_bids_gas = [] # total gas consumption of successfull bids in the energy market at the time t
        suceessful_asks_gas = [] # total gas consumption of successfull asks in the energy market at the time t
        national_grid_bids_gas = []  # total gas cost of all bids from the national grid

        for consumer in agentsConsumer:
            # demand
            demand_value = next(
                (rec['demand'] for rec in consumer.historicalDemand if rec['time'] == t), None)
            demand.append(demand_value)
            # gas
            gas_bid_value = next(
                (rec['transactionCost'] for rec in consumer.bidHistory if rec['timeRow'] == t), None)
            if gas_bid_value is not None:
                gas_cost_bids.append(gas_bid_value)

            gas_bid_suc_value = next(
                (rec['transactionCost'] for rec in consumer.successfulBidHistory if rec['timeRow'] == t), None)
            if gas_bid_suc_value is not None:
                successful_bids_gas.append(gas_bid_suc_value)

            gas_bid_ng_value = next(
                (rec['transactionCost'] for rec in consumer.nationalGridPurchases if rec['timeRow'] == t), None)
            if gas_bid_ng_value is not None:
                national_grid_bids_gas.append(gas_bid_ng_value)

        for prosumer in agentsProsumer:
            # demand && supply
            demand_value = next(
                (rec['demand'] for rec in prosumer.historicalDemand if rec['time'] == t), None)
            demand.append(demand_value)
            supply_value = next(
                (rec['supply'] for rec in prosumer.historicalSupply if rec['time'] == t), None)
            supply.append(supply_value)

            # gas
            gas_bid_value = next(
                (rec['transactionCost'] for rec in prosumer.bidHistory if rec['timeRow'] == t), None)
            # attention here, even none will be a object of this array, so there should be a filter
            if gas_bid_value is not None:
                gas_cost_bids.append(gas_bid_value)
            gas_ask_value = next(
                (rec['transactionCost'] for rec in prosumer.askHistory if rec['timeRow'] == t), None)
            if gas_ask_value is not None:
                gas_cost_asks.append(gas_ask_value)

            # gas
            gas_bid_suc_value = next(
                (rec['transactionCost'] for rec in consumer.successfulBidHistory if rec['timeRow'] == t), None)
            if gas_bid_suc_value is not None:
                successful_bids_gas.append(gas_bid_suc_value)

            gas_ask_suc_value = next(
                (rec['transactionCost'] for rec in consumer.successfulBidHistory if rec['timeRow'] == t), None)
            if gas_ask_suc_value is not None:
                suceessful_asks_gas.append(gas_ask_suc_value)

            gas_bid_ng_value = next(
                (rec['transactionCost'] for rec in consumer.nationalGridPurchases if rec['timeRow'] == t), None)
            if gas_bid_ng_value is not None:
                national_grid_bids_gas.append(gas_bid_ng_value)

        # global index
        idx = t - ((Config.TIME_WINDOW_START - 1) * 24 + 1)
        # sums to dollars/CHF
        bid_cost_dollars = conversion.convert_array_gas_to_dollars(
            gas_cost_bids)
        # ensure arrays index by (t - start) rather than absolute t, correct time number
        if len(transaction_cost_bid) <= idx:
            transaction_cost_bid.extend(
                [0.0] * (idx - len(transaction_cost_bid) + 1))
        transaction_cost_bid[idx] = bid_cost_dollars

        ask_cost_dollars = conversion.convert_array_gas_to_dollars(
            gas_cost_asks)
        # ensure arrays index by (t - start) rather than absolute t, correct time number
        if len(transaction_cost_ask) <= idx:
            transaction_cost_ask.extend(
                [0.0] * (idx - len(transaction_cost_ask) + 1))
        transaction_cost_ask[idx] = ask_cost_dollars

        # ensure arrays index by (t - start) rather than absolute t, correct time number
        trade_cost_dollars = conversion.convert_array_gas_to_dollars(
            successful_bids_gas)
        if len(transaction_cost_trade) <= idx:
            transaction_cost_trade.extend(
                [0.0] * (idx - len(transaction_cost_trade) + 1))
        transaction_cost_trade[idx] = trade_cost_dollars

        # total counts, revision1
        sum_transactions = (
            len(national_grid_bids_gas)
            + len(gas_cost_asks)
            + len(gas_cost_bids)
            + len(successful_bids_gas)
        )
        total_number_transactions.append(sum_transactions)

        number_market_transactions = (
            len(gas_cost_asks)
            + len(gas_cost_bids)
            + len(successful_bids_gas)
        )
        total_number_transactions_EM.append(number_market_transactions)

        # aggregate supply/demand
        sum_demand = sum(demand) if demand else 0.0
        sum_supply = sum(supply) if supply else 0.0

        # ensure arrays index by (t - start) rather than absolute t, correct time number
        if len(aggregated_demand) <= idx:
            aggregated_demand.extend(
                [0.0] * (idx - len(aggregated_demand) + 1))
        if len(aggregated_supply) <= idx:
            aggregated_supply.extend(
                [0.0] * (idx - len(aggregated_supply) + 1))

        aggregated_demand[idx] = sum_demand
        aggregated_supply[idx] = sum_supply

        csv_row = {
            "time": t,
            "total_demand": aggregated_demand[idx],
            "total_supply": aggregated_supply[idx],
            "tradePirces": conversion.convert_wei_to_dollars(agents[0].historicalTradePrices[idx]),
            "total_transactions": total_number_transactions[idx],
            "total_gas_cost_bids": transaction_cost_bid[idx],
            "total_gas_cost_asks": transaction_cost_ask[idx],
            "total_gas_cost_trade": transaction_cost_trade[idx],
            "total_trades_from_Market": len(successful_bids_gas),
            "total_transactions_from_Market": number_market_transactions,
            "total_transactions_from_NG": len(national_grid_bids_gas),
        }
        csv_data.append(csv_row)

    # output file
    output_file = os.path.join(OUTPUTS_DIR, outputFileName)

    logger.info(
        f"Writing results of simulation to csv file : {output_file}...")

    # write CSV (headers from keys of first row)
    if csv_data:
        fieldnames = list(csv_data[0].keys())
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in csv_data:
                writer.writerow(row)

        logger.info(f"  ✔ Energy Market simulation finished...")

    return {
        "data": data_to_return_api,
        "nationalGridAddress": agentNationalGrid.ethereumAddress,
        "nationalGridPrice": data_to_return_api_national_grid_price,
    }

# Close the underlying aiohttp session used by AsyncHTTPProvider.
async def shutdown():
    global w3
    if w3 is not None:
        provider = getattr(w3, "provider", None)
        close = getattr(provider, "close", None)
        if callable(close):
            await close()   # close aiohttp.ClientSession
        w3 = None