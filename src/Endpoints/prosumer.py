# prosumer.py
# prosumer agent class, onchain unit, whatt/amount and wei/price
from Blockchain.interact import get_energy_market
import time
import pandas as pd
import os
import csv
import math
import random
from datetime import datetime
import sg_config as Config
from Endpoints.AIAgent import consumption_prediction

# prosumer
class AgentProsumer:
    def __init__(self, batteryCapacity, houseID):
        # account settings
        self.householdID = houseID  # household ID in the smart grid, number from 1
        self.ethereumAddress = None  # ethereum address of the prosumer
        self.balance = 0  # balance of the prosumer in wei

        # simulation settings
        self.timeRow = 0  # current time row in the historic data
        self.day = 0 # current day
        self.hasBattery = True  # prosumer has a battery
        self.batteryCapacity = batteryCapacity  # maximum in whatt
        self.amountOfCharge = batteryCapacity  # current amount of charge in whatt

        # current supply and production status, single time row
        self.excessEnergy = 0  # whether there is excess energy in whatt at this time row
        self.shortageEnergy = 0  # whether there is shortage of energy in whatt at this time row
        # current demand in whatt at this time row, {'time': int, 'demand': float}
        self.currentDemand = 0
        # current supply in whatt at this time row, {'time': int, 'supply': float}
        self.currentSupply = 0

        # global settings of the national grid
        self.nationalGridAddress = None  # ethereum address of the national grid
        self.nationalGridPrice = 0  # in wei

        # history settings
        # list of dict {'time': int, 'demand': float}
        self.historicalDemand = []
        # list of dict {'time': int, 'supply': float}
        self.historicalSupply = []
        # list of float, historical cleared prices at each time row
        self.historicalClearPrices = [0] * Config.TIME_WINDOW_END
        # list of float, historical trading price at each time point
        self.historicalTradePrices = []
        # list of dict {'transactionReceipt': receipt, 'transactionCost': int, 'transactionAmount': int, 'date': timestamp, 'quantity': float, 'timeRow': int}
        self.nationalGridPurchases = []
        # list of dict {'address': addr, 'price': int, 'amount': int, 'date': timestamp, 'timeRow': int, 'transactionCost': int}
        self.bidHistory = []  # onchain only with int format
        # list of dict {'address': addr, 'price': int, 'amount': int, 'date': timestamp, 'timeRow': int, 'transactionCost': int}
        self.askHistory = []
        # list of dict {'transactionReceipt': receipt, 'transactionCost': int, 'transactionAmount': int, 'date': timestamp, 'quantity': float, 'timeRow': int}
        self.successfulBidHistory = []
        # list of dict {'amount': float, 'date': timestamp, 'timeRow': int}
        self.successfulAskHistory = []

        # endpoints control
        # household machine switch
        self.availableMachines = {
            "television": False,
            "heat_pump": True,
            "lamps": False,
            "dishwasher": False,
            "washing_machine": False,
            "microwave": False,
            "oven": False,
            "electric_water_heater": True,
            "vacuum_cleaner": False,
            "laptop_desktop": False,
            "tumble_dryer": False
        }
        # store prediction value for next 24 hours/points, [float]
        self.predictionNextDay = []
        # stroe real consumption of today 24 hours/points, [float]
        self.realConsumption = [0] * 24

        # contracts instances
        self.contact_em = None
        self.w3 = None

        # get logger
        self.logger = Config.get_logger(os.path.splitext(os.path.basename(__file__))[0] + str(self.householdID))

    # get contract instances
    async def init(self, w3):
        # Every instance gets the same shared contract
        self.w3 = w3
        self.contract_em = await get_energy_market(self.w3)

    # get the recommanded prediction for the next 24 hours
    async def predictNextDay(self, predictionDay):
        # predicting data for the next day
        self.logger.info("AIagent is predicting data for the next day...")
        # attention converting unit to whatt
        self.predictionNextDay = consumption_prediction(
            self.getHistoricalDataDir(),
            self.householdID,
            predictionDay,
            0.3
        ) * 1000
        self.storePredictiveConsumption(predictionDay)
        self.logger.info("  ✔ Prediction finished...")
    
    # simulation house consumption and production
    async def loadSmartMeterData(self):
        # load historic data
        data_dir = self.getHistoricalDataDir()
        historicData = pd.read_csv(data_dir)
        for i in range(0, len(historicData)-1):
            currentDemand = {
                'time': historicData.iloc[i, 0],
                'demand': float(historicData.iloc[i, 1]) * 1000  # in whatt
            }
            currentSupply = {
                'time': historicData.iloc[i, 0],
                'supply': float(historicData.iloc[i, 2]) * 1000  # in whatt
            }
            self.historicalDemand.append(currentDemand)
            self.historicalSupply.append(currentSupply)
        # logger
        self.logger.info(f"  ✔ Prosumer created successfully...")
        return True

    # get ethereum account
    async def getAccount(self, index):
        accounts = await self.w3.eth.accounts
        self.ethereumAddress = accounts[index]
        self.logger.info(f"  ✔ Initialize with address of {self.ethereumAddress}...")

    # get balance of the account
    async def getAgentBalance(self):
        balance = await self.w3.eth.get_balance(self.ethereumAddress)
        self.balance = balance
    
    # get data dir
    def getHistoricalDataDir(self):
        historicalDataDir = os.path.join(
            Config.DATA_DIR, "prosumer", "historicaldata", f"house_{self.householdID}.csv")
        return historicalDataDir

    def getPredictedDataDir(self):
        predictedDataDir = os.path.join(
            Config.DATA_DIR, "prosumer", "predictiondata", f"house_{self.householdID}.csv")
        return predictedDataDir

    def getRealDataDir(self):
        realDataDir = os.path.join(Config.DATA_DIR, "prosumer", "realdata", f"house_{self.householdID}.csv")
        return realDataDir

    # convert price in CHF to wei, onchain unit-wei
    def convertToWei(self, price):
        calcPrice_eth = round(price / Config.PRICE_OF_ETHER, 18)
        try:
            priceWei = self.w3.to_wei(float(calcPrice_eth), 'ether')
        except Exception as err:
            print('Error from conversion', err)
            self.logger.error(f"Error from conversion: {err}")
            priceWei = 0
        priceWei = int(priceWei)  # onchain unit wei in integer format
        return priceWei

    # set national grid info
    def setNationalGrid(self, nationalGridPrice, nationalGridAddress):
        self.nationalGridAddress = nationalGridAddress
        price = round(nationalGridPrice / 1000.0, 8)
        self.nationalGridPrice = self.convertToWei(price)
        self.logger.info(f"  ✔ Set national grid successfully...")

    # buy energy from national grid
    async def buyFromNationalGrid(self, amount):
        # calculate the amount to pay
        timerow = self.timeRow
        amountTransaction = int(round(self.nationalGridPrice * amount, 18))
        transactionReceipt = None
        try:
            tx_hash = await self.w3.eth.send_transaction({
                'to': self.nationalGridAddress,
                'from': self.ethereumAddress,
                'value': amountTransaction,
                'gas': 3_000_000
            })
            receipt = await self.w3.eth.wait_for_transaction_receipt(tx_hash)
            transactionReceipt = receipt
        except Exception as err:
            print('Error buying from national grid', err)
            self.logger.error(f"Error buying from national grid: {err}")

        # record the purchase
        date = int(time.time())
        gasUsed = transactionReceipt['gasUsed'] if transactionReceipt and transactionReceipt['gasUsed'] else 0
        newTransactionReceipt = {
            'transactionReceipt': transactionReceipt,
            'transactionCost': gasUsed,
            'transactionAmount': amountTransaction,
            'date': date,
            'quantity': amount,
            'timeRow': timerow
        }
        self.nationalGridPurchases.append(newTransactionReceipt)
        # battery charge update
        self.charge(amount)
        self.logger.info(f"Buying power {amount} whatt from the national grid...")
        return transactionReceipt

    # send funds to other address, buy action
    async def sendFunds(self, price, amount, receiver):
        timerow = self.timeRow
        amountTransaction = int(round(price * amount, 18))
        transactionReceipt = None
        try:
            tx_hash = await self.w3.eth.send_transaction({
                'to': receiver,
                'from': self.ethereumAddress,
                'value': amountTransaction
            })
            receipt = await self.w3.eth.wait_for_transaction_receipt(tx_hash)
            transactionReceipt = receipt
        except Exception as err:
            print('Error in sending funds', err)
            self.logger.error(f"Error in sending funds: {err}")

        # record the successful bid
        date = int(time.time())
        gasUsed = transactionReceipt['gasUsed'] if transactionReceipt and transactionReceipt['gasUsed'] else 0
        newTransactionReceipt = {
            'transactionReceipt': transactionReceipt,
            'transactionCost': gasUsed,
            'transactionAmount': amountTransaction,
            'timeRow': timerow,
            'date': date,
            'quantity': amount,
            'receiver': receiver
        }
        self.successfulBidHistory.append(newTransactionReceipt)
        # battery charge update
        self.charge(amount)
        self.logger.info(f"Buying power {amount} whatt from the prosumer {receiver}...")
        return transactionReceipt

    # place bid to buy energy from the market
    async def placeBuy(self, price, amount, date):
        timerow = self.timeRow
        transactionReceipt = None
        order_id = self.w3.solidity_keccak(
            ["address", "uint256", "uint256", "uint256"],
            [self.ethereumAddress, price, amount, date]
        )
        try:
            tx_hash = await self.contract_em.functions.addBid(math.floor(price), math.floor(amount), date, order_id).transact({
                'from': self.ethereumAddress,
                'gas': 3_000_000
            })
            receipt = await self.w3.eth.wait_for_transaction_receipt(tx_hash)
            transactionReceipt = receipt
        except Exception as err:
            print('Error in placeBuy', err)
            self.logger.error(f"Error in placeBuy: {err}")

        # record the bid
        gasUsed = transactionReceipt['gasUsed'] if transactionReceipt and transactionReceipt['gasUsed'] else 0
        newBid = {
            'address': self.ethereumAddress,
            'price': price,
            'amount': amount,
            'date': date,
            'timeRow': timerow,
            'transactionCost': gasUsed
        }
        self.bidHistory.append(newBid)
        self.logger.info(f"Send a bid with {amount} whatt to the energy market...")
        return True
    
    # delete bid
    async def delBuy(self, orderId):
        try:
            await self.contract_em.functions.removeBid(orderId).transact({
                'from': self.ethereumAddress,
                'gas': 3_000_000
            })
        except Exception as err:
            self.logger.error(f"Error in placeBuy: {err}")
        self.logger.info(f"Delete the bid {orderId.hex()} in the energy market...")

    # revise bid
    async def reviseBuy(self, amount, orderId):
        try:
            await self.contract_em.functions.updateBidAmount(orderId, amount).transact({
                'from': self.ethereumAddress,
                'gas': 3_000_000
            })
        except Exception as err:
            self.logger.error(f"Error in placeBuy: {err}")
        self.logger.info(f"Revise the bid {orderId.hex()}'s amount in the energy market...")

    # place ask to sell energy to the market
    async def placeAsk(self, price, amount, date):
        timerow = self.timeRow
        transactionReceipt = None
        order_id = self.w3.solidity_keccak(
            ["address", "uint256", "uint256", "uint256"],
            [self.ethereumAddress, price, amount, date]
        )
        try:
            tx_hash = await self.contract_em.functions.addAsk(math.floor(price), math.floor(amount), date, order_id).transact({
                'from': self.ethereumAddress,
                'gas': 3_000_000
            })
            receipt = await self.w3.eth.wait_for_transaction_receipt(tx_hash)
            transactionReceipt = receipt
        except Exception as err:
            print('Error in placeAsk', err)
            self.logger.error(f"Error in placeAsk: {err}")

        # record the ask
        gasUsed = transactionReceipt['gasUsed'] if transactionReceipt and transactionReceipt['gasUsed'] else 0
        newAsk = {
            'address': self.ethereumAddress,
            'price': price,
            'amount': amount,
            'date': date,
            'timeRow': timerow,
            'transactionCost': gasUsed
        }
        self.askHistory.append(newAsk)
        self.logger.info(f"Send a ask with {amount} whatt to the energy market...")
        return True
    
    # delete ask
    async def delAsk(self, orderId):
        try:
            await self.contract_em.functions.removeAsk(orderId).transact({
                'from': self.ethereumAddress,
                'gas': 3_000_000
            })
        except Exception as err:
            self.logger.error(f"Error in placeBuy: {err}")
        self.logger.info(f"Delete the ask {orderId.hex()} in the energy market...")

    # revise ask
    async def reviseAsk(self, amount, orderId):
        try:
            await self.contract_em.functions.updateAskAmount(orderId, amount).transact({
                'from': self.ethereumAddress,
                'gas': 3_000_000
            })
        except Exception as err:
            self.logger.error(f"Error in placeBuy: {err}")
        self.logger.info(f"Revise the bid {orderId.hex()}'s amount in the energy market...")

    # record successful ask, that is, sell energy successfully
    def addSuccessfulAsk(self, amount):
        date = int(time.time())  # in seconds
        newReceivedTransaction = {'amount': amount,
                                  'date': date, 'timeRow': self.timeRow}
        self.successfulAskHistory.append(newReceivedTransaction)
        self.logger.info(f"Successfully sell {amount} energy...")

    # charge battery
    def charge(self, amount):
        self.amountOfCharge += amount
        if self.amountOfCharge > self.batteryCapacity:
            self.amountOfCharge = self.batteryCapacity
            self.logger.info("Battery has been fully charged...")

    # discharge battery
    def discharge(self, amount):
        self.amountOfCharge -= amount
        if self.amountOfCharge <= 0:
            self.logger.info("No enough power left...")
            return False
        return True

    # set current time row
    def setCurrentTime(self, row, day):
        self.timeRow = row
        self.day = day

    # calculate yesterday average price
    def calculateYesterdayAverage(self):
        start = (self.day - Config.TIME_WINDOW_START - 1) * 24
        end = start + 24
        if start < 0:
            return None
        sumPrices = 0
        for i in range(start, end):
            sumPrices += self.historicalTradePrices[i]
        return round(sumPrices / 24, 3)

    # formulate the amount to buy based on historical data
    def formulateAmount(self):
        timeInterval = 12  # demand for next 12 hours
        supplySum = 0
        demandSum = 0
        for i in range(self.timeRow - 1, self.timeRow + timeInterval):
            supplySum += self.historicalSupply[i]['supply']
            demandSum += self.historicalDemand[i]['demand']
        # if supply is more than demand, no need to buy
        if supplySum - demandSum >= 0:
            return False
        else:
            energyNeeded = abs(supplySum - demandSum)
        # consider battery capacity
        if self.amountOfCharge + energyNeeded >= self.batteryCapacity:
            energyNeeded = self.amountOfCharge + energyNeeded - self.batteryCapacity
        return energyNeeded

    # get bids in asending price order
    async def getBids(self):
        bids = []
        bidsCount = await self.contract_em.functions.getBidsCount().call()
        if bidsCount == 0:
            return bids
        for i in range(0, bidsCount-1):
            bid = await self.contract_em.functions.getBid(i).call()
            Abid = {
                'owner': bid[0],
                'price': bid[1],
                'amount': bid[2],
                'date': datetime.fromtimestamp(bid[3]),
                'orderId': bid[4]
            }
            bids.append(Abid)
        bids = sorted(bids, key=lambda x: x['price'])
        return bids

    # get asks in asending price order
    async def getAsks(self):
        asks = []
        asksCount = await self.contract_em.functions.getAsksCount().call()
        if asksCount == 0:
            return asks
        for i in range(0, asksCount-1):
            ask = await self.contract_em.functions.getAsk(i).call()
            Aask = {
                'owner': ask[0],
                'price': ask[1],
                'amount': ask[2],
                'date': datetime.fromtimestamp(ask[3]),
                'orderId': ask[4]
            }
            asks.append(Aask)
        asks = sorted(asks, key=lambda x: x['price'])
        return asks
    
    # turn off machines
    def turnOffMachines(self):
        # Count how many are ON
        active_count = sum(
            1 for status in self.availableMachines.values() if status)
        self.logger.info(f"Currently active machines: {active_count}...")

        if active_count > 0:
            # Randomly choose one active machine to turn off
            active_machines = [name for name,
                               status in self.availableMachines.items() if status]
            to_turn_off = random.choice(active_machines)
            self.availableMachines[to_turn_off] = False
            self.logger.info(f"Turned off: {to_turn_off} to manage self...")
            return True
        else:
            self.logger.info(f"Nothing to turn off...")
            return False

    # turn off all machines when a day is over
    def turnOffAllMachines(self):
        # Count how many are ON
        for machine in self.availableMachines:
            self.availableMachines[machine] = False
        self.availableMachines["heat_pump"] = True
        self.availableMachines["electric_water_heater"] = True
        self.logger.info(f"Turned off all machines at the end of the day...")

    # turn on machines
    def turnOnMachines(self):
        # Count how many are Off
        active_count = sum(
            1 for status in self.availableMachines.values() if not status)
        self.logger.info(f"Currently inactive machines: {active_count}...")

        if active_count > 0:
            # Randomly choose one active machine to turn on
            active_machines = [name for name,
                               status in self.availableMachines.items() if not status]
            to_turn_on = random.choice(active_machines)
            self.availableMachines[to_turn_on] = True
            self.logger.info(f"Turned on: {to_turn_on} to manage self...")
            return True
        else:
            self.logger.info(f"Nothing to turn on...")
            return False

    # write the real consumption into the real dir
    def storeRealConsumption(self):
        # path
        realPath = self.getRealDataDir()
        values = list(self.realConsumption)

        file_exists = os.path.isfile(realPath)
        if not file_exists or os.path.getsize(realPath) == 0:
            with open(realPath, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["time", "demand"])
            last_time = (self.day - 1) * 24
        else:
            with open(realPath, mode="r") as f:
                reader = csv.reader(f)
                rows = list(reader)
                if len(rows) <= 1:
                    last_time = (self.day - 1) * 24
                else:
                    last_time = int(rows[-1][0])

        with open(realPath, mode="a", newline="") as f:
            writer = csv.writer(f)
            for i, val in enumerate(values, start=1):
                writer.writerow([last_time + i, round(val / 1000, 3)])
        self.logger.info(f"Store the real consumption of day {self.day} into database...")

    # store Predictive consumption
    def storePredictiveConsumption(self, predictionDay):
        # path
        realPath = self.getPredictedDataDir()
        values = list(self.predictionNextDay)

        file_exists = os.path.isfile(realPath)
        if not file_exists or os.path.getsize(realPath) == 0:
            with open(realPath, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["time", "demand"])
            last_time = (predictionDay - 1) * 24
        else:
            with open(realPath, mode="r") as f:
                reader = csv.reader(f)
                rows = list(reader)
                if len(rows) <= 1:
                    last_time = (predictionDay - 1) * 24
                else:
                    last_time = int(rows[-1][0])

        with open(realPath, mode="a", newline="") as f:
            writer = csv.writer(f)
            for i, val in enumerate(values, start=1):
                writer.writerow([last_time + i, round(val / 1000, 3)])
        self.logger.info(f"Store the predictive consumption of day {predictionDay} into database...")
    
    # self management
    def selfManagement(self, idx, prediction, real):
        if prediction < real:
            if self.turnOffMachines():  # save energy
                return prediction
            else:
                return real
        elif prediction > real:
            if self.turnOnMachines():  # have more energy, can use more machines
                return prediction
            else:
                return real
        else:
            return real

            
    # main logic of the prosumer agent, when to act with energy market, when to buy, when to sell
    async def decisionMaker(self):
        # demand and supply status
        # demand at the current time row
        idx = (self.timeRow % 24) - 1
        pd = self.predictionNextDay[idx]
        d = self.historicalDemand[self.timeRow - 1]['demand']
        # self management

        demand = self.selfManagement(idx, pd, d)
        self.realConsumption[idx] = demand
                
        # supply at the current time row
        supply = self.historicalSupply[self.timeRow - 1]['supply']
        excessEnergy = 0  # whether there is excess energy at this time row
        shortageOfEnergy = 0  # whether there is shortage of energy at this time row
        time_ms = int(time.time())  # date

        # get market info
        bids = await self.getBids()  # bids info from the market
        bidsCount = len(bids)
        # asks = await self.getAsks()  # asks info from the market
        # asksCount = len(asks)
        lowestBidPrice = 0  # lowest bid price
        price = 0  # random price

        # determine the status of energy
        if supply >= demand:
            excessEnergy = supply - demand
        if supply < demand:
            shortageOfEnergy = demand - supply

        # return data json to API of UI
        returnData = {}
        returnData['supply'] = supply
        returnData['demand'] = demand
        if excessEnergy:
            returnData['excessEnergy'] = excessEnergy
        if shortageOfEnergy:
            returnData['shortageEnergy'] = shortageOfEnergy
        returnData['hasBattery'] = self.hasBattery

        # decision logic
        # for prosumer with surplus energy
        if excessEnergy > 0:
            # charge battery first if battery is less than 50% charged
            if self.amountOfCharge <= 0.5 * self.batteryCapacity:
                self.charge(excessEnergy)
                returnData['batteryPercentage'] = '< 50%'
                returnData['action'] = 'Battery Charged'
                self.logger.info(f"Storing excess energy into the battery...")
            # sell or charge battery based on market situation if battery is 50%-80% charged
            elif 0.5 * self.batteryCapacity < self.amountOfCharge < 0.8 * self.batteryCapacity:
                # check the market situation
                returnData['batteryPercentage'] = '50-80%'
                returnData['bidsCount'] = bidsCount
                # if there is bid in the market
                if bidsCount > 0:
                    lowestBidPrice = bids[0]['price']
                    if (self.timeRow - 24) > 0 and (self.calculateYesterdayAverage() is not None):
                        # calculate yesterday average price
                        averagePrice = self.calculateYesterdayAverage()
                        returnData['yesterdayAverage'] = averagePrice
                        returnData['lowestBidPrice'] = lowestBidPrice
                        # if the bid price is higher than yesterday average price, sell energy
                        if lowestBidPrice >= averagePrice:
                            await self.placeAsk(lowestBidPrice, math.floor(
                                excessEnergy), time_ms)
                            returnData['task'] = 'Send Ask, selling energy'
                        # otherwise charge battery and sell excess energy if battery is full
                        else:
                            if self.amountOfCharge + excessEnergy <= self.batteryCapacity:
                                self.charge(excessEnergy)
                                returnData['task'] = 'Charge Battery'
                                self.logger.info(f"Storing excess energy into the battery since no good price...")
                            else:
                                self.charge(excessEnergy)
                                excessEnergy = self.amountOfCharge + excessEnergy - self.batteryCapacity
                                await self.placeAsk(lowestBidPrice, math.floor(
                                    excessEnergy), time_ms)
                                returnData['task'] = 'Send Ask, selling energy'
                                self.logger.info(f"Storing excess energy into the battery and sell energy to the network if more left...")
                # if there is no bid in the market, charge battery if not full
                else:
                    if self.amountOfCharge + excessEnergy <= self.batteryCapacity:
                        self.charge(excessEnergy)
                        returnData['task'] = 'Excess energy & no bid => Charged Battery'
                    else:
                        self.charge(excessEnergy)
                        excessEnergy = self.amountOfCharge + excessEnergy - self.batteryCapacity
                        price = random.random() * 0.1 + 0.1
                        price = self.convertToWei(price/1000)
                        await self.placeAsk(price, math.floor(excessEnergy), time_ms)
                        returnData['task'] = 'Excess energy => Charged Battery & bid'
            # sell excess energy if battery is more than 90% charged
            else:
                # sell excess energy in the price of 0.1-0.2 CHF/kWh
                price = random.random() * 0.1 + 0.1
                price = self.convertToWei(price/1000)
                await self.placeAsk(price, math.floor(excessEnergy), time_ms)
                returnData['batteryPercentage'] = '> 80 %'
                returnData['task'] = 'Send Ask, selling energy'
        # for prosumer with shortage of energy
        elif shortageOfEnergy > 0:
            # use battery if battery is more than 50% charged
            if self.amountOfCharge >= 0.5 * self.batteryCapacity:
                returnData['batteryPercentage'] = '> 50%'
                returnData['task'] = 'Battery Discharged'
                self.discharge(shortageOfEnergy)
                self.logger.info(f"Using the power from the battery...")
            # buy energy from market if battery is 20-50% charged
            elif 0.2 * self.batteryCapacity <= self.amountOfCharge < 0.5 * self.batteryCapacity:
                price = random.random() * 0.1 + 0.1
                amount = self.formulateAmount()
                if amount is False:
                    return returnData
                price = self.convertToWei(price/1000)
                await self.placeBuy(price, math.floor(amount), time_ms)
                returnData['batteryPercentage'] = '20-50%'
                returnData['task'] = 'Send Bid, buying energy'
            # buy from the national grid with emergency situation, less than 20% charged
            else:
                await self.buyFromNationalGrid(0.2 * self.batteryCapacity)
                returnData['batteryPercentage'] = '< 20%'
                returnData['task'] = 'Buy From National Grid'
                
        return returnData
