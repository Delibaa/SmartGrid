# consumer.py
# consumer agent class, onchain unit, whatt/amount and wei/price
import time
import os
import csv
import math
import random
import pandas as pd
import sg_config as Config
from Blockchain.interact import get_energy_market
from Endpoints.AIAgent import consumption_prediction

# consumer
class AgentConsumer:
    def __init__(self, batteryCapacity, houseID):
        # account settings
        self.householdID = houseID  # household ID in the smart grid, number from 0
        self.ethereumAddress = None  # ethereum address of the prosumer
        self.balance = 0  # balance of the prosumer in wei

        # simulation settings
        self.timeRow = 0  # current time row in the historic data
        self.day = 0  # current day
        self.hasBattery = True  # consumer also with a battery to deal with instant requirements
        self.batteryCapacity = batteryCapacity  # maximum in whatt
        self.amountOfCharge = batteryCapacity  # current amount of charge in whatt

        # current consumption status, single time row
        self.shortageEnergy = 0  # whether there is shortage of energy in whatt at this time row
        # current demand in whatt at this time row, {'time': int, 'demand': float}
        self.currentDemand = 0

        # global settings of the national grid
        self.nationalGridAddress = None  # ethereum address of the national grid
        self.nationalGridPrice = 0  # in wei/whatt

        # history settings
        # list of dict {'time': int, 'demand': float}
        self.historicalDemand = []
        # list of float, historical cleared prices at each day
        self.historicalClearPrices = [0] * Config.TIME_WINDOW_END
        # list of float, historical trading price at each time point
        self.historicalTradePrices = []
        # list of dict {'transactionReceipt': receipt, 'transactionCost': int, 'transactionAmount': int, 'date': timestamp, 'quantity': float, 'timeRow': int}
        self.nationalGridPurchases = []
        # list of dict {'address': addr, 'price': int, 'amount': int, 'date': timestamp, 'timeRow': int, 'transactionCost': int}
        self.bidHistory = []  # onchain only with int format
        # list of dict {'transactionReceipt': receipt, 'transactionCost': int, 'transactionAmount': int, 'date': timestamp, 'quantity': float, 'timeRow': int}
        self.successfulBidHistory = []

        # endpoints control
        # household machine switch (those can be trurned off/on by the agent)
        # starting with false since data starts at 00:00 of one day
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

        # contracts instance
        self.contract_em = None
        self.w3 = None

        # log
        self.logger = Config.get_logger(os.path.splitext(
            os.path.basename(__file__))[0] + str(self.householdID))

    # get contracts instance
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
            self.historicalDemand.append(currentDemand)

        # get logger
        self.logger.info(f"  ✔ Consumer created successfully...")
        return True

    # get ethereum account
    async def getAccount(self, index):
        accounts = await self.w3.eth.accounts
        self.ethereumAddress = accounts[index]
        self.logger.info(
            f"  ✔ Initialize with address of {self.ethereumAddress}...")

    # get balance of the account
    async def getAgentBalance(self):
        balance = await self.w3.eth.get_balance(self.ethereumAddress)
        self.balance = balance

    # get data dir
    def getHistoricalDataDir(self):
        historicalDataDir = os.path.join(
            Config.DATA_DIR, "consumer", "historicaldata", f"house_{self.householdID}.csv")
        return historicalDataDir

    def getPredictedDataDir(self):
        predictedDataDir = os.path.join(
            Config.DATA_DIR, "consumer", "predictiondata", f"house_{self.householdID}.csv")
        return predictedDataDir

    def getRealDataDir(self):
        realDataDir = os.path.join(
            Config.DATA_DIR, "consumer", "realdata", f"house_{self.householdID}.csv")
        return realDataDir

    # convert price in CHF to wei, onchain unit-wei
    def convertToWei(self, price):
        calcPrice_eth = round(price / Config.PRICE_OF_ETHER, 18)
        try:
            priceWei = self.w3.to_wei(calcPrice_eth, 'ether')
        except Exception as err:
            self.logger.error(f"Error from conversion: {err}")
            priceWei = 0
        priceWei = int(priceWei)  # onchain unit wei in integer format
        return priceWei

    # set national grid info
    def setNationalGrid(self, nationalGridPrice, nationalGridAddress):
        self.nationalGridAddress = nationalGridAddress
        price = round(nationalGridPrice / 1000.0, 18)
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
        self.logger.info(
            f"Buying power {amount} whatt from the national grid...")
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
        self.logger.info(
            f"Buying power {amount} whatt from the prosumer {receiver}...")
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
        self.logger.info(
            f"Send a bid with {amount} whatt to the energy market...")
        return True

    # place bid to buy energy from the market
    async def delBuy(self, orderId):
        try:
            await self.contract_em.functions.removeBid(orderId).transact({
                'from': self.ethereumAddress,
                'gas': 3_000_000
            })
        except Exception as err:
            self.logger.error(f"Error in placeBuy: {err}")
        self.logger.info(
            f"Delete the bid {orderId.hex()} in the energy market...")

    # revise bid
    async def reviseBuy(self, amount, orderId):
        try:
            await self.contract_em.functions.updateBidAmount(orderId, amount).transact({
                'from': self.ethereumAddress,
                'gas': 3_000_000
            })
        except Exception as err:
            self.logger.error(f"Error in placeBuy: {err}")
        self.logger.info(
            f"Revise the bid {orderId.hex()}'s amount in the energy market...")

    # charge battery

    def charge(self, amount):
        self.amountOfCharge += amount
        if self.amountOfCharge > self.batteryCapacity:
            self.logger.info("Battery has been fully charged...")
            self.amountOfCharge = self.batteryCapacity

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
        self.logger.info(
            f"Store the real consumption of day {self.day} into database...")

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
        self.logger.info(
            f"Store the predictive consumption of day {predictionDay} into database...")

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

    # main logic of the consumer agent, when to act with energy market, when to buy, when to sell
    async def decisionMaker(self):
        # demand and supply status at the current time row
        idx = (self.timeRow % 24) - 1
        pd = self.predictionNextDay[idx]
        d = self.historicalDemand[self.timeRow - 1]['demand']
        demand = self.selfManagement(idx, pd, d)

        self.realConsumption[idx] = demand  # store real consumption
        shortageOfEnergy = demand

        time_ms = int(time.time())  # date
        price = 0

        # return to API of UI
        returnData = {}
        returnData['demand'] = demand
        returnData['shortageEnergy'] = shortageOfEnergy
        returnData['hasBattery'] = self.hasBattery

        # decision logic
        if shortageOfEnergy > 0:
            # use power from the battery if more than 50% charged
            if self.amountOfCharge >= 0.5 * self.batteryCapacity:
                returnData['batteryPercentage'] = '> 50%'
                returnData['task'] = 'Battery Discharged'
                self.discharge(shortageOfEnergy)
                self.logger.info(f"Using power from the battery...")
            # send bids for the energy markets if 20%-50% charged
            elif 0.2 * self.batteryCapacity < self.amountOfCharge < 0.5 * self.batteryCapacity:
                price = random.random() * 0.1 + 0.1
                price = self.convertToWei(price/1000)
                await self.placeBuy(math.floor(price), math.floor(
                    shortageOfEnergy), time_ms)
                returnData['batteryPercentage'] = '20-50%'
                returnData['task'] = 'Send Bid, buying energy'
            # emergency situation
            else:
                await self.buyFromNationalGrid(0.2 * self.batteryCapacity)
                returnData['batteryPercentage'] = '< 20%'
                returnData['task'] = 'Buy From National Grid'

        return returnData
