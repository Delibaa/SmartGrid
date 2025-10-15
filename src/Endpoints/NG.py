# NG.py
# National Grid agent class
import os
import time
import math
import sg_config as Config
from Blockchain.interact import get_energy_market, get_score_billboard
from EnergyMarketSimulation.score import calculateLeaderForNextDay


# National Grid
class AgentNationalGrid:
    def __init__(self, nationalGridPrice):
        # settings
        self.nationalGridPrice = nationalGridPrice  # in CHF for whatt
        self.ethereumAddress = None

        self.submitLeaderHistory = []

        # contract instance
        self.contract_em = None
        self.contract_sb = None
        self.w3 = None

        # logger
        self.logger = Config.get_logger(
            os.path.splitext(os.path.basename(__file__))[0])

    # get contracts instances
    async def init(self, w3):
        self.w3 = w3
        # Initialize the contracts
        self.contract_em = await get_energy_market(w3)
        self.contract_sb = await get_score_billboard(w3)

    # get account address
    async def getAccount(self):
        accts = await self.w3.eth.accounts
        self.ethereumAddress = accts[len(accts)-1]
        self.logger.info(
            f"Initialize with address of {self.ethereumAddress}...")

    # calculate the leader for the next day
    async def calculateNextLeader(self, day):

        info = await calculateLeaderForNextDay(day)
        houseid = info["leader"]
        err = info["min_error"] * 1000
        try:
            tx_hash = await self.contract_sb.functions.addLeader(math.floor(houseid), math.floor(day), math.floor(err)).transact({
                'from': self.ethereumAddress,
                'gas': 3_000_000
            })
            receipt = await self.w3.eth.wait_for_transaction_receipt(tx_hash)
            transactionReceipt = receipt
        except Exception as err:
            self.logger.error(f"Error in submitting leader: {err}")

        # record the bid
        date = int(time.time())
        gasUsed = transactionReceipt['gasUsed'] if transactionReceipt and transactionReceipt['gasUsed'] else 0
        newLeader = {
            'address': self.ethereumAddress,
            'houseId': houseid,
            'err': info["min_error"],
            'date': date,
            'day': day,
            'transactionCost': gasUsed,
            'evaluated': info["evaluated"]
        }
        self.submitLeaderHistory.append(newLeader)
        self.logger.info(
            f"Send a leader to the score billboard...")
        return True

    async def clearScoreBillboard(self):
        try:
            tx_hash = await self.contract_sb.functions.clearLeaders().transact({
                'from': self.ethereumAddress,
                'gas': 90_000_000
            })
            receipt = await self.w3.eth.wait_for_transaction_receipt(tx_hash)
            transactionReceipt = receipt
        except Exception as err:
            self.logger.error(f"Error in clearing score billboard: {err}")

        # record the bid
        date = int(time.time())
        gasUsed = transactionReceipt['gasUsed'] if transactionReceipt and transactionReceipt['gasUsed'] else 0
        newLeader = {
            'address': self.ethereumAddress,
            'date': date,
            'transactionCost': gasUsed,
        }
        self.logger.info(
            f"Clear the score billboard...")
        return True

