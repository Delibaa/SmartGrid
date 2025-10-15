# getOnChainInfo.py
# get infomation from the blockchain
from web3 import Web3
import os
import sg_config as Config
from pathlib import Path

# make sure path existed
ARTIFACTS_DIR: Path = Config.ARTIFACTS_DIR
ON_CHAIN_INFO_DIR: Path = Config.ON_CHAIN_INFO_DIR

ON_CHAIN_INFO_DIR.mkdir(parents=True, exist_ok=True)

logger = Config.get_logger(os.path.splitext(os.path.basename(__file__))[0])

# Connect to local Ganache or Hardhat node
w3 = Web3(Web3.HTTPProvider(Config.RPC_URL))
assert w3.is_connected(), "Cannot connect to blockchain"

# Get accounts
accounts = w3.eth.accounts

outputAcc = os.path.join(Config.ON_CHAIN_INFO_DIR, "accounts.txt")

# Write to a TXT file
with open(outputAcc, "w") as f:
    for i, acct in enumerate(accounts):
        f.write(f"Account {i}: {acct}\n")

logger.info("  ✔ Accounts written...")

