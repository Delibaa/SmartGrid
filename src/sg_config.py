# global configuration settings
from pathlib import Path
import logging
import os

# Blockchain settings
RPC_URL = "http://127.0.0.1:7545"
# this address is used to deploy contracts
PRIVATE_KEY = "" 
CHAIN_ID = 1337

# Directory paths
BASE_DIR = Path(__file__).resolve().parents[1]
CONTRACTS_DIR = BASE_DIR / "contracts"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "src" / "Model" / "trainedModels"
ON_CHAIN_INFO_DIR = BASE_DIR / "on_chain_info"
OUTPUTS_DIR = BASE_DIR / "output"
LOGS_DIR = BASE_DIR / "logs"
LOG_FILE = os.path.join(LOGS_DIR, "SmartGrid.log")

# Energy market settings
PRICE_OF_ETHER = 3500          # CHF/Ether
WEI_IN_ETHER = 10**18          # 1 ether = 1e18 wei
GAS_PRICE = 30_000_000_000      # 30 gwei
MAX_BATTERY_CAPACITY = 15000      # Whatt
# CHF/kWh, for running solidity, using whatt and wei for on-chain records
NATIONAL_GRID_PRICE = 0.25

# Simulation settings
CONSUMER_NUMBER = 5
PROSUMER_NUMBER = 10
TIME_WINDOW_START = 91
TIME_WINDOW_END = 105

# logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ]
)

# logger
def get_logger(name=None):
    return logging.getLogger(name)
