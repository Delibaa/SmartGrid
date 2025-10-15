# interact.py
# instance to interact with deployed contract
import asyncio
import json
import sg_config as Config  # must define RPC_URL, ARTIFACTS_DIR, ON_CHAIN_INFO_DIR

# Lazy, cached contract instance + lock to avoid race on first load
contract_em = None
_em_lock = asyncio.Lock()

contract_sb = None
_sb_lock = asyncio.Lock()

# get instance
async def get_energy_market(w3):
    global contract_em
    if contract_em is None:
        async with _em_lock:
            if contract_em is None:  # double-check inside lock
                name = "EnergyMarket"
                abi = json.loads((Config.ARTIFACTS_DIR / f"{name}.json").read_text())["abi"]
                address = (Config.ON_CHAIN_INFO_DIR / f"{name}.address").read_text().strip()
                contract_em = w3.eth.contract(
                    address=w3.to_checksum_address(address),
                    abi=abi,
                )
    return contract_em

# get instance
async def get_score_billboard(w3):
    global contract_sb
    if contract_sb is None:
        async with _sb_lock:
            if contract_sb is None:  # double-check inside lock
                name = "ScoreBillboard"
                abi = json.loads((Config.ARTIFACTS_DIR / f"{name}.json").read_text())["abi"]
                address = (Config.ON_CHAIN_INFO_DIR / f"{name}.address").read_text().strip()
                contract_sb = w3.eth.contract(
                    address=w3.to_checksum_address(address),
                    abi=abi,
                )
    return contract_sb

# If you redeploy, call this to drop the cache and reload.
async def reload_energy_market(w3):
    global contract_em
    async with _em_lock:
        contract_em = None
    return await get_energy_market(w3)

async def reload_score_billboard(w3):
    global contract_sb
    async with _sb_lock:
        contract_sb = None
    return await get_score_billboard(w3)
