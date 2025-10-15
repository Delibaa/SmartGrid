# deploy.py
# deploy all zero-arg-ctor contracts from artifacts/
import json
import os
import re
from pathlib import Path
from web3 import Web3
from eth_account import Account
import sg_config as Config

logger = Config.get_logger(os.path.splitext(os.path.basename(__file__))[0])

# make sure path existed
ARTIFACTS_DIR: Path = Config.ARTIFACTS_DIR
ON_CHAIN_INFO_DIR: Path = Config.ON_CHAIN_INFO_DIR

ON_CHAIN_INFO_DIR.mkdir(parents=True, exist_ok=True)

def ctor_meta(abi):
    ctor = next((i for i in abi if i.get("type") == "constructor"), None)
    if not ctor:
        return [], "nonpayable"
    return ctor.get("inputs", []), ctor.get("stateMutability", "nonpayable")

def has_unlinked_libs(bytecode):
    return "__$" in (bytecode or "")

def main():
    w3 = Web3(Web3.HTTPProvider(Config.RPC_URL))
    assert w3.is_connected(), "Cannot connect to RPC"
    acct = Account.from_key(Config.PRIVATE_KEY)

    artifacts = sorted(ARTIFACTS_DIR.glob("*.json"))
    if not artifacts:
        raise SystemExit(f"No artifacts in {ARTIFACTS_DIR}")

    logger.info(f"Deploying with {acct.address} (chainId={Config.CHAIN_ID})")
    nonce = w3.eth.get_transaction_count(acct.address)

    for art in artifacts:
        if art.name == "_index.json":
            continue

        data = json.loads(art.read_text(encoding="utf-8"))

        # Two shapes supported:
        # A) single-contract artifact (Hardhat-like): {"contractName","abi","bytecode",...}
        # B) map of contracts: {Name:{abi,bytecode}, Name2:{...}}
        if "abi" in data and "bytecode" in data:
            candidates = [(data.get("contractName") or art.stem, data)]
        else:
            candidates = list(data.items())

        for contract_name, obj in candidates:
            abi = obj["abi"]
            bytecode = obj["bytecode"]

            if not bytecode or bytecode == "0x":
                logger.info(f"  ⤷ skip {contract_name} (no bytecode)")
                continue

            # ouput whether constructor need to pay (if with parameters)
            inputs, state_mut = ctor_meta(abi)
            logger.info(f"  • {contract_name} ctor inputs={inputs} mutability={state_mut}")

            if inputs:
                logger.info(f"  ⤷ skip {contract_name} (constructor args required)")
                continue

            if has_unlinked_libs(bytecode):
                placeholders = set(re.findall(r"__\$\w{34}\$__", bytecode))
                logger.error(f"  ✖ {contract_name} has unlinked libraries: {placeholders}. Link before deploy.")
                continue

            Contract = w3.eth.contract(abi=abi, bytecode=bytecode)

            tx_opts = {"from": acct.address}
            if state_mut == "payable":
                # Adjust value if your ctor requires a specific amount
                tx_opts["value"] = w3.to_wei("0.01", "ether")

            # Try gas estimate (surfaces many ctor issues)
            try:
                est = Contract.constructor().estimate_gas(tx_opts)
                gas_limit = int(est * 1.2)
                logger.info(f"  • gas estimate {contract_name}: {est} (using {gas_limit})")
            except Exception as e:
                logger.error(f"  ✖ gas estimate failed for {contract_name}: {e}")
                continue

            tx = Contract.constructor().build_transaction({
                **tx_opts,
                "nonce": nonce,
                "chainId": Config.CHAIN_ID,
                # EIP-1559 fields (works on Hardhat/Ganache recent):
                "maxFeePerGas": w3.to_wei("30", "gwei"),
                "maxPriorityFeePerGas": w3.to_wei("1.5", "gwei"),
                "gas": gas_limit,
            })
            nonce += 1

            signed = w3.eth.account.sign_transaction(tx, private_key=Config.PRIVATE_KEY)
            tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
            rcpt = w3.eth.wait_for_transaction_receipt(tx_hash)

            (ON_CHAIN_INFO_DIR / f"{contract_name}.address").write_text(rcpt.contractAddress)
            logger.info(f"  ✔ deployed {contract_name} -> {rcpt.contractAddress}")

    logger.info("All done.")

if __name__ == "__main__":
    main()
