# compile.py
# compile all Solidity sources in contracts/ into artifacts/
import json
import os
from pathlib import Path
from solcx import install_solc, set_solc_version, compile_standard
import sg_config as Config

logger = Config.get_logger(os.path.splitext(os.path.basename(__file__))[0])

# ---- Compiler version ----
SOLC_VER = "0.8.30"
install_solc(SOLC_VER)
set_solc_version(SOLC_VER)

CONTRACTS_DIR: Path = Config.CONTRACTS_DIR        # e.g., Path("contracts")
ARTIFACTS_DIR: Path = Config.ARTIFACTS_DIR        # e.g., Path("artifacts_py")
PROJECT_ROOT: Path = getattr(Config, "PROJECT_ROOT", Path.cwd())

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# ---- Gather sources ----
sol_files = sorted(CONTRACTS_DIR.glob("*.sol"))
if not sol_files:
    raise SystemExit("No contract to compile...")

sources = {p.name: {"content": p.read_text(encoding="utf-8")} for p in sol_files}

# ---- Standard JSON input (targets PARIS) for ganache version's deploy ----
std_input = {
    "language": "Solidity",
    "sources": sources,
    "settings": {
        "optimizer": {"enabled": True, "runs": 200},
        "evmVersion": "paris",  # <- ensure no PUSH0 opcode for older nodes
        "outputSelection": {
            "*": {
                "*": [
                    "abi",
                    "evm.bytecode",
                    "evm.deployedBytecode",
                    "metadata",
                ]
            }
        },
    },
}

# ---- Allow import paths if any other module ----
allow_paths = [str(CONTRACTS_DIR)]
node_modules = PROJECT_ROOT / "node_modules"
if node_modules.exists():
    allow_paths.append(str(node_modules))

logger.info(f"Compiling with solc {SOLC_VER} (EVM=paris)...")
res = compile_standard(std_input, allow_paths=",".join(allow_paths))

# ---- Write one artifact per contract ----
written = 0
index_per_source = {}

for src_name, contracts in res["contracts"].items():
    index_per_source[src_name] = []
    for contract_name, compiled in contracts.items():
        abi = compiled.get("abi", [])
        bytecode = (compiled.get("evm", {}).get("bytecode", {}).get("object", "") or "")
        deployed_bytecode = (compiled.get("evm", {}).get("deployedBytecode", {}).get("object", "") or "")

        # Normalize 0x prefix
        if bytecode and not bytecode.startswith("0x"):
            bytecode = "0x" + bytecode
        if deployed_bytecode and not deployed_bytecode.startswith("0x"):
            deployed_bytecode = "0x" + deployed_bytecode

        # artiface
        artifact = {
            "contractName": contract_name,
            "sourceName": src_name,
            "abi": abi,
            "bytecode": bytecode,                   
            "deployedBytecode": deployed_bytecode,  
            "compiler": {"version": SOLC_VER, "evmVersion": "paris"},
        }

        out_path = ARTIFACTS_DIR / f"{contract_name}.json"
        out_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
        index_per_source[src_name].append(contract_name)
        written += 1

        if "__$" in artifact["bytecode"]:
            logger.warning(f"  ⚠ {contract_name} has unlinked libraries (placeholders present).")

        logger.info(f"  ✔ compiled {contract_name} (from {src_name}) -> {out_path.name}")

# Optional: index file
(ARTIFACTS_DIR / "_index.json").write_text(json.dumps(index_per_source, indent=2), encoding="utf-8")

logger.info(f"Done. Wrote {written} artifact(s) to {ARTIFACTS_DIR}")
