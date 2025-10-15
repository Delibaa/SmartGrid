# Blockchain-Based Energy Trading Architecture

This project is a prototype of the paper "A Blockchain-Based Architecture for Energy Trading to Enhance Power Grid Stability", please find more details of this project in this paper if you are interested.

## Project Summary
Store Smart Contracts:
contracts/

Processed Data:
data/

Running result:
output/

Main logic of simulation:
src/

## Quickstart

```bash
# 1. create virtual environment (recommand)
python -m venv .BTA
source .BTA/bin/activate  
python -m pip install -U pip
pip install -e .

# 2. train models for each endpoints
python src/Model/trainModel.py

# 3. install ganache or hardhat node
npx hardhat
# revise RPC_URL, PRIVATE_KEY and CHAIN_ID according to your hardhat node

# 4. compile and deploy contracts
python src/SmDeployments/compile.py
python src/SmDeployments/deploy.py
python src/SmDeployments/getOnChainInfo.py

# 5. start simulation
python src/server.py

```
