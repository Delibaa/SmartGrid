# Blockchain-Based Energy Trading Architecture

This project is a prototype implementation of the paper **“A Blockchain-Based Architecture for Energy Trading to Enhance Power Grid Stability.”**

## 📚 Project Summary 
- 📜 Smart Contracts:`contracts/`  
- 📄 Processed Data:`data/`  
  - Data Format (📦 Source-[Sundance](https://traces.cs.umass.edu/docs/traces/smartstar/)):  
    1. Consumer:[time(1-based, 24),demand(kW)]  
    2. Prosumer:[time(1-based, 24),demand(kW),supply(kW)]
  - Original consumption data: `data/consumer(prosumer)/historicaldata`
  - Predicted consumption data:`data/consumer(prosumer)/predictiondata`
  - Real consumption data:`data/consumer(prosumer)/realdata`
- 📂 Simulation Output:`output/`  
- 🧠 Main Simulation Logic:`src/`
  - configuration:`src/sg_config.py`
  - contract compilation & deployment:`src/SmDeployments`
  - Blockchain interaction:`src/Blockchain`
  - EMS models:`src/Model`
  - simulation entry point:`src/server.py`

## ⚡ Quickstart
1️⃣ create a virtual environment (recommended)
```bash
python -m venv .BTA
source .BTA/bin/activate  
python -m pip install -U pip
pip install -e .
```
2️⃣ train models for each endpoint
```bash
python src/Model/trainModel.py
```
3️⃣ start a local Ethereum node
```bash
# use hardhat
npm install --save-dev hardhat    # install Hardhat (if not installed)
npx hardhat                       # create a project (if not created)
npx hardhat node                  # start local blockchain

# use ganache
# install ganache
ganache -p 7545 -i 1337

# revise RPC_URL, PRIVATE_KEY and CHAIN_ID in 'src/sg_config.py' according to your node
```
4️⃣ compile and deploy smart contracts
```bash
python src/SmDeployments/compile.py
python src/SmDeployments/deploy.py
python src/SmDeployments/getOnChainInfo.py
```
5️⃣ run the simulation
```bash
python src/server.py
```
6️⃣ get results visualization   
(You can run these directly, as example data is already provided for the simulation window [91, 105))
```bash
python src/Analysis/plot1.py
python src/Analysis/plot2-1.py
python src/Analysis/plot2-2.py
python src/Analysis/plot3.py
python src/Analysis/plot4.py
```
7️⃣ modify configuration and re-run
```bash
# in src/sg_config.py
CONSUMER_NUMBER = 5
PROSUMER_NUMBER = 10
TIME_WINDOW_START = 91
TIME_WINDOW_END = 105
```
If you change the number of consumers/prosumers:
```bash
# import sundance dataset in dir 'data/rawdata'
python src/DataBean/data_bean.py
```
re-run above steps again
## License
This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

