# simulation server
import EnergyMarketSimulation.emSimulation as emSimulation
import asyncio
import os
import glob 
import shutil
import sg_config as Config

# delete historical data
DATA_DIRS = [
    os.path.join(Config.DATA_DIR, "consumer", "predictiondata"),
    os.path.join(Config.DATA_DIR, "consumer", "realdata"),
    os.path.join(Config.DATA_DIR, "prosumer", "predictiondata"),
    os.path.join(Config.DATA_DIR, "prosumer", "realdata"),
    Config.OUTPUTS_DIR
]

# run the whole server for simulation
async def server():
    # clear data
    for dir_path in DATA_DIRS:
        if os.path.exists(dir_path):
            files = glob.glob(os.path.join(dir_path, "*"))
            for f in files:
                try:
                    os.remove(f)
                    print(f"Deleted: {f}")
                except IsADirectoryError:
                    # subdirectory
                    shutil.rmtree(f)
                    print(f"Deleted directory: {f}")
        else:
            print(f"Path not found: {dir_path}")
    
    # start energy market simulation
    try:
        await emSimulation.energyMarketSimulation()
    finally:
        await emSimulation.shutdown()

if __name__ == "__main__":
    asyncio.run(server())