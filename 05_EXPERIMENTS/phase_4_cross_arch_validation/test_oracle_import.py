
import sys
from pathlib import Path
import torch

LOG_FILE = "/home/cse-sdpl/research/ACC/06_RESULTS/oracle_import_log.txt"

def log(msg):
    print(msg)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")

if Path(LOG_FILE).exists(): Path(LOG_FILE).unlink()

log("DEBUG: Oracle Import Test Started")

PROJECT_ROOT = Path("/home/cse-sdpl/research/ACC")
sys.path.insert(0, str(PROJECT_ROOT / "02_SRC"))

try:
    log("DEBUG: Attempting to import OracleBridge...")
    from acc_core.system.oracle_bridge import OracleBridge
    log("DEBUG: Import Successful")
    
    log("DEBUG: Initializing OracleBridge (without loading teacher)...")
    bridge = OracleBridge()
    log("DEBUG: Initialization Successful")
except Exception as e:
    import traceback
    log(f"DEBUG: Failed: {e}")
    with open(LOG_FILE, "a") as f:
        traceback.print_exc(file=f)

log("DEBUG: Script Finished")
