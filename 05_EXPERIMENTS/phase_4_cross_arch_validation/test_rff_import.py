
import sys
from pathlib import Path
import torch

LOG_FILE = "/home/cse-sdpl/research/ACC/06_RESULTS/rff_import_log.txt"

def log(msg):
    print(msg)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")

if Path(LOG_FILE).exists(): Path(LOG_FILE).unlink()

log("DEBUG: RFF Import Test Started")

PROJECT_ROOT = Path("/home/cse-sdpl/research/ACC")
sys.path.insert(0, str(PROJECT_ROOT / "02_SRC"))

try:
    log("DEBUG: Attempting to import RandomFourierFeatures...")
    from acc_core.detector.rff_kernel import RandomFourierFeatures
    log("DEBUG: Import Successful")
    
    log("DEBUG: Initializing RFF...")
    rff = RandomFourierFeatures(input_dim=128, rff_dim=512, device='cpu')
    log("DEBUG: Initialization Successful")
except Exception as e:
    import traceback
    log(f"DEBUG: Failed: {e}")
    with open(LOG_FILE, "a") as f:
        traceback.print_exc(file=f)

log("DEBUG: Script Finished")
