
import sys
import time

LOG_FILE = "/home/cse-sdpl/research/ACC/06_RESULTS/ipex_import_log.txt"

def log(msg):
    print(msg)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")

log("DEBUG: IPEX Import Test Started")

try:
    log("DEBUG: Attempting to import intel_extension_for_pytorch...")
    import intel_extension_for_pytorch as ipex
    log("DEBUG: Import Successful")
except Exception as e:
    log(f"DEBUG: Failed: {e}")

log("DEBUG: Script Finished")
