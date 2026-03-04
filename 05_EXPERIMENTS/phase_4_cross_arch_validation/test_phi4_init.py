
import sys
from pathlib import Path
import torch

LOG_FILE = "/home/cse-sdpl/research/ACC/06_RESULTS/phi4_init_log.txt"

def log(msg):
    print(msg)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")

if Path(LOG_FILE).exists(): Path(LOG_FILE).unlink()

log("DEBUG: Phi-4 Test Started")

PROJECT_ROOT = Path("/home/cse-sdpl/research/ACC")
sys.path.insert(0, str(PROJECT_ROOT / "02_SRC"))

from wrappers.wrapper_phi4_multimodal import Phi4MultimodalStudentAgent

try:
    log("DEBUG: Initializing Phi4MultimodalStudentAgent...")
    agent = Phi4MultimodalStudentAgent(use_teacher=False)
    log("DEBUG: Success!")
    agent.unload_model()
except Exception as e:
    import traceback
    log(f"DEBUG: Failed: {e}")
    with open(LOG_FILE, "a") as f:
        traceback.print_exc(file=f)
log("DEBUG: Script finished")
