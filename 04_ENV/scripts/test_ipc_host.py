import sys
import os
sys.path.insert(0, '/home/cse-sdpl/research/ACC/02_SRC')
from acc_core.system.ring_buffer import ACCOpsBridge
import time

print(" ORACLE: Creating Shared Memory...", flush=True)
bridge = ACCOpsBridge(create=True)
print(" ORACLE: Listening for Student updates (Ctrl+C to stop)...", flush=True)

try:
    last_step = -1
    while True:
        ts, flag, step = bridge.read_latest_state()
        if step != last_step and step > 0:
            print(f"    Received Step {step} from Student!", flush=True)
            last_step = step
            if step == 5:
                print("    ORACLE: Triggering Intervention at Step 5!", flush=True)
                bridge.trigger_intervention()
        time.sleep(0.001) # 1ms poll
except KeyboardInterrupt:
    print("\nCleaning up...", flush=True)
    bridge.shm.unlink()
