import sys
import numpy as np
import time
# Mock path addition since we mount the source code
sys.path.append('/app/src') 
from acc_core.system.ring_buffer import ACCOpsBridge

print(" STUDENT: Connecting to Shared Memory...")
# Wait for Oracle to create it
time.sleep(2) 
bridge = ACCOpsBridge(create=False)

for i in range(1, 10):
    dummy_vector = np.random.randn(128).astype(np.float32)
    print(f" STUDENT: Generating Step {i}...")
    bridge.write_state(i, dummy_vector)
    
    # Check for intervention
    if bridge.check_for_intervention():
        print(f"    INTERRUPT RECEIVED at Step {i}! Pausing...")
        break
    time.sleep(0.5)

bridge.close()
