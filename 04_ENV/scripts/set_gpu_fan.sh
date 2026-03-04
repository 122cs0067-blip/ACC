#!/bin/bash
# GPU THERMAL SAFETY FOR 24/7 ACC EXPERIMENTS
# Enables persistence mode and attempts fan control for RTX 2000 Ada

echo " Checking GPU State..."

# Step 1: Enable Persistence Mode (critical for 24/7 stability)
echo " Enabling Persistence Mode (requires sudo)..."
sudo nvidia-smi -pm 1

# Step 2: Attempt fan control via nvidia-settings
if command -v nvidia-settings &> /dev/null; then
    echo "  Attempting to set Fan Speed to 80%..."
    nvidia-settings -a "[gpu:0]/GPUFanControlState=1" -a "[fan:0]/GPUTargetFanSpeed=80" 2>/dev/null || {
        echo "  Manual fan control not available (X11/coolbits config needed)"
        echo "   → GPU will use auto fan control"
        echo "   → To enable manual control, add 'Option \"Coolbits\" \"4\"' to /etc/X11/xorg.conf.d/20-nvidia.conf"
    }
else
    echo "  nvidia-settings not found. Using auto fan control."
fi

# Step 3: Verify current state
echo ""
echo " Current GPU Status:"
nvidia-smi --query-gpu=name,temperature.gpu,fan.speed,power.draw,power.limit,persistence_mode --format=csv,noheader

echo ""
echo " GPU is ready for 24/7 operation"
echo "   → Monitor thermal during experiments with: watch -n 1 nvidia-smi"
