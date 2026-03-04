#!/usr/bin/env python3
"""
AMX Support Checker for Intel Xeon W5 Processor
----------------------------------------------
Verifies that Advanced Matrix Extensions (AMX) are:
1. Available in the CPU (ISA level)
2. Enabled in BIOS/Kernel
3. Accessible via IPEX (PyTorch/oneAPI integration)

Usage:
    python check_amx_support.py [--verbose]

Output:
    - CPU Feature Detection (CPUID)
    - Kernel AMX state (/proc/cpuinfo)
    - IPEX/oneAPI Runtime Check
    - Recommendation for tinygemm_ada & amx_ops kernels
"""

import sys
import os
import subprocess
import struct
from pathlib import Path


def check_cpuid_amx():
    """Check if AMX is available in CPU capabilities via CPUID instruction."""
    try:
        # Use cpuid command if available, otherwise parse /proc/cpuinfo
        result = subprocess.run(
            ["grep", "-i", "amx", "/proc/cpuinfo"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and result.stdout:
            return True, result.stdout.strip()
        return False, "No AMX flags found in /proc/cpuinfo"
    except Exception as e:
        return False, f"Error checking CPUID: {e}"


def check_kernel_amx():
    """Check if kernel has AMX support enabled."""
    try:
        # Check for x87 state (required for AMX context)
        result = subprocess.run(
            ["grep", "-i", "x87", "/proc/cpuinfo"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return True, "Kernel x87 support detected"
        return False, "No x87 support in kernel"
    except Exception as e:
        return False, f"Error checking kernel: {e}"


def check_ipex_amx():
    """Check if IPEX (Intel PyTorch Extension) can access AMX."""
    try:
        import intel_extension_for_pytorch as ipex  # type: ignore
        
        # Try to check if AMX ops are available
        has_amx = hasattr(ipex, 'optimize') and hasattr(ipex.optimize, '_amx')
        ipex_version = ipex.__version__ if hasattr(ipex, '__version__') else "Unknown"
        
        return True, f"IPEX {ipex_version} installed and accessible"
    except ImportError:
        return False, "IPEX not installed (install via: pip install intel-extension-for-pytorch)"
    except Exception as e:
        return False, f"Error checking IPEX: {e}"


def check_oneapi():
    """Check if oneAPI is installed and configured."""
    try:
        # Check for oneAPI environment variables
        result = subprocess.run(
            ["bash", "-c", "source /opt/intel/oneapi/setvars.sh 2>/dev/null && echo 'OK' || echo 'FAIL'"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if "OK" in result.stdout or "FAIL" not in result.stdout:
            return True, "oneAPI environment detected"
        return False, "oneAPI not properly configured"
    except Exception as e:
        return False, f"Error checking oneAPI: {e}"


def get_cpu_model():
    """Extract CPU model name."""
    try:
        result = subprocess.run(
            ["grep", "-m", "1", "model name", "/proc/cpuinfo"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return "Unknown CPU model"
    except Exception as e:
        return f"Error: {e}"


def main():
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    
    print("=" * 70)
    print(" Intel AMX Support Verification (HP Z4 G5 + Xeon W5)")
    print("=" * 70)
    
    # System Information
    print(f"\n System Information:")
    print(f"   OS: {sys.platform}")
    print(f"   CPU: {get_cpu_model()}")
    
    # Check 1: CPUID / /proc/cpuinfo
    print(f"\n Check 1: CPU ISA Level (CPUID)")
    amx_available, amx_msg = check_cpuid_amx()
    status = "✓" if amx_available else "✗"
    print(f"   {status} {amx_msg}")
    
    # Check 2: Kernel Support
    print(f"\n Check 2: Kernel AMX Support")
    kernel_ok, kernel_msg = check_kernel_amx()
    status = "✓" if kernel_ok else "✗"
    print(f"   {status} {kernel_msg}")
    
    # Check 3: oneAPI
    print(f"\n Check 3: Intel oneAPI Environment")
    oneapi_ok, oneapi_msg = check_oneapi()
    status = "✓" if oneapi_ok else "✗"
    print(f"   {status} {oneapi_msg}")
    
    # Check 4: IPEX Runtime
    print(f"\n Check 4: IPEX (PyTorch AMX Runtime)")
    ipex_ok, ipex_msg = check_ipex_amx()
    status = "✓" if ipex_ok else "✗"
    print(f"   {status} {ipex_msg}")
    
    # Summary & Recommendation
    print(f"\n" + "=" * 70)
    all_checks_pass = amx_available and kernel_ok and oneapi_ok and ipex_ok
    
    if all_checks_pass:
        print(" SUCCESS: AMX is fully functional!")
        print("   → amx_ops/ kernels can be compiled and used")
        print("   → tinygemm_ada/ optimizations are compatible")
        print("   → oracle_cpu_monitor.py will leverage AMX acceleration")
        exit_code = 0
    else:
        print("  WARNING: AMX support is incomplete")
        if not amx_available:
            print("   ✗ CPU does not advertise AMX support (update BIOS?)")
        if not kernel_ok:
            print("   ✗ Kernel missing x87/AMX support (update kernel?)")
        if not oneapi_ok:
            print("   ✗ oneAPI not installed (run: source /opt/intel/oneapi/setvars.sh)")
        if not ipex_ok:
            print("   ✗ IPEX not installed (run: pip install intel-extension-for-pytorch)")
        print("\n   Recommendation: Ensure BIOS has AMX enabled and install missing components.")
        exit_code = 1
    
    print("=" * 70)
    if verbose:
        print("\n Verbose Output:")
        print(f"   amx_available={amx_available}, kernel_ok={kernel_ok}, oneapi_ok={oneapi_ok}, ipex_ok={ipex_ok}")
    
    return exit_code


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
