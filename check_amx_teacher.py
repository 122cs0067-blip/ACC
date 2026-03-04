#!/usr/bin/env python3
"""
AMX Teacher Verification Script for UAI 2026 (IPEX-Safe Version)

Tests IPEX/AMX availability across different conda environments.
Handles version mismatches gracefully and provides recommendations.
"""

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os

# Try to import IPEX, but don't fail if unavailable
IPEX_AVAILABLE = False
try:
    import intel_extension_for_pytorch as ipex
    IPEX_AVAILABLE = True
    print(f"[OK] IPEX loaded successfully (version {ipex.__version__})")
except Exception as e:
    print(f"[ALERT]️  IPEX not available: {e}")
    print("   Will run without AMX acceleration")


def check_amx_support():
    """Verify that AMX is available on this CPU."""
    print("\n" + "=" * 80)
    print("STEP 1: Checking AMX Hardware Support")
    print("=" * 80)
    
    try:
        with open('/proc/cpuinfo', 'r') as f:
            cpuinfo = f.read()
            has_amx_bf16 = 'amx_bf16' in cpuinfo
            has_amx_int8 = 'amx_int8' in cpuinfo
            has_amx_tile = 'amx_tile' in cpuinfo
            
        print(f"✓ AMX-BF16 support: {has_amx_bf16}")
        print(f"✓ AMX-INT8 support: {has_amx_int8}")
        print(f"✓ AMX-TILE support: {has_amx_tile}")
        
        if not (has_amx_bf16 and has_amx_tile):
            print("\n[ALERT]️  WARNING: AMX not fully supported on this CPU!")
            return False
        
        print("\n[OK] AMX hardware support confirmed!")
        return True
        
    except Exception as e:
        print(f"[ERROR] Could not verify AMX support: {e}")
        return False


def check_ipex_amx():
    """Verify that IPEX can utilize AMX."""
    print("\n" + "=" * 80)
    print("STEP 2: Checking IPEX AMX Integration")
    print("=" * 80)
    
    if not IPEX_AVAILABLE:
        print("[ERROR] IPEX not available")
        print(f"   PyTorch version: {torch.__version__}")
        print("\n[ALERT]️  RECOMMENDATIONS:")
        print("   1. Try different conda environment (see test_ipex_envs.sh)")
        print("   2. Use GPU for teacher (RTX 2000 Ada)")
        print("   3. Accept slower CPU inference without AMX")
        return False
    
    print(f"[OK] IPEX version: {ipex.__version__}")
    print(f"[OK] PyTorch version: {torch.__version__}")
    
    # Test AMX with simple matmul
    try:
        a = torch.randn(128, 128, dtype=torch.bfloat16)
        b = torch.randn(128, 128, dtype=torch.bfloat16)
        
        with torch.cpu.amp.autocast(dtype=torch.bfloat16):
            t0 = time.perf_counter()
            c = torch.matmul(a, b)
            t1 = time.perf_counter()
        
        latency_ms = (t1 - t0) * 1000
        print(f"\n✓ BF16 matmul (128x128): {latency_ms:.2f}ms")
        
        if latency_ms < 1.0:
            print("[OK] IPEX AMX acceleration active!")
            return True
        else:
            print("[ALERT]️  Slow matmul - AMX may not be active")
            return False
            
    except Exception as e:
        print(f"[ERROR] IPEX AMX test failed: {e}")
        return False


def benchmark_teacher_simple(model_name="meta-llama/Llama-3.1-8B-Instruct"):
    """Quick benchmark of teacher model (1 token generation)."""
    print("\n" + "=" * 80)
    print(f"STEP 3: Quick Teacher Benchmark")
    print("=" * 80)
    print(f"Model: {model_name}")
    
    try:
        print("\nLoading model...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        
        # Apply IPEX optimization if available
        if IPEX_AVAILABLE:
            print("✓ Applying IPEX optimization...")
            model = ipex.optimize(model, dtype=torch.bfloat16)
        else:
            print("[ALERT]️  Running without IPEX (slower)")
        
        model.eval()
        
        # Test prompt
        prompt = "The capital of France is"
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # Warmup
        print("\nWarmup (3 iterations)...")
        with torch.no_grad(), torch.cpu.amp.autocast(dtype=torch.bfloat16):
            for _ in range(3):
                _ = model.generate(**inputs, max_new_tokens=1, do_sample=False)
        
        # Benchmark
        print("Benchmarking (5 iterations)...")
        latencies = []
        
        with torch.no_grad(), torch.cpu.amp.autocast(dtype=torch.bfloat16):
            for i in range(5):
                t0 = time.perf_counter()
                output = model.generate(**inputs, max_new_tokens=1, do_sample=False)
                t1 = time.perf_counter()
                
                latency_ms = (t1 - t0) * 1000
                latencies.append(latency_ms)
                print(f"  Iteration {i+1}: {latency_ms:.2f}ms")
        
        avg_latency = sum(latencies) / len(latencies)
        
        # Results
        print("\n" + "=" * 80)
        print("RESULTS")
        print("=" * 80)
        print(f"Average latency: {avg_latency:.2f}ms")
        
        generated = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"Generated: {generated}")
        
        # Verdict
        print("\n" + "=" * 80)
        print("VERDICT")
        print("=" * 80)
        
        if not IPEX_AVAILABLE:
            print(f"[ALERT]️  NO IPEX: {avg_latency:.2f}ms (baseline CPU)")
            print("   → Try different conda environment for IPEX support")
            return False, avg_latency
        elif avg_latency < 100:
            print(f"[OK] PASS: {avg_latency:.2f}ms (AMX-accelerated)")
            return True, avg_latency
        else:
            print(f"[ALERT]️  SLOW: {avg_latency:.2f}ms (target: <100ms)")
            return False, avg_latency
            
    except Exception as e:
        print(f"\n[ERROR] ERROR: {e}")
        return False, None


def main():
    """Run AMX verification."""
    print("\n" + "=" * 80)
    print("UAI 2026 AMX TEACHER VERIFICATION")
    print("=" * 80)
    print(f"PyTorch: {torch.__version__}")
    print(f"IPEX Available: {IPEX_AVAILABLE}")
    
    # Check hardware
    amx_hw = check_amx_support()
    
    # Check IPEX
    ipex_ok = check_ipex_amx()
    
    # Quick benchmark
    success, latency = benchmark_teacher_simple()
    
    # Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"AMX Hardware: {'[OK]' if amx_hw else '[ERROR]'}")
    print(f"IPEX Available: {'[OK]' if IPEX_AVAILABLE else '[ERROR]'}")
    print(f"IPEX Working: {'[OK]' if ipex_ok else '[ERROR]'}")
    
    if latency:
        print(f"Teacher Latency: {latency:.2f}ms")
    
    if success:
        print("\n🎉 READY FOR UAI 2026 CAMPAIGN!")
        print("   AMX acceleration confirmed.")
    else:
        print("\n[ALERT]️  AMX NOT OPTIMAL")
        print("\n   NEXT STEPS:")
        print("   1. Run: bash test_ipex_envs.sh")
        print("   2. Or use GPU for teacher inference")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
