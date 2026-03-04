import importlib.util
import sys
from pathlib import Path

# Add project root to sys.path
root = Path("/home/cse-sdpl/research/ACC")
sys.path.append(str(root))

# Dynamic import to bypass digit-prefixed folder issue
spec = importlib.util.spec_from_file_location(
    "benchmark_loaders_vlm", 
    root / "05_EXPERIMENTS/phase_4_cross_arch_validation/benchmark_loaders_vlm.py"
)
loaders_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loaders_mod)
load_vlm_benchmark = loaders_mod.load_vlm_benchmark

def test_loaders():
    bench_dir = Path("/home/cse-sdpl/research/ACC/01_DATA/Benchmarks")
    
    for bench in ["mathvista", "vqav2", "pope"]:
        print(f"\nTesting {bench} loader...")
        try:
            samples = load_vlm_benchmark(bench, bench_dir / bench, num_samples=5)
            print(f"Successfully loaded {len(samples)} samples.")
            for i, s in enumerate(samples):
                print(f"  Sample {i}: id={s.sample_id}, question='{s.question[:50]}...', image={s.image}")
                if not Path(s.image).exists():
                    print(f"    [ERROR] Image path does not exist: {s.image}")
                else:
                    print(f"    [OK] Image exists.")
        except Exception as e:
            print(f"  [FAILED] {bench} loader error: {e}")

if __name__ == "__main__":
    test_loaders()
