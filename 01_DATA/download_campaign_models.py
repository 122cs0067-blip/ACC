#!/usr/bin/env python3
"""
ACC campaign — Model Download Script
======================================
Downloads all VLM student models (4-bit) and teacher oracle (BF16)
into /home/cse-sdpl/research/ACC/01_DATA/models/

Student Fleet (GPU, 4-bit):
  - Qwen2.5-VL-3B-Instruct   → student_vlm/qwen25vl_3b/
  - Phi-4-Multimodal         → student_vlm/phi4_multimodal/
  - LLaVA-v1.6-Vicuna-7B     → student_vlm/llava16_7b/

Teacher Oracle (CPU BF16):
  - Llama-3.2-Vision-11B     → teacher_vlm/llama32_vision_11b/

Also downloads VLM benchmarks into 01_DATA/Benchmarks/:
  - MathVista                → Benchmarks/mathvista/
  - VQAv2 (val2014 subset)   → Benchmarks/vqav2/
  - POPE                     → Benchmarks/pope/

Usage:
  source /home/cse-sdpl/research/ACC/.venv/bin/activate
  export HF_TOKEN=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
  python /home/cse-sdpl/research/ACC/01_DATA/download_campaign_models.py
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download, login

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
ROOT          = Path("/home/cse-sdpl/research/ACC/01_DATA")
STUDENT_DIR   = ROOT / "models" / "student_vlm"
TEACHER_DIR   = ROOT / "models" / "teacher_vlm"
BENCH_DIR     = ROOT / "Benchmarks"

HF_TOKEN      = os.environ.get("HF_TOKEN", "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")

# ─────────────────────────────────────────────────────────────────────────────
# Model Registry
# ─────────────────────────────────────────────────────────────────────────────
STUDENT_MODELS = {
    "qwen25vl_3b": {
        "repo_id":  "Qwen/Qwen2.5-VL-3B-Instruct",
        "local_dir": STUDENT_DIR / "qwen25vl_3b",
        "est_gb":   6.5,
        "notes":    "Edge tier — NF4 quantized at inference time",
    },
    "phi4_multimodal": {
        "repo_id":  "microsoft/Phi-4-multimodal-instruct",
        "local_dir": STUDENT_DIR / "phi4_multimodal",
        "est_gb":   8.5,
        "notes":    "Logic tier — AWQ quantized at inference time",
    },
    "llava16_7b": {
        "repo_id":  "llava-hf/llava-v1.6-vicuna-7b-hf",
        "local_dir": STUDENT_DIR / "llava16_7b",
        "est_gb":   14.0,
        "notes":    "Standard tier — NF4 quantized at inference time",
    },
}

TEACHER_MODELS = {
    "llama32_vision_11b": {
        "repo_id":  "meta-llama/Llama-3.2-11B-Vision-Instruct",
        "local_dir": TEACHER_DIR / "llama32_vision_11b",
        "est_gb":   22.6,
        "notes":    "Oracle — BF16 CPU inference via Intel AMX",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Benchmark Download (HuggingFace datasets)
# ─────────────────────────────────────────────────────────────────────────────
def download_benchmarks(max_samples: dict = None):
    """Download VLM benchmarks and save as JSON lists for the campaign."""
    if max_samples is None:
        max_samples = {"mathvista": 500, "vqav2": 1000, "pope": 1000}

    try:
        from datasets import load_dataset
        import PIL.Image
    except ImportError:
        print("[WARN] `datasets` or `PIL` not installed — skipping benchmark download.")
        return

    # ── MathVista ──────────────────────────────────────────────────────────
    out = BENCH_DIR / "mathvista"
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    out_file = out / "mathvista_testmini.json"
    
    # Remove potentially broken file
    if out_file.exists():
        out_file.unlink()

    print("Downloading MathVista (testmini) …")
    try:
        ds = load_dataset("AI4Math/MathVista", split="testmini", token=HF_TOKEN)
        items = []
        for row in ds:
            pid = str(row.get("pid", row.get("id", len(items))))
            img = row.get("decoded_image")
            img_path = ""
            if img:
                img_path = f"images/{pid}.jpg"
                img.convert("RGB").save(out / img_path)
            
            items.append({
                "id":           pid,
                "image":        img_path,
                "question":     row.get("question", ""),
                "answer":       str(row.get("answer", "")),
                "question_type": row.get("question_type", ""),
            })
            if len(items) >= max_samples["mathvista"]:
                break
        with open(out_file, "w") as f:
            json.dump(items, f, indent=2)
        print(f"  → MathVista: {len(items)} samples + images → {out_file}")
    except Exception as e:
        print(f"  [WARN] MathVista download failed: {e}")

    # ── VQAv2 ──────────────────────────────────────────────────────────────
    out = BENCH_DIR / "vqav2"
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    out_file = out / "vqav2_val.json"
    
    if out_file.exists():
        out_file.unlink()

    print("Downloading VQAv2 (subset) …")
    try:
        # Using a more robust repo for VQAv2
        ds = load_dataset("lmms-lab/VQAv2", split="validation", 
                          token=HF_TOKEN, streaming=True)
        items = []
        for row in ds:
            qid = str(row.get("question_id", row.get("id", len(items))))
            img = row.get("image")
            img_path = ""
            if img:
                img_path = f"images/{qid}.jpg"
                img.convert("RGB").save(out / img_path)

            items.append({
                "id":       qid,
                "image":    img_path,
                "question": row.get("question", ""),
                "answers":  [row.get("answer", "")] if row.get("answer") else [],
            })
            if len(items) >= max_samples["vqav2"]:
                break
        with open(out_file, "w") as f:
            json.dump(items, f, indent=2)
        print(f"  → VQAv2: {len(items)} samples + images → {out_file}")
    except Exception as e:
        print(f"  [WARN] VQAv2 download failed: {e}")

    # ── POPE ───────────────────────────────────────────────────────────────
    out = BENCH_DIR / "pope"
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    out_file = out / "pope_all.json"
    
    if out_file.exists():
        out_file.unlink()

    print("Downloading POPE …")
    try:
        ds = load_dataset("lmms-lab/POPE", split="test", token=HF_TOKEN)
        items = []
        for row in ds:
            pid = str(row.get("id", len(items)))
            img = row.get("image")
            img_path = ""
            if img:
                img_path = f"images/{pid}.jpg"
                img.convert("RGB").save(out / img_path)

            items.append({
                "id":       pid,
                "image":    img_path,
                "question": row.get("question", ""),
                "label":    str(row.get("answer", "")),
                "category": row.get("category", ""),
            })
            if len(items) >= max_samples["pope"]:
                break
        with open(out_file, "w") as f:
            json.dump(items, f, indent=2)
        print(f"  → POPE: {len(items)} samples + images → {out_file}")
    except Exception as e:
        print(f"  [WARN] POPE download failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Model Downloader
# ─────────────────────────────────────────────────────────────────────────────
def download_model(name: str, cfg: dict, ignore_patterns=None):
    """Download a model via huggingface_hub.snapshot_download."""
    local_dir = cfg["local_dir"]
    repo_id   = cfg["repo_id"]
    local_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded (look for config.json)
    if (local_dir / "config.json").exists():
        print(f"[SKIP] {name} already exists at {local_dir}")
        return True

    print(f"\n{'='*70}")
    print(f"  Downloading: {name}")
    print(f"  Repo:        {repo_id}")
    print(f"  Dest:        {local_dir}")
    print(f"  Est. size:   ~{cfg['est_gb']:.1f} GB")
    print(f"  Notes:       {cfg['notes']}")
    print(f"{'='*70}")
    t0 = time.time()

    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            token=HF_TOKEN,
            ignore_patterns=ignore_patterns or [],
            local_files_only=False,
        )
        elapsed = time.time() - t0
        print(f"  [OK] Done in {elapsed/60:.1f} min — {local_dir}")
        return True
    except Exception as e:
        print(f"  [ERROR] FAILED: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="ACC campaign Model Downloader")
    parser.add_argument("--students",  nargs="*",
                        choices=list(STUDENT_MODELS.keys()) + ["all"],
                        default=["all"],
                        help="Which student models to download")
    parser.add_argument("--teacher",   action="store_true", default=True,
                        help="Download teacher oracle (Llama-3.2-Vision-11B)")
    parser.add_argument("--no-teacher", dest="teacher", action="store_false")
    parser.add_argument("--benchmarks", action="store_true", default=True,
                        help="Download VLM benchmarks (MathVista/VQAv2/POPE)")
    parser.add_argument("--no-benchmarks", dest="benchmarks", action="store_false")
    parser.add_argument("--dry-run",   action="store_true",
                        help="Print plan without downloading")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  ACC campaign — VLM Fleet Download")
    print("=" * 70)

    # Login
    login(token=HF_TOKEN, add_to_git_credential=False)
    print(f"  HF Token:  {HF_TOKEN[:8]}…")

    # Determine students
    students_to_dl = (
        list(STUDENT_MODELS.keys()) if "all" in (args.students or [])
        else (args.students or [])
    )

    if args.dry_run:
        print("\n[DRY RUN] Would download:")
        for k in students_to_dl:
            c = STUDENT_MODELS[k]
            print(f"  Student: {k:25s}  {c['repo_id']:50s}  ~{c['est_gb']:.1f} GB")
        if args.teacher:
            for k, c in TEACHER_MODELS.items():
                print(f"  Teacher: {k:25s}  {c['repo_id']:50s}  ~{c['est_gb']:.1f} GB")
        return

    # Download students
    failed = []
    for name in students_to_dl:
        cfg = STUDENT_MODELS[name]
        # For safety, ignore original float weights to save disk (we quantise at runtime)
        ignore = ["*.safetensors.index.json"]  # keep all — need full weights for BnB
        ok = download_model(name, cfg, ignore_patterns=[])
        if not ok:
            failed.append(name)

    # Download teacher
    if args.teacher:
        for name, cfg in TEACHER_MODELS.items():
            ok = download_model(name, cfg, ignore_patterns=[])
            if not ok:
                failed.append(name)

    # Download benchmarks
    if args.benchmarks:
        print("\n" + "=" * 70)
        print("  Downloading VLM Benchmarks …")
        print("=" * 70)
        download_benchmarks()

    # Summary
    print("\n" + "=" * 70)
    print("  DOWNLOAD COMPLETE")
    if failed:
        print(f"  [ERROR] Failed: {', '.join(failed)}")
    else:
        print("  [OK] All models & benchmarks ready")
    print(f"\n  Student VLMs  →  {STUDENT_DIR}")
    print(f"  Teacher VLM   →  {TEACHER_DIR}")
    print(f"  Benchmarks    →  {BENCH_DIR}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
