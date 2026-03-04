#!/usr/bin/env python3
import os
import json
from pathlib import Path
from datasets import load_dataset
from PIL import Image

HF_TOKEN = "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"
BENCH_DIR = Path("/home/cse-sdpl/research/ACC/01_DATA/Benchmarks")

def download_mathvista(count=1001):
    out = BENCH_DIR / "mathvista"
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    out_file = out / "mathvista_testmini.json"
    
    print(f"Downloading MathVista ({count}) ...")
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
        if len(items) >= count:
            break
    with open(out_file, "w") as f:
        json.dump(items, f, indent=2)
    print(f"  Done: {len(items)} samples.")

def download_pope(count=3001):
    out = BENCH_DIR / "pope"
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    out_file = out / "pope_all.json"
    
    print(f"Downloading POPE ({count}) ...")
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
        if len(items) >= count:
            break
    with open(out_file, "w") as f:
        json.dump(items, f, indent=2)
    print(f"  Done: {len(items)} samples.")

def download_vqav2(count=1001):
    out = BENCH_DIR / "vqav2"
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    out_file = out / "vqav2_val.json"
    
    print(f"Downloading VQAv2 ({count}) ...")
    ds = load_dataset("lmms-lab/VQAv2", split="validation", token=HF_TOKEN, streaming=True)
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
        if len(items) >= count:
            break
    with open(out_file, "w") as f:
        json.dump(items, f, indent=2)
    print(f"  Done: {len(items)} samples.")

if __name__ == "__main__":
    download_mathvista(1001)
    download_pope(3001)
    download_vqav2(1001)
