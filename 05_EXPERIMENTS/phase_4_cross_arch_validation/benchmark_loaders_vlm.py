"""
VLM Benchmark Data Loaders for ACC campaign
===========================================
Loaders for:
  1. MathVista (Mathematical Reasoning)
  2. POPE      (Object Probing / Hallucination)
  3. ALFWorld  (Visual Agentic Tasks)
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from PIL import Image

@dataclass
class VLMSample:
    """Standardized multimodal sample format."""
    task_type: str
    question: str
    image: Optional[Any]  # PIL Image or path
    ground_truth: str
    sample_id: str
    metadata: Dict

def _load_jsonl(path: Path, num_samples: int, seed: int = 42) -> List[dict]:
    with open(path) as f:
        data = [json.loads(line) for line in f if line.strip()]
    if num_samples < len(data):
        random.seed(seed)
        data = random.sample(data, num_samples)
    return data

# 1. MathVista
class MathVistaLoader:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)

    def get_samples(self, num_samples: int = 170, seed: int = 42) -> List[VLMSample]:
        path = self.data_dir / "mathvista_testmini.json"
        with open(path) as f:
            data_list = json.load(f)
        
        if num_samples < len(data_list):
            random.seed(seed)
            data_list = random.sample(data_list, num_samples)
            
        samples = []
        for item in data_list:
            img_path = self.data_dir / item["image"]
            samples.append(VLMSample(
                task_type="mathvista",
                question=item["question"],
                image=img_path,
                ground_truth=str(item["answer"]),
                sample_id=str(item["id"]),
                metadata={"question_type": item.get("question_type", "")}
            ))
        return samples

# 2. POPE
class POPELoader:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)

    def get_samples(self, num_samples: int = 171, seed: int = 42) -> List[VLMSample]:
        path = self.data_dir / "pope_all.json"
        with open(path) as f:
            data = json.load(f)
        
        if num_samples < len(data):
            random.seed(seed)
            data = random.sample(data, num_samples)
            
        samples = []
        for item in data:
            img_path = self.data_dir / item["image"]
            samples.append(VLMSample(
                task_type="pope",
                question=item["question"],
                image=img_path,
                ground_truth=item["label"],
                sample_id=str(item["id"]),
                metadata={"category": item.get("category", "")}
            ))
        return samples

# 3. VQAv2
class VQAv2Loader:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)

    def get_samples(self, num_samples: int = 1001, seed: int = 42) -> List[VLMSample]:
        path = self.data_dir / "vqav2_val.json"
        with open(path) as f:
            data = json.load(f)
        
        if num_samples < len(data):
            random.seed(seed)
            data = random.sample(data, num_samples)
            
        samples = []
        for item in data:
            img_path = self.data_dir / item["image"]
            # VQAv2 val split does not ship answers in the public JSON.
            # The ground_truth is marked as OPEN so scorers can use
            # LLM-as-judge or soft-match against VQAv2 annotation API.
            # Collect all available answers (may be empty for val set).
            answers = item.get("answers", []) or []
            if isinstance(answers, str):
                answers = [answers]
            gt = "; ".join(str(a) for a in answers) if answers else "OPEN"
            samples.append(VLMSample(
                task_type="vqav2",
                question=item["question"],
                image=img_path,
                ground_truth=gt,
                sample_id=str(item["id"]),
                metadata={"open_answer": (gt == "OPEN")}
            ))
        return samples

# 4. ALFWorld Visual
class ALFWorldVisualLoader:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.task_templates = [
            {'task': 'pick_and_place',
             'template': "You are in a kitchen. {goal}. What is your first action?",
             'goals': [
                 ('find the apple and put it in the fridge', 'apple'),
                 ('find the mug and place it on the coffee maker', 'mug'),
             ]},
            {'task': 'examine',
             'template': "You are in a bedroom. {goal}. What is your first action?",
             'goals': [
                 ('examine the painting with the lamp', 'painting'),
                 ('examine the book with the desk lamp', 'book'),
             ]},
        ]

    def get_samples(self, num_samples: int = 135, seed: int = 42) -> List[VLMSample]:
        random.seed(seed)
        # Expanded ALFWorld task templates for broader coverage
        templates = [
            ("You are in a kitchen. find the apple and put it in the fridge. State your first action in 5 words or less.",
             "go to apple"),
            ("You are in a kitchen. find the mug and place it on the coffee maker. State your first action in 5 words or less.",
             "go to mug"),
            ("You are in a bedroom. examine the painting using the desk lamp. State your first action in 5 words or less.",
             "go to painting"),
            ("You are in a bathroom. find the soap and place it in the bathtub. State your first action in 5 words or less.",
             "go to soap"),
            ("You are in a living room. find the remote and put it on the sofa. State your first action in 5 words or less.",
             "go to remote"),
            ("You are in a kitchen. heat the egg and place it on the counter. State your first action in 5 words or less.",
             "go to egg"),
        ]
        samples = []
        for idx in range(num_samples):
            q, gt = templates[idx % len(templates)]
            samples.append(VLMSample(
                task_type="alfworld",
                question=q,
                image=None,
                ground_truth=gt,
                sample_id=f"alf_v_{idx}",
                metadata={"requires_vision": False,
                           "repetition_penalty": 1.2,
                           "no_repeat_ngram_size": 3,
                           "max_new_tokens": 20}  # short answer expected
            ))
        return samples

# Unified Loader
LOADER_REGISTRY = {
    "mathvista": MathVistaLoader,
    "pope": POPELoader,
    "vqav2": VQAv2Loader,
    "alfworld": ALFWorldVisualLoader,
}

def load_vlm_benchmark(name: str, data_dir: Path, num_samples: int = 150) -> List[VLMSample]:
    loader = LOADER_REGISTRY[name.lower()](data_dir)
    return loader.get_samples(num_samples=num_samples)
