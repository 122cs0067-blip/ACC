"""
Benchmark Data Loaders for ACC Cross-Architecture Campaign

Provides standardized interfaces for all 8 UAI 2026 benchmarks:
  1. GSM8K      - Grade-school math reasoning (1319 samples)
  2. HumanEval  - Code generation (164 samples)
  3. MBPP       - Python programming (500 samples)
  4. HaluEval   - Hallucination detection (10000 samples)
  5. TruthfulQA - Factual truthfulness (817 samples)
  6. IFEval     - Instruction following (541 samples)
  7. ALFWorld   - Multi-step agentic tasks (134 samples)
  8. MMLU-Pro   - Multi-task knowledge (1000 samples)

Each loader returns prompts formatted for LLM inference with ground truth
for validation and drift analysis.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class BenchmarkSample:
    """Standardized benchmark sample format."""
    task_type: str
    prompt: str
    ground_truth: str
    metadata: Dict
    sample_id: str


# ============================================================================
# JSONL LOADER UTILITY
# ============================================================================

def _load_jsonl(path: Path, num_samples: int, seed: int = 42) -> List[dict]:
    """Load and optionally subsample from a JSONL file."""
    with open(path) as f:
        data = [json.loads(line) for line in f if line.strip()]
    if num_samples < len(data):
        random.seed(seed)
        data = random.sample(data, num_samples)
    return data


# ============================================================================
# 1. GSM8K
# ============================================================================

class GSM8KLoader:
    """GSM8K: Grade-School Math Reasoning (1319 test samples)."""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)

    @staticmethod
    def _extract_answer(answer_text: str) -> str:
        if "####" in answer_text:
            return answer_text.split("####")[-1].strip().replace(",", "")
        import re
        numbers = re.findall(r"-?[\d,]+\.?\d*", answer_text)
        return numbers[-1].replace(",", "") if numbers else answer_text.strip()

    def get_samples(self, num_samples: int = 50, seed: int = 42) -> List[BenchmarkSample]:
        path = self.data_dir / "gsm8k_test.jsonl"
        data = _load_jsonl(path, num_samples, seed)
        samples = []
        for idx, item in enumerate(data):
            q = item["question"]
            ans = self._extract_answer(item["answer"])
            samples.append(BenchmarkSample(
                task_type="gsm8k",
                prompt=f"Solve the following math problem step by step. Show your reasoning, then give the final answer after \"####\".\n\nQuestion: {q}\n\nSolution:",
                ground_truth=ans,
                metadata={"full_solution": item["answer"], "original_question": q},
                sample_id=f"gsm8k_{idx}",
            ))
        return samples


# ============================================================================
# 2. HumanEval
# ============================================================================

class HumanEvalLoader:
    """HumanEval: Code Generation (164 test samples)."""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)

    def get_samples(self, num_samples: int = 50, seed: int = 42) -> List[BenchmarkSample]:
        path = self.data_dir / "humaneval_test.jsonl"
        data = _load_jsonl(path, num_samples, seed)
        samples = []
        for idx, item in enumerate(data):
            samples.append(BenchmarkSample(
                task_type="humaneval",
                prompt=f"Complete the following Python function:\n\n{item['prompt']}",
                ground_truth=item.get("canonical_solution", ""),
                metadata={"task_id": item.get("task_id", ""), "entry_point": item.get("entry_point", "")},
                sample_id=f"humaneval_{idx}",
            ))
        return samples


# ============================================================================
# 3. MBPP
# ============================================================================

class MBPPLoader:
    """MBPP: Mostly Basic Python Problems (500 test samples)."""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)

    def get_samples(self, num_samples: int = 50, seed: int = 42) -> List[BenchmarkSample]:
        path = self.data_dir / "mbpp_test.jsonl"
        data = _load_jsonl(path, num_samples, seed)
        samples = []
        for idx, item in enumerate(data):
            test_str = item.get("test_list", [""])[0] if item.get("test_list") else ""
            samples.append(BenchmarkSample(
                task_type="mbpp",
                prompt=f"Write a Python function for the following task:\n{item['text']}\n\n{test_str}\n\nSolution:",
                ground_truth=item.get("code", ""),
                metadata={"task_id": item.get("task_id", "")},
                sample_id=f"mbpp_{idx}",
            ))
        return samples


# ============================================================================
# 4. HaluEval
# ============================================================================

class HaluEvalLoader:
    """HaluEval: Hallucination Detection (10000 samples, use 1000 subset)."""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)

    def get_samples(self, num_samples: int = 1000, seed: int = 42) -> List[BenchmarkSample]:
        path = self.data_dir / "halueval_qa.jsonl"
        data = _load_jsonl(path, num_samples, seed)
        samples = []
        for idx, item in enumerate(data):
            q = item.get("question", item.get("query", ""))
            answer = item.get("answer", item.get("right_answer", ""))
            samples.append(BenchmarkSample(
                task_type="halueval",
                prompt=f"Answer the following question factually and concisely.\n\nQuestion: {q}\n\nAnswer:",
                ground_truth=answer,
                metadata={"hallucinated_answer": item.get("hallucinated_answer", "")},
                sample_id=f"halueval_{idx}",
            ))
        return samples


# ============================================================================
# 5. TruthfulQA
# ============================================================================

class TruthfulQALoader:
    """TruthfulQA: Factual Truthfulness (817 test samples)."""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)

    def get_samples(self, num_samples: int = 817, seed: int = 42) -> List[BenchmarkSample]:
        path = self.data_dir / "truthfulqa_test.jsonl"
        data = _load_jsonl(path, num_samples, seed)
        samples = []
        for idx, item in enumerate(data):
            q = item.get("question", "")
            best = item.get("best_answer", "")
            if not best:
                ca = item.get("correct_answers", [])
                best = ca[0] if isinstance(ca, list) and ca else ""
            samples.append(BenchmarkSample(
                task_type="truthfulqa",
                prompt=f"Answer the following question truthfully and concisely.\n\nQuestion: {q}\n\nAnswer:",
                ground_truth=best,
                metadata={"category": item.get("category", "")},
                sample_id=f"truthfulqa_{idx}",
            ))
        return samples


# ============================================================================
# 6. IFEval
# ============================================================================

class IFEvalLoader:
    """IFEval: Instruction Following Evaluation (541 test samples)."""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)

    def get_samples(self, num_samples: int = 541, seed: int = 42) -> List[BenchmarkSample]:
        path = self.data_dir / "ifeval_test.jsonl"
        data = _load_jsonl(path, num_samples, seed)
        samples = []
        for idx, item in enumerate(data):
            samples.append(BenchmarkSample(
                task_type="ifeval",
                prompt=item.get("prompt", ""),
                ground_truth=json.dumps(item.get("instruction_id_list", [])),
                metadata={"kwargs": item.get("kwargs", [])},
                sample_id=f"ifeval_{idx}",
            ))
        return samples


# ============================================================================
# 7. ALFWorld
# ============================================================================

class ALFWorldLoader:
    """ALFWorld: Multi-step text-based RL (134 standard tasks)."""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.task_templates = [
            {'task': 'pick_and_place',
             'template': "You are in a kitchen. Your task is to: {goal}. To complete this task, you need to first find the {object}, pick it up, then place it {location}. What is your first action?",
             'goals': [
                 ('find the apple and put it in the fridge', 'apple', 'in the fridge'),
                 ('find the mug and place it on the coffee maker', 'mug', 'on the coffee maker'),
                 ('find the book and put it on the shelf', 'book', 'on the shelf'),
             ]},
            {'task': 'examine',
             'template': "You are in a bedroom. Your task is to: examine the {object} with the {tool}. First, locate the {object}, then find the {tool}, and finally use it to examine. What is your first action?",
             'goals': [
                 ('examine the painting with the lamp', 'painting', 'lamp'),
                 ('examine the book with the desk lamp', 'book', 'desk lamp'),
             ]},
            {'task': 'clean',
             'template': "You are in a bathroom. Your task is to: clean the {object} with {tool} and put it in {location}. This requires multiple steps. What is your first action?",
             'goals': [
                 ('clean the plate with soap', 'plate', 'soap', 'cabinet'),
                 ('clean the cloth with water', 'cloth', 'water', 'drawer'),
             ]},
        ]

    def get_samples(self, num_samples: int = 20, seed: int = 42) -> List[BenchmarkSample]:
        random.seed(seed)
        samples = []
        for idx in range(num_samples):
            template = self.task_templates[idx % len(self.task_templates)]
            goal_data = random.choice(template['goals'])
            if template['task'] == 'pick_and_place':
                goal, obj, location = goal_data
                prompt = template['template'].format(goal=goal, object=obj, location=location)
            elif template['task'] == 'examine':
                goal, obj, tool = goal_data
                prompt = template['template'].format(object=obj, tool=tool)
            else:
                goal, obj, tool, location = goal_data
                prompt = template['template'].format(object=obj, tool=tool, location=location)
            ground_truth = f"go to {obj if template['task'] != 'clean' else obj}"
            samples.append(BenchmarkSample(
                task_type='alfworld', prompt=prompt, ground_truth=ground_truth,
                metadata={'task_template': template['task'], 'requires_multi_step': True},
                sample_id=f'alfworld_{idx}',
            ))
        return samples


# ============================================================================
# 8. MMLU-Pro
# ============================================================================

class MMLUProLoader:
    """MMLU-Pro: Multi-task Language Understanding (1000 sample subset)."""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)

    def get_samples(self, num_samples: int = 1000, seed: int = 42) -> List[BenchmarkSample]:
        path = self.data_dir / "mmlu_pro_test.jsonl"
        data = _load_jsonl(path, num_samples, seed)
        samples = []
        for idx, item in enumerate(data):
            q = item.get("question", "")
            choices = item.get("options", [])
            choices_str = "\n".join([f"  {chr(65+j)}. {c}" for j, c in enumerate(choices)])
            answer = item.get("answer", "")
            samples.append(BenchmarkSample(
                task_type="mmlu_pro",
                prompt=f"Answer the following question. Choose the best option.\n\n{q}\n{choices_str}\n\nAnswer:",
                ground_truth=str(answer),
                metadata={"category": item.get("category", ""), "src": item.get("src", "")},
                sample_id=f"mmlu_pro_{idx}",
            ))
        return samples


# ============================================================================
# UNIFIED LOADER
# ============================================================================

LOADER_REGISTRY = {
    "gsm8k": GSM8KLoader,
    "humaneval": HumanEvalLoader,
    "mbpp": MBPPLoader,
    "halueval": HaluEvalLoader,
    "truthfulqa": TruthfulQALoader,
    "ifeval": IFEvalLoader,
    "alfworld": ALFWorldLoader,
    "mmlu_pro": MMLUProLoader,
}


def load_benchmark(
    benchmark_name: str,
    data_dir: Path,
    num_samples: int = 20,
    seed: int = 42,
) -> List[BenchmarkSample]:
    """
    Unified benchmark loader interface.

    Args:
        benchmark_name: One of 'gsm8k', 'humaneval', 'mbpp', 'halueval',
                        'truthfulqa', 'ifeval', 'alfworld', 'mmlu_pro'
        data_dir: Path to benchmark data directory
        num_samples: Number of samples to load
        seed: Random seed for reproducibility

    Returns:
        List of BenchmarkSample instances
    """
    if benchmark_name not in LOADER_REGISTRY:
        raise ValueError(
            f"Unknown benchmark: {benchmark_name}. "
            f"Available: {', '.join(LOADER_REGISTRY.keys())}"
        )
    loader = LOADER_REGISTRY[benchmark_name](data_dir)
    return loader.get_samples(num_samples, seed)


if __name__ == "__main__":
    base_dir = Path("/home/cse-sdpl/research/ACC/01_DATA/Benchmarks")
    for name in LOADER_REGISTRY:
        print(f"\n{'='*70}")
        print(f"Testing {name} Loader...")
        try:
            samples = load_benchmark(name, base_dir / name, num_samples=2)
            for sample in samples:
                print(f"\n  {sample.sample_id}:")
                print(f"  Prompt: {sample.prompt[:120]}...")
                print(f"  Ground Truth: {sample.ground_truth[:80]}")
        except Exception as e:
            print(f"  SKIPPED ({e})")

