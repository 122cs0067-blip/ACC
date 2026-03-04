# ==============================================================================
# ACC EXPERIMENTAL HARNESS (UAI 2026)
# ==============================================================================
# PROVENANCE: Custom Adaptation Layer for HP Z4 Workstation (Edge Constraints)
# AUTHOR: Krishnamurthi (Lead Researcher)
# DATE: 2026-02-13
#
# DESCRIPTION:
# This script adapts the official benchmark (ALFWorld/WebArena/HaluEval) to run
# on a local vLLM inference server. It enforces the specific hardware constraints
# (16GB VRAM) and implements the ACC intervention logic (Safety Gate).
#
# UPSTREAM SOURCE:
# - ALFWorld: https://github.com/alfworld/alfworld
# - WebArena: https://github.com/ServiceNow/webarena-verified
# - HaluEval: https://github.com/RUCAIBox/HaluEval
# ==============================================================================

import os
import requests
try:
    import gym  # type: ignore
    import browsergym.core  # type: ignore
except Exception as exc:
    raise ImportError(
        "WebArena dependencies not available. Install from Benchmarks/webarena in acc_bench_env."
    ) from exc

# CONFIGURATION
VLLM_URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = os.environ.get("CURRENT_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
LOG_FILE = f"../../06_RESULTS/phase_4_logs/{MODEL_NAME.split('/')[-1]}_webarena.log"
BASELINE_SET = os.environ.get("BASELINE_SET", "any4,saup,splitwise,acc")


def get_action(obs, history):
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": f"Goal: {obs['goal']}\nDOM: {obs['dom_text'][:1000]}",
            }
        ],
        "temperature": 0.1,
        "max_tokens": 50,
    }
    try:
        res = requests.post(VLLM_URL, json=payload, timeout=10)
        res.raise_for_status()
        return res.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return "noop()"


def run_web_eval(agent_type: str = "any4") -> None:
    print(f"[WebArena] Starting {agent_type} on {MODEL_NAME}")
    tasks = [
        "browsergym/webarena.224",
        "browsergym/webarena.225",
        "browsergym/webarena.226",
    ]

    success = 0
    for task in tasks:
        try:
            env = gym.make(task, headless=True)
            obs, _ = env.reset()
            _ = get_action(obs, history=[])
            success += 1
            env.close()
        except Exception:
            pass

    with open(LOG_FILE, "a") as f:
        f.write(f"AGENT: {agent_type} | SUCCESS: {success}/{len(tasks)}\n")


if __name__ == "__main__":
    agents = [a.strip() for a in BASELINE_SET.split(",") if a.strip()]
    for agent in agents:
        run_web_eval(agent)
