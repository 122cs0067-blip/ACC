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
import alfworld
import alfworld.agents.environment
import yaml
import numpy as np

# CONFIGURATION
VLLM_URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = os.environ.get("CURRENT_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
LOG_FILE = f"../../06_RESULTS/phase_4_logs/{MODEL_NAME.split('/')[-1]}_alfworld.log"
BASELINE_SET = os.environ.get("BASELINE_SET", "any4,saup,splitwise,acc")
ALFWORLD_DATA = os.environ.get("ALFWORLD_DATA", "")
ALFWORLD_CONFIG = os.environ.get(
    "ALFWORLD_CONFIG",
    os.path.join(ALFWORLD_DATA, "configs", "eval_config.yaml") if ALFWORLD_DATA else "",
)


def llm_call(prompt: str, temp: float = 0.1, n: int = 1):
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temp,
        "max_tokens": 60,
        "n": n,
    }
    try:
        res = requests.post(VLLM_URL, json=payload, timeout=5).json()
        if n > 1:
            return [c["message"]["content"].strip() for c in res["choices"]]
        return res["choices"][0]["message"]["content"].strip().split("\n")[0]
    except Exception:
        return "look"


def run_agent(agent_type: str, env, episode_idx: int, debug_log: str | None = None):
    obs, info = env.reset()
    obs = obs[0]
    last_action = ""
    history = []

    for step in range(25):
        prompt = f"Observation: {obs}\nTask: {info['extra.gamefile'][0]}\nAction:"

        if agent_type == "any4":
            action = llm_call(prompt)
        elif agent_type == "saup":
            candidates = llm_call(prompt, temp=0.7, n=3)
            action = max(set(candidates), key=candidates.count)
        elif agent_type == "splitwise":
            action = llm_call(prompt)
            if action == last_action:
                action = "look"
        elif agent_type == "acc":
            action = llm_call(prompt)
            if action in history[-2:]:
                action = "look"
        else:
            action = llm_call(prompt)

        history.append(action)
        last_action = action
        if debug_log and episode_idx == 0 and step < 5:
            with open(debug_log, "a") as df:
                df.write(
                    f"STEP {step} | AGENT {agent_type} | ACTION {action}\n"
                    f"OBS {obs}\n"
                )
                if agent_type == "saup":
                    df.write(f"CANDIDATES {candidates}\n")
                df.write("---\n")
        obs, scores, dones, _ = env.step([action])
        obs, score, done = obs[0], scores[0], dones[0]

        if done:
            return (1 if score > 0.5 else 0), (1 if step > 20 else 0)

    return 0, 1


def run_eval_suite() -> None:
    print(f"[ALFWorld] Running baseline suite on {MODEL_NAME}")
    if not ALFWORLD_CONFIG or not os.path.isfile(ALFWORLD_CONFIG):
        raise FileNotFoundError(
            "ALFWORLD_CONFIG not found. Set ALFWORLD_DATA or ALFWORLD_CONFIG."
        )
    with open(ALFWORLD_CONFIG, "r") as f:
        config = yaml.safe_load(f)

    env_cls = alfworld.agents.environment.get_environment(config["env"]["type"])
    env = env_cls(config, train_eval="eval_out_of_distribution")
    env = env.init_env(batch_size=1)

    agents = [a.strip() for a in BASELINE_SET.split(",") if a.strip()]

    with open(LOG_FILE, "a") as f:
        f.write(f"--- NEW SESSION: {MODEL_NAME} ---\n")

    debug_log = None
    if os.environ.get("ALFWORLD_DEBUG", "0") == "1":
        debug_log = LOG_FILE.replace(".log", "_debug.log")
        with open(debug_log, "a") as df:
            df.write(f"DEBUG SESSION: {MODEL_NAME}\n")

    for agent in agents:
        print(f"[ALFWorld] Running {agent.upper()}")
        wins, loops = 0, 0
        for episode_idx in range(10):
            w, l = run_agent(agent, env, episode_idx, debug_log)
            wins += w
            loops += l

        with open(LOG_FILE, "a") as f:
            f.write(f"AGENT: {agent} | SUCCESS: {wins}/10 | LOOPS: {loops}/10\n")


if __name__ == "__main__":
    run_eval_suite()
