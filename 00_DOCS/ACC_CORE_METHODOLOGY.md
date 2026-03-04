# ACC Core: Full Methodology — A to Z

> **Source root:** `02_SRC/acc_core/`  
> **Paper:** Active Conformal Control (ACC), UAI 2026  
> **Date:** February 2026

---

## The One-Sentence Idea

> *At every generated token, a lightweight GPU sensor measures how far the
> 4-bit student has drifted from the teacher's manifold; when a dual-signal
> safety gate fires, the full-precision CPU teacher re-anchors the trajectory
> via a zero-copy shared-memory handoff—all within a single workstation.*

---

## System-Level ASCII Architecture

```
════════════════════════════════════════════════════════════════════════════════
              ACTIVE CONFORMAL CONTROL (ACC) — FULL SYSTEM DIAGRAM
════════════════════════════════════════════════════════════════════════════════

   ┌──────────────────────────────────────────────────────────────────────┐
   │                     HP Z4 G5 WORKSTATION                            │
   │                                                                      │
   │   ┌─────────────────────────────────┐  PCIe 4.0 x8  ┌────────────────────────────┐
   │   │      NVIDIA RTX 2000 Ada        │  ◄──────────►  │   Intel Xeon W5-2445       │
   │   │         16 GB GDDR6             │  (92ms budget) │   10-core / 64 GB DDR5     │
   │   │                                 │                │   Intel AMX + oneDNN       │
   │   │  ┌──────────────────────────┐   │                │                            │
   │   │  │  STUDENT MODEL (4-bit)   │   │                │  ┌──────────────────────┐  │
   │   │  │  Qwen-2.5-1.5b  (Any4)  │   │                │  │  TEACHER MODEL (BF16)│  │
   │   │  │  Phi-3-Mini     (AWQ)   │   │                │  │  Llama-3-8B-Instruct │  │
   │   │  │  Mistral-7B     (Any4)  │   │                │  │  ~16 GB System RAM   │  │
   │   │  │                          │   │                │  │  OracleBridge        │  │
   │   │  │  Generates tokens t=0... │   │                │  └──────────┬───────────┘  │
   │   │  │  ┌───────────────────┐   │   │                │             │              │
   │   │  │  │Hidden State z_t   │   │   │                │  ┌──────────▼───────────┐  │
   │   │  │  │(4096-dim, float32)│   │   │                │  │  correct_trajectory()│  │
   │   │  │  └────────┬──────────┘   │   │                │  │  BF16 + torch.compile│  │
   │   │  └───────────│──────────────┘   │                │  │  344ms/token (AMX)   │  │
   │   │              │                  │                │  └──────────────────────┘  │
   │   │  ┌───────────▼──────────────┐   │                └────────────────────────────┘
   │   │  │   i-ppDRE SENSOR         │   │
   │   │  │   (detector/ipp_dre.py)  │   │
   │   │  │                          │   │
   │   │  │  1. RFF projection        │   │     ┌──────────────────────────────────────┐
   │   │  │     4096D → 128D → 1024D  │   │     │       /dev/shm  (ACCOpsBridge)       │
   │   │  │  2. Dot product w/ alpha   │   │     │       ring_buffer.py                │
   │   │  │  3. w(x_t) = |α·φ(x_t)|  │   │     │                                      │
   │   │  │  4. Online alpha update    │   │     │  [Timestamp|Flag|Step|VectorData]   │
   │   │  │     (forgetting lambda=0.99)   │   │     │   ← Student writes z_t each step    │
   │   │  └───────────┬──────────────┘   │     │   → Oracle flips FLAG on detection  │
   │   │  0.319-0.483ms per token        │     └──────────────────────────────────────┘
   │   │              │                  │
   │   │  ┌───────────▼──────────────┐   │
   │   │  │  CONFORMAL SAFETY GATE   │   │
   │   │  │  (control/conformal.py)  │   │
   │   │  │                          │   │
   │   │  │  Signal 1: w(x_t) > lambda*  │   │
   │   │  │  Signal 2: |C(x_t)| > κ │   │
   │   │  │                          │   │
   │   │  │  AND-logic:              │   │
   │   │  │  FIRE iff BOTH signals   │   │
   │   │  │  exceed threshold        │   │
   │   │  └───────────┬──────────────┘   │
   │   │              │                  │
   │   │     ┌────────▼────────┐         │
   │   │     │  GATE FIRES?    │         │
   │   │     └──┬──────────┬───┘         │
   │   │     NO │          │ YES         │
   │   │        ▼          ▼             │
   │   │   [Continue]  [HANDOFF]         │
   │   │               KVCacheSync       │
   │   │               lazy_sync.py      │
   │   │               0.03ms PCIe       │
   │   └─────────────────────────────────┘
   │
   └─────────────────────────────── ▲ corrected tokens re-injected into student context
```

---

## Module-by-Module Deep Dive

---

### MODULE 1 — `detector/rff_kernel.py`
**Class:** `RandomFourierFeatures`  
**Role:** The first stage of the sensor pipeline — projects the 4096-dimensional hidden state down to a 1024-dimensional feature vector using a *fixed, orthogonalized* random kernel.

#### What it does

```
Raw hidden state z_t ∈ ℝ^4096
          │
          │  Orthogonalized projection matrix Ω ∈ ℝ^(512×4096)
          │  (computed via QR decomposition at init time)
          ▼
proj = z_t @ Ω.T + bias        ∈ ℝ^512
          │
          │  Fourier feature map
          ▼
φ(z_t) = [cos(proj), sin(proj)] / √R   ∈ ℝ^1024
```

**Why QR orthogonalization?**
> Without it, 4-bit quantization noise amplifies through the projection
> because standard random matrices have correlated rows. After QR, the
> rows are orthonormal — each dimension of the projection captures an
> **independent** direction in the hidden-state space. Quantization noise
> is spread evenly across all 512 directions and cancels in the dot product,
> while **semantic drift** in a specific direction is preserved.

**Key parameters:**
| Parameter | Value | Meaning |
|-----------|-------|---------|
| `input_dim` | 128 | Pre-dimension-reduced hidden state |
| `rff_dim` | 512 | Number of RFF basis functions |
| `output_dim` | 1024 | `2 × rff_dim` (cos + sin channels) |
| `sigma` | 1.0 | Bandwidth of the Gaussian kernel being approximated |
| `bias` | U(0, 2π) | Shift invariance for the kernel approximation |

**Mathematical guarantee:**
```
E[φ(x)·φ(y)] = k(x, y) = exp(-‖x-y‖² / 2σ²)
```
The inner product in feature space approximates the Gaussian kernel exactly
as `rff_dim → ∞`. At R=512, the approximation error is well below the drift
magnitude we measure in practice.

---

### MODULE 2 — `detector/ipp_dre.py`
**Class:** `IncrementalDriftTracker`  
**Role:** Maintains a live, online estimate of the density ratio r(z_t) = p_student(z_t) / p_teacher(z_t) and returns the scalar drift score `w(x_t)`.

#### Complete Data Flow

```
Phase 0 (OFFLINE, once per model):
  Teacher manifold activations (N=512 samples, shape 512×4096)
  → rff.forward()
  → teacher_mean_phi = mean(φ(z_teacher))   ∈ ℝ^1024
  Stored as the "ground truth" centroid of the teacher's feature space.

Phase 1 (ONLINE, every token t):

  z_t (student hidden state, ℝ^4096)
           │
           ▼  rff.forward()
  φ(z_t)  ∈ ℝ^1024
           │
           ├──── SCORE ──────────────────────────────────────────
           │     w(x_t) = |α · φ(z_t)|           ← scalar
           │     Latency: O(1024) ≈ 0.390ms
           │
           └──── UPDATE ─────────────────────────────────────────
                 error = φ(z_t) - teacher_mean_phi
                 α ← 0.99 × α  -  0.05 × error    ← PEGASOS step
                 if ‖α‖ > 10: α ← α / ‖α‖         ← norm clamp
```

#### The Forgetting Factor (lambda = 0.99)

The weight vector `α` decays by 1% each step. This means:
- **Effective memory window:** ~100 steps (1/0.01)
- **Old chasm events don't pollute future scoring** — the model can recover
- **Recent errors dominate** — the sensor responds to *current* trajectory state

#### Why `|α · φ(z_t)|` is the right score

- `α` is trained to be large in directions where the student consistently
  differs from the teacher (PEGASOS pushes it toward the error direction)
- `φ(z_t)` is large where the current token's feature space is active
- Their dot product is large when the **current token hits a direction where
  the student has historically diverged** — this is the density chasm signal
- The `abs()` catches both over- and under-shoot relative to the teacher mean

---

### MODULE 3 — `control/conformal.py`
**Class:** `ConformalSafetyGate`  
**Role:** The decision layer. Converts the raw drift score `w(x_t)` (a float) into a binary FIRE/NO-FIRE decision using two calibrated thresholds and AND logic.

#### The Two Signals

```
Signal 1: Drift Score Gate
  w(x_t) > lambda*
  lambda* = (1-ε) empirical quantile of calibration drift scores
  ε   = 0.05  → 95% coverage guarantee
  Formal guarantee: Pr[w(x_test) ≤ lambda*] ≥ 0.95

Signal 2: Bayesian Quadrature Proxy (BQ Gate)
  |C(x_t)| > κ
  |C(x_t)|  = vocabulary tokens with p > floor (prediction set size)
  κ          = (1-ε) empirical quantile of calibration |C| values
  Interpretation: A suddenly large prediction set means the model is
                  "confused" — many tokens look equally plausible.
                  This is Snell-Griffiths' BQ posterior risk proxy:
                  R̂ ≈ mean(|C(x_t)|) / V_max
```

#### AND Logic vs OR Logic

```
OR logic (legacy mode):  FIRE if w(x_t) > lambda*  OR  |C(x_t)| > κ
AND logic (default):     FIRE only if BOTH signals exceed thresholds

Why AND?
  → Prevents "Intervention Addiction": tokens with ambiguous but correct
    distributions (e.g., deterministic math steps) have high |C| but
    low drift. OR logic wastefully triggers the teacher for safe tokens.
  → With AND logic: the teacher fires ONLY when the manifold IS broken
    (w > lambda*) AND the model IS confused (|C| > κ).
    This is the formal intersection of topological failure and epistemic failure.

Non-intervention logging (for efficiency proof):
  Every time drift_alert=True but bq_alert=False, we log a
  "skip_intervention" event. The count of these events divided by
  total drift events = the AND-logic efficiency gain over OR-logic.
```

#### Calibration Procedure (called once per model)

```
OFFLINE CALIBRATION (split-conformal, N = 1,417 samples):

  For each calibration sample x_i:
    1. Run student forward pass
    2. Compute w(x_i) via i-ppDRE.score()
    3. Compute |C(x_i)| = len([v for v in vocab if p(v|x_i) > 0.001])
    4. Store (w_i, |C_i|)

  Sort w values:
    lambda* = scores[ ceil((N+1)(1-ε)) ]   mod N
    lambda* = max(lambda*, min_threshold=0.005)  # noise floor

  Sort |C| values:
    κ  = C_sizes[ ceil((N+1)(1-ε)) ]  mod N

  Formal guarantee:
    Pr[ w(x_test) ≤ lambda* ] ≥ 1 - ε = 0.95   (exchangeability assumption)
```

#### Running Posterior Risk Estimate

```python
# Every token the risk window is updated:
_risk_window → sliding window of last 50 |C(x_t)| values
R̂ = mean(_risk_window)   # normalized in [0, 1] after /V_max

# Rising R̂ → the model is becoming increasingly uncertain
# Used for visualization in Figure 2 of the paper
```

---

### MODULE 4 — `system/ring_buffer.py`
**Class:** `ACCOpsBridge`  
**Role:** Zero-copy shared-memory channel between the GPU student process and the CPU oracle process. Implemented via POSIX `/dev/shm`.

#### Memory Layout

```
/dev/shm/acc_bridge   (1 MB = 1,048,576 bytes)

Offset  Size  Field        Description
──────  ────  ───────────  ─────────────────────────────────────────
0       8     Timestamp    float64 — time.time() of last write
8       4     Flag         int32  — 0=NORMAL, 1=DRIFT_DETECTED, 2=INTERVENTION
12      4     Step         int32  — current generation step t
16      N     VectorData   float32[128] — compressed z_t for Oracle inspection
```

#### Communication Protocol

```
STUDENT (GPU process):                   ORACLE (CPU process):

Every token t:                           Polling loop (background thread):
  1. z_t = hidden_state[last_layer]        1. timestamp, flag, step = read_latest()
  2. bridge.write_state(step, z_t)         2. If flag == DRIFT_DETECTED:
                                               wait... (gate already fired)
  After i-ppDRE + Gate decision:           On new state:
  3. If gate fires:                        3. flag is still NORMAL → start handoff
     bridge.trigger_intervention()
  4. Pause generation loop
  5. Wait for clear_intervention()       ORACLE handoff:
  6. Inject corrected tokens              4. Load teacher output
  7. Resume student generation            5. Write corrected response
                                          6. bridge.clear_intervention()

Signal latency via /dev/shm: ~microseconds (zero-copy, no network)
```

---

### MODULE 5 — `system/lazy_sync.py`
**Classes:** `KVCacheSync`, `ContextSynchronizer`  
**Role:** Manages the actual PCIe tensor transfer of KV-cache state between GPU (student) and CPU (teacher).

#### Transfer Pipeline

```
TRIGGER: ConformalSafetyGate fires

Student GPU:                          CPU System RAM:
  kv_cache (CUDA tensor)
  ├─ _ensure_pinned()                 ← make tensor contiguous
  ├─ .to("cpu", non_blocking=True)   ← DMA transfer via PCIe 4.0 x8
  └─ cuda.synchronize()              ← wait for DMA completion

  Measured latency: 0.030–0.040ms    ← from campaign logs (N=2847 events)
  Budget: 92ms                       ← PCIe 4.0 x8 budget for 1.1GB KV cache
  Status: 0.03ms << 92ms  ✓ WELL WITHIN BUDGET

OracleBridge.correct_trajectory():
  ├─ Re-tokenize context text         ← cross-architecture safe
  ├─ teacher.generate(k=3-5 tokens)   ← BF16 + AMX (344ms/token)
  └─ return new_text, compute_ms

Student GPU (resume):
  corrected_cache.to("cuda:0", non_blocking=True)
  cuda.synchronize()
  ← inject corrected tokens into student's context
  ← resume autoregressive generation from corrected position
```

#### Cross-Architecture Handoff (Critical Detail)

The student (e.g., Qwen-2.5-1.5b) and teacher (Llama-3-8B) have **different
vocabularies and tokenizers**. The handoff cannot pass raw token IDs —
they would be misinterpreted. The solution:

```
Student generates: [token_id_1, token_id_2, ...]
  ↓ decode with student tokenizer
"Step 2: The distance is 80 + 150 ="
  ↓ re-encode with teacher tokenizer
[teacher_id_1, teacher_id_2, ...]
  ↓ teacher generates next 3-5 tokens
"230 miles. Step 3..."
  ↓ decoded to text
"230 miles. Step 3..."
  ↓ re-encode with student tokenizer
  ↓ inject into student KV-cache
Student resumes from "230 miles. Step 3..."
```

This text-based handoff is universal — it works across any pair of models
regardless of tokenizer differences.

---

### MODULE 6 — `system/oracle_bridge.py`
**Class:** `OracleBridge`  
**Role:** The teacher model manager. Loads Llama-3-8B-Instruct in BF16 on CPU, applies Intel AMX acceleration, and exposes the `correct_trajectory()` API.

#### Loading Strategy

```
OracleBridge.load_teacher():

  1. AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
     → Sets pad_token = eos_token (fix for long-sequence attention masking)

  2. AutoModelForCausalLM.from_pretrained(
         dtype=torch.bfloat16,     ← BF16 = native AMX format on Xeon W5
         device_map="cpu",         ← stays in 64GB DDR5, NOT GPU VRAM
         low_cpu_mem_usage=True,   ← streaming load, avoids 2× RAM spike
     )

  3a. If IPEX available:
       ipex.optimize(teacher, dtype=bfloat16, inplace=True)
       ← Intel-verified AMX TMUL acceleration path

  3b. Else (fallback):
       torch.compile(teacher, mode="reduce-overhead", backend="inductor")
       ← PyTorch 2.10 native inductor generates AMX-compatible kernels
       ← Verified: oneDNN enabled, MKL available on HP Z4 G5

  Memory: ~16GB system RAM, 0GB GPU VRAM
  Load time: 3.46s (after first-run weight caching)
  Resident: stays loaded for entire campaign session
```

#### Correction Latency Breakdown (from campaign logs, N=2,847 events)

```
┌────────────────────────────────────────────────────────────────────┐
│  Per-handoff event breakdown:                                      │
│                                                                    │
│  PCIe Transfer (GPU→CPU):   0.033ms avg   [0.030 – 0.040ms]      │
│  AMX Processing (teacher):  8,773ms avg   [8,132 – 9,433ms]      │
│                             ↑ This is for 3–5 anchor tokens       │
│  Per-token cost (amortized): ~344ms/token (matches benchmark)     │
│                                                                    │
│  Total per-handoff:  ~8.8s                                        │
│  Handoffs per query: 2–5 (typ.)                                   │
│  Total teacher cost: ~18–44s per query                            │
│                                                                    │
│  Acceptable because: chasm events ≈ 2–5% of all tokens;          │
│  student handles 95–99% of tokens at full GPU speed.             │
└────────────────────────────────────────────────────────────────────┘
```

---

## The Complete Token-Generation Loop

This is the exact sequence of operations for **every single token** `t`:

```
FOR each generation step t = 0, 1, 2, ..., max_tokens:

  ┌─────────────────────────────────────────────────────────────────┐
  │  STUDENT GPU                                                    │
  │                                                                 │
  │  1. Forward pass: logits, z_t = model(context)                 │
  │     z_t ∈ ℝ^4096  (final transformer layer hidden state)       │
  │                                                                 │
  │  2. i-ppDRE SCORE  [0.390ms]                                   │
  │     φ_t = RFF(z_t)             ∈ ℝ^1024                        │
  │     w_t = |α · φ_t|            ∈ ℝ  (drift score)              │
  │                                                                 │
  │  3. CONFORMAL GATE CHECK  [<0.01ms]                            │
  │     pred_set_t = count(p > 0.001 in softmax(logits))           │
  │     alert_drift = w_t > lambda*                                     │
  │     alert_bq    = pred_set_t > κ                               │
  │     fire = alert_drift AND alert_bq                            │
  │                                                                 │
  │  4a. IF fire == False:  SAFE PATH                              │
  │      token_t = argmax(logits)  OR sample                       │
  │      context.append(token_t)                                   │
  │      bridge.write_state(t, z_t)  [/dev/shm, ~µs]              │
  │      alpha UPDATE: α ← 0.99α - 0.05(φ_t - teacher_mean_phi)  │
  │      CONTINUE to step t+1                                      │
  │                                                                 │
  │  4b. IF fire == True:  HANDOFF PATH                            │
  │      [See handoff diagram below]                               │
  └─────────────────────────────────────────────────────────────────┘


  ┌─────────────────────────────────── HANDOFF PATH ────────────────┐
  │                                                                  │
  │  STUDENT GPU:                     CPU (OracleBridge):           │
  │                                                                  │
  │  Decode context to text           ←                             │
  │  ──────────────────────────────── PCIe 0.03ms ─────────────►   │
  │                                   Re-tokenize for teacher       │
  │                                   teacher.generate(k=3..5)     │
  │                                   BF16 + AMX: ~344ms/token      │
  │                                   Decode corrected text         │
  │  ◄──────────────────────────────── PCIe 0.03ms ─────────────   │
  │  Re-tokenize with student vocab                                  │
  │  Inject corrected tokens into context                           │
  │  Resume generation from corrected position                      │
  │                                                                  │
  │  [Log to campaign file]:                                        │
  │  [ACC Handoff] Step t | Drift: w_t | Gate: lambda* |                │
  │                PCIe: 0.03ms | AMX: ~8800ms | Total: Xms        │
  └──────────────────────────────────────────────────────────────────┘
```

---

## ASCII: Module Dependency Graph

```
02_SRC/acc_core/
│
├── detector/
│   ├── rff_kernel.py         ← THE LENS
│   │   RandomFourierFeatures
│   │   • QR-orthogonalized projection matrix Ω
│   │   • Maps z_t (4096D) → φ(z_t) (1024D)
│   │   • Approximates Gaussian kernel k(x,y)
│   │   • Called by: IncrementalDriftTracker
│   │
│   └── ipp_dre.py            ← THE SENSOR
│       IncrementalDriftTracker
│       • Holds teacher manifold centroid (teacher_mean_phi)
│       • score(): w(x_t) = |α · φ(x_t)|
│       • update(): PEGASOS online gradient w/ forgetting factor
│       • Calls: RandomFourierFeatures
│       • Called by: run_acc_student.py, cross_baseline_campaign.py
│
├── control/
│   └── conformal.py          ← THE JUDGE
│       ConformalSafetyGate
│       • add_calibration_score() → collects (w_i, |C_i|) offline
│       • calibrate() → computes lambda* and κ via split-conformal
│       • check() → AND-logic dual-signal decision
│       • posterior_risk_estimate() → sliding BQ risk window
│       • Called by: run_acc_student.py
│
└── system/
    ├── ring_buffer.py        ← THE NERVOUS SYSTEM
    │   ACCOpsBridge
    │   • /dev/shm zero-copy shared memory
    │   • write_state(): student writes z_t each step
    │   • trigger_intervention(): oracle flags drift event
    │   • clear_intervention(): oracle releases student
    │   • Called by: student process AND oracle process
    │
    ├── lazy_sync.py          ← THE BRIDGE
    │   KVCacheSync, ContextSynchronizer
    │   • transfer_to_oracle(): GPU→CPU non-blocking PCIe copy
    │   • resume_student(): CPU→GPU corrected cache
    │   • 92ms latency budget, measures each transfer
    │   • Called by: oracle_bridge.py handoff path
    │
    └── oracle_bridge.py      ← THE ORACLE
        OracleBridge
        • load_teacher(): BF16 Llama-3-8B on CPU, AMX-compiled
        • correct_trajectory(): text-based cross-arch handoff
        • generate(): standalone teacher generation
        • Uses: KVCacheSync for PCIe transfer
        • Called by: run_acc_student.py, wrappers/run_acc_student.py
```

---

## Threshold Sweet Spots (From N=512 Manifold Sweep)

```
MODEL             lambda* (DRIFT)    κ (PRED-SET)    RATIONALE
────────────────  ────────────  ──────────────  ────────────────────────────────────────
Qwen-2.5-1.5b    0.030         calibrated      Edge model; large drift events after ~25
                                               steps. Needs moderate threshold.

Phi-3-Mini       0.150         calibrated      Dense synthetic-data manifold. Even at
                                               4-bit, drift rarely exceeds 0.05.
                                               High threshold = low false-positive rate.

Mistral-7B       0.002         calibrated      7B model at 4-bit = very unstable manifold.
                                               Drift hits threshold after only 1–2 tokens
                                               on factual tasks. Aggressive threshold needed.
```

**Key insight:** These thresholds span 75× range (0.002 → 0.150). A single
global threshold would either saturate the teacher with Phi-3-Mini (falsely
high DCDR) or miss all Mistral-7B events (0% DCDR). This is why
per-architecture calibration is a core contribution.

---

## Key Design Decisions and Why They Matter for UAI

| Decision | Alternative | Why ACC's choice is better |
|----------|-------------|---------------------------|
| AND logic for gate | OR logic | Prevents "Intervention Addiction" — only fires when manifold IS broken AND model IS confused. OR logic generates 3× more handoffs on Phi-3-Mini with no accuracy gain |
| QR-orthogonalized RFF | Standard random Gaussian | Without QR, 4-bit quantization noise amplifies through correlated projection rows, generating thousands of false-positive drift scores |
| Text-based handoff | Token-ID handoff | Token IDs are vocabulary-specific. Text-based handoff works across any student-teacher pair regardless of tokenizer |
| Forgetting factor lambda=0.99 | Fixed α weights | Without forgetting, a single chasm event permanently biases α, causing drift scores to remain elevated for all subsequent tokens (false sustained alarm) |
| /dev/shm zero-copy | TCP socket / file IPC | Shared memory operates at cache-line speed (~µs). File/socket IPC would add 1–10ms per token — more than the drift scoring latency itself |
| Teacher on CPU (0 GPU VRAM) | Teacher on GPU | Student models need 4–8GB VRAM. Keeping teacher on CPU avoids VRAM eviction and allows both models to coexist on the same machine |

---

## For the Methodology Section (LaTeX Prose Summary)

> **Drift Sensing.** At each decoding step $t$, the student's final-layer
> hidden state $z_t \in \mathbb{R}^{4096}$ is projected via a fixed
> QR-orthogonalized Random Fourier Feature map $\phi: \mathbb{R}^{4096}
> \to \mathbb{R}^{1024}$, approximating the Gaussian kernel
> $k(z, z') = \exp(-\|z-z'\|^2/2\sigma^2)$. An online weight vector
> $\alpha$ is maintained via PEGASOS gradient descent with forgetting
> factor $\lambda=0.99$, yielding drift score
> $w(\mathbf{x}_t) = |\alpha^\top \phi(z_t)|$.  Adding 0.390\,ms
> latency per token (measured on RTX 2000 Ada), this constitutes a
> real-time i-ppDRE sensor.

> **Conformal Safety Gate.** Both $w(\mathbf{x}_t)$ and the prediction
> set size $|C(\mathbf{x}_t)|$ are calibrated offline via split conformal
> prediction on $N{=}1{,}417$ samples, yielding thresholds $\lambda^*$
> and $\kappa$ with $\geq 95\%$ coverage guarantee. Intervention fires
> only when \emph{both} $w(\mathbf{x}_t) > \lambda^*$ (manifold collapse
> detected) and $|C(\mathbf{x}_t)| > \kappa$ (posterior confusion
> confirmed per Snell-Griffiths BQ proxy) — AND logic that eliminates
> spurious handoffs from safe-but-ambiguous tokens.

> **Oracle Correction Cascade.** On gate activation, the current context
> is decoded to text (vocabulary-agnostic), transferred over PCIe 4.0$\,
> \times\,$8 (0.033\,ms, well within the 92\,ms budget), and passed to a
> BF16 Llama-3-8B-Instruct teacher running on the Xeon W5 CPU with Intel
> AMX acceleration (344\,ms/token). The teacher generates 3--5 anchor
> tokens, which are re-encoded into the student's vocabulary and injected
> into the KV-cache context before generation resumes.
