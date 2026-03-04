/home/cse-sdpl/research/ACC/
├── 00_DOCS/
│   ├── daily_reports/
│   │   ├── 2026-02-02.txt
│   │   ├── 2026-02-03.md
│   │   ├── 2026-02-09.md
│   │   └── 2026-02-10.md
│   ├── directory/
│   │   ├── directory.md
│   │   └── full_tree.txt
│   ├── math_proofs/
│   ├── hp_z4_hardware_constraints.md
│   ├── Master_Implementation_Plan.md
│   └── uai_2026_roadmap.md
├── 01_DATA/
│   ├── benchmark_tasks.json
│   ├── generate_benchmark_100.py
│   ├── models/
│   │   ├── student_any4/
│   │   │   └── models--meta-llama--Meta-Llama-3-8B-Instruct/
│   │   └── teacher_fp16/
│   └── raw/
├── 02_SRC/
│   ├── acc_core/
│   │   ├── __init__.py
│   │   ├── control/
│   │   │   ├── __init__.py
│   │   │   ├── bayesian_quad.py
│   │   │   └── conformal.py
│   │   ├── detector/
│   │   │   ├── __init__.py
│   │   │   ├── ipp_dre.py
│   │   │   └── rff_kernel.py
│   │   └── system/
│   │       ├── __init__.py
│   │       ├── lazy_sync.py
│   │       └── ring_buffer.py
│   └── wrappers/
│       ├── baseline_01_any4.py
│       ├── baseline_02_saup.py
│       ├── baseline_03_splitwise.py
│       ├── campaign_logger.py
│       ├── oracle_campaign_runner.py
│       ├── oracle_cpu_monitor.py
│       ├── run_acc_student.py
│       ├── setup_student.py
│       └── student_gpu_agent.py
├── 03_BASELINES/
│   ├── README.md
│   ├── any4/
│   ├── saup/
│   └── splitwise-sim/
├── 04_ENV/
│   ├── docker/
│   └── scripts/
├── 05_EXPERIMENTS/
│   ├── phase_1_calibration/
│   ├── phase_2_baselines/
│   └── phase_3_acc_campaign/
├── 06_RESULTS/
│   ├── analysis/
│   └── logs/
├── benchmark_results/
├── run_100_benchmark.sh
├── start_vllm_server.sh
├── vllm_server.log
└── README.md

Note: The full, exhaustive listing (including all child files) is available in
00_DOCS/directory/full_tree.txt for complete reference.