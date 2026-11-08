# Reconciled & Verified Benchmark Evaluation Tables

> **Verification Note**: Every figure in these tables is directly computed from committed per-row
> prediction logs in `benchmarks/predictions/`. Each percentage is presented with its exact integer
> rational fraction `(n/N)`. Subgroup row counts and correct counts strictly partition overall totals
> with mathematical consistency checks enforced via `reconcile_and_verify.py`.

### Table 1: External Held-Out Generalization (1,200 rows across 115 unique base queries)

> **Note on Query Stratification**: The 1,200 evaluations comprise 115 unique base queries tested across ~10 format/distractor views
> (1,095 reference MCP rows + 105 $\tau^2$ persona-dialogue rows). Rows of one base query are correlated; see `per_query_ci.json`.

| System | Overall (1,200) | Non-Abstain (1,065) | Abstain Refusal (135)* | Strictly Disjoint Tools (1,038) | Shared Tool Names (162) | Conformal Coverage @ 90% | Mean Set Size |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BM25 (fixed abstain τ)** | 58.67% (704/1200) | 62.72% (668/1065) | 26.67% (36/135) | 58.29% (605/1038) | 61.11% (99/162) | 84.1% ± 2.4% | 4.32 ± 0.21 |
| **Dense MiniLM-L6 cosine (fixed abstain τ)** | 79.17% (950/1200) | 84.51% (900/1065) | 37.04% (50/135) | 81.70% (848/1038) | 62.96% (102/162) | 90.4% ± 1.6% | 1.54 ± 0.10 |
| **NLI cross-encoder DeBERTa-v3-small (fixed abstain τ)** | 24.00% (288/1200) | 17.56% (187/1065) | 74.81% (101/135) | 26.88% (279/1038) | 5.56% (9/162) | 90.3% ± 1.7% | 4.83 ± 0.10 |
| **Qwen3.5-0.8B-Base option letters (untrained)** | 20.75% (249/1200) | 23.38% (249/1065) | 0.00% (0/135) | 20.13% (209/1038) | 24.69% (40/162) | 90.4% ± 1.7% | 5.06 ± 0.09 |
| **Soro-Kev-0.8B (Ours)** | 85.67% (1028/1200) | 86.95% (926/1065) | 75.56% (102/135) | 88.63% (920/1038) | 66.67% (108/162) | 90.3% ± 1.6% | 1.14 ± 0.04 |
| **Soro-Kev-4B (Ours)** | 90.58% (1087/1200) | 90.52% (964/1065) | 91.11% (123/135) | 92.77% (963/1038) | 76.54% (124/162) | 91.0% ± 1.1% | 1.01 ± 0.02 |

*\*Note on Abstention: The 135 abstain rows represent 15 unique base queries evaluated across 9 format/distractor views each ($15 \times 9 = 135$). Denominators strictly reconcile: $1,065 + 135 = 1,200$. Numerators strictly reconcile: e.g. 0.8B gets $926 + 102 = 1,028$; 4B gets $964 + 123 = 1,087$. Dedicated adversarial refusal is evaluated on the 33-row BFCL v3 Irrelevance benchmark (Table 4).*

### Table 2: External Held-Out Per-Package Breakdown (1,200 rows)

| System | system_mcp (370) | geo_mcp (364) | media_mcp (361) | tau2/telecom (85) | tau2/retail (20) | Total Held-Out (1,200) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **BM25 (fixed abstain τ)** | 64.59% (239/370) | 56.32% (205/364) | 56.51% (204/361) | 52.94% (45/85) | 55.00% (11/20) | 58.67% (704/1200) |
| **Dense MiniLM-L6 cosine (fixed abstain τ)** | 81.89% (303/370) | 85.16% (310/364) | 83.66% (302/361) | 37.65% (32/85) | 15.00% (3/20) | 79.17% (950/1200) |
| **NLI cross-encoder DeBERTa-v3-small (fixed abstain τ)** | 20.27% (75/370) | 25.55% (93/364) | 31.30% (113/361) | 7.06% (6/85) | 5.00% (1/20) | 24.00% (288/1200) |
| **Qwen3.5-0.8B-Base option letters (untrained)** | 20.81% (77/370) | 19.23% (70/364) | 21.61% (78/361) | 21.18% (18/85) | 30.00% (6/20) | 20.75% (249/1200) |
| **Soro-Kev-0.8B (Ours)** | 87.57% (324/370) | 95.33% (347/364) | 95.84% (346/361) | 9.41% (8/85) | 15.00% (3/20) | 85.67% (1028/1200) |
| **Soro-Kev-4B (Ours)** | 95.68% (354/370) | 98.35% (358/364) | 96.68% (349/361) | 24.71% (21/85) | 25.00% (5/20) | 90.58% (1087/1200) |

*\*Note on Tau-2 persona simulation: Across the 1,095 reference MCP tools (system, geo, media), Soro-Kev-4B achieves 96.89% (1,061/1,095). On the 105 Tau-2 multi-turn dialogue instructions, Soro-Kev correctly refuses to blindly fire backend mutation tools without customer identifiers: 64.56% (51/79) of Soro-Kev-4B's misses were explicit abstentions (21 rows) or preliminary read/status lookups (30 rows, e.g. check_service_status, get_device_details) rather than premature account mutations.*

### Table 3: Unseen Tool Package Split (2,220 evaluations across 55 held-out packages)

> **Note on Queries & Schemas**: Queries share structural template distributions with training while all 55 tool schemas
> and packages are strictly disjoint (0 package overlap with the 319 training packages).

| System | Overall Zero-Shot (2,220 rows) | Clean Menus (2,168 rows) | Collision Distractors (52 rows) |
| :--- | :---: | :---: | :---: |
| **BM25 (fixed abstain τ)** | 48.60% (1079/2220) | 49.40% (1071/2168) | 15.38% (8/52) |
| **Dense MiniLM-L6 cosine (fixed abstain τ)** | 62.48% (1387/2220) | 62.82% (1362/2168) | 48.08% (25/52) |
| **NLI cross-encoder DeBERTa-v3-small (fixed abstain τ)** | 13.60% (302/2220) | 13.75% (298/2168) | 7.69% (4/52) |
| **Qwen3.5-0.8B-Base option letters (untrained)** | 17.57% (390/2220) | 17.57% (381/2168) | 17.31% (9/52) |
| **Soro-Kev-0.8B (Ours)** | 81.67% (1813/2220) | 81.83% (1774/2168) | 75.00% (39/52) |
| **Soro-Kev-4B (Ours)** | 88.06% (1955/2220) | 88.28% (1914/2168) | 78.85% (41/52) |
| **Arch-Router-1.5B (Katanemo)** | - | - | 25.00% (13/52) |
| **xLAM-2-1b-fc-r (Salesforce)** | - | - | 69.23% (36/52) |
| **Hammer2.1-1.5b (MadeAgents)** | - | - | 59.62% (31/52) |

### Table 4: Berkeley Function-Calling Leaderboard (BFCL v3) Irrelevance & Abstention Diagnostic (100 rows)

> **Abstention Methodology**: The external held-out split contains only 15 abstain rows (1.25% of 1,200),
> which is insufficient to ground general abstention claims. Refusal and abstention capabilities are evaluated
> against 33 dedicated adversarial irrelevance distractors from official BFCL v3.

| System | Multiple Tools (34) | Live Multiple (33) | Irrelevance / Abstention (33) | Overall Diagnostic (100) | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **BM25 (fixed abstain τ)** | 94.12% (32/34) | 57.58% (19/33) | 36.36% (12/33) | 63.00% (63/100) | 0.1 ms |
| **Dense MiniLM-L6 cosine (fixed abstain τ)** | 97.06% (33/34) | 84.85% (28/33) | 24.24% (8/33) | 69.00% (69/100) | 19.6 ms |
| **NLI cross-encoder DeBERTa-v3-small (fixed abstain τ)** | 55.88% (19/34) | 30.30% (10/33) | 33.33% (11/33) | 40.00% (40/100) | 317.0 ms |
| **Qwen3.5-0.8B-Base option letters (untrained)** | 44.12% (15/34) | 33.33% (11/33) | 0.00% (0/33) | 26.00% (26/100) | 456.8 ms |
| **Soro-Kev-0.8B (Ours)** | 100.00% (34/34) | 96.97% (32/33) | 51.52% (17/33) | 83.00% (83/100) | 124.2 ms |
| **Soro-Kev-4B (Ours)** | 100.00% (34/34) | 96.97% (32/33) | 75.76% (25/33) | 91.00% (91/100) | 648.6 ms |
| **Arch-Router-1.5B (Katanemo)** | 11.76% (4/34) | 15.15% (5/33) | 96.97% (32/33) | 41.00% (41/100) | 8465.4 ms |
| **xLAM-2-1b-fc-r (Salesforce)** | 79.41% (27/34) | 84.85% (28/33) | 30.30% (10/33) | 65.00% (65/100) | 11453.6 ms |
| **Hammer2.1-1.5b (MadeAgents)** | 97.06% (33/34) | 93.94% (31/33) | 27.27% (9/33) | 73.00% (73/100) | 10564.5 ms |

> *Latency* is the median measured wall-clock of each system's own forward pass(es) on an RTX 4050 Laptop GPU
> (`time.perf_counter()` with CUDA sync), as recorded per row in the prediction files. Generative baselines
> (Arch-Router, xLAM, Hammer) include their decode steps. Baseline abstain thresholds are fixed constants, not fitted.
> Soro-Kev rows were scored with the v0.2.1 inference prompt. The training-aligned prompt (default since v0.2.2)
> changed accuracy by at most 5 of 300 rows per split in a 700-row A/B on both checkpoints.
