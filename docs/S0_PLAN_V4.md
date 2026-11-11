# S0 notes (PLAN v4)

Pointer A only. Action cards ANSWER / ASK / ESCALATE / ABSTAIN are always in the menu (S2). Error dump uses released Soro-Kev-4B (query-first pointer). The 1k Qwen3-0.6B A run did not save weights.

## Papers (one line each)

- **TRUST (2606.06976):** Puts uncertainty into the RL reward so wrong tool calls stay uncertain; we must beat When2Call-style decision quality with **exact expected cost** (S2), not GRPO, and keep conformal abstain calibrated instead of overconfident mistakes.
- **Role-stratified CRC (2607.24343):** Certifies each argument *role* (recipient vs body) with its own conformal budget; cite as the argument-level counterpart of our tool-level LAC sets, and do not claim per-field guarantees until S1/S4 touch arguments.
- **When2Call (2504.18851):** Four-way choice (call / ask / abstain / direct-answer); SOTA generators still hallucinate calls. S2 must beat this with the same four actions always in the menu, as a pointer, not a generator.
- **ClarifyBench / SAGE (2511.08798):** Structured uncertainty over *parameters* (EVPI vs question cost) for when to ASK; we cite it, and S2's ASK card is the discrete action, not free-form clarifying questions.
- **LettuceDetect follow-on (2607.00895):** Span-level hallucination over tool output/code, not just RAG docs; S4's RESULT_FAILED / USE_PRIOR gate must beat a span detector used as a baseline on tool results.
- **TRACER (2602.11409):** Trajectory-level risk on tau2-bench (critical episodes, not token entropy); S5 compares our per-turn conformal |S| vs first fail against TRACER, without multi-turn GRPO unless last.

## Error dump

See `veyra/runs/s0_a_errors/` (filled by `veyra/training/dump_a_errors.py`).
