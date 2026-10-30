# Open questions

These come from the v0.2.1 audit. Each one needs an answer from the maintainer, or data that is
not in the repo. None of them changes the code.

1. **Which external file were the tables scored on?** The repo has only `val_external_human.jsonl`.
   The README cites `val_external_heldout.jsonl`. On HF the two are byte-identical, so the numbers
   do not change, but the naming should be made consistent.
2. **A real human split.** Should the duplicated "human" file be removed from the HF dataset, or
   should it be replaced with about 300 genuinely human-written queries?
3. **Calibration provenance.** The HF `config.json` says the deployed q̂ (0.9518 for 0.8B, 0.8066
   for 4B) was fitted on a 100-row seed-42 slice of `val_zeroshot`. README § Benchmarks says
   "100 random splits, N_cal = 500". Which is right for each table?
4. **Gate versus argmax.** Which accuracy tables were scored with the conformal gate, and which
   with plain top-1 argmax? (The README already carries this as a TODO.)
5. **Licenses.** Upstream licenses for MCPAgentBench, `catalog_paraphrase`, Toucan and ToolACE.
   See `DATA_LICENSES.md`.
6. **Does the 0.8B abstention regression matter for the target use?** On BFCL irrelevance the
   0.8B model refuses 17/33, versus 27/33 for base `kev-0.8b`. Options: raise q̂, add more
   no-tool training rows, or recommend the 4B model when false calls are costly.
   See `docs/RETRAIN_ANALYSIS.md`.
7. **τ² performance.** τ² rows score 10.5% (0.8B) and 24.8% (4B). Is τ² in scope? If it is, it
   needs targeted data (see `docs/RETRAIN_ANALYSIS.md`).
8. **Order sensitivity.** The 1.5–3.0% order-flip rate quoted in the README has no committed
   script that reproduces it.
9. **4B JSD.** The 4B model was trained in `single_sequence` mode (no JSD term). Is its order-flip
   rate measured separately from the 0.8B model's?
10. **Release mechanics.** Tag `v0.2.1`, publish to PyPI, update the HF cards from `docs/hf_cards/`,
    and upload the social preview. All four are pending the maintainer's decision.
11. **External q̂ mismatch.** Refitting LAC q̂ (α = 0.10, s = 1 − p_true) on `benchmarks/predictions/external_heldout`
    gives 0.9546 (0.8B) and 0.4825 (4B). `config.json` lists `external_human_q_hat` = 0.9813 and 0.845. Was a
    different score, row subset or α used?
