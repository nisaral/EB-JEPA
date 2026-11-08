# Draft Hugging Face card updates (not uploaded)

These are proposed replacements for the cards on the Hub. **Nothing here has been pushed to Hugging Face.**

| File | Hub repo | Changes compared with the live card (as of 2026-10-30) |
| :--- | :--- | :--- |
| `soro-kev-0.8b.README.md` | `nisaralll/soro-kev-0.8b` | Loss formula now states the JSD weight (0.30); flags the human split as a duplicate; adds a Limitations section (115 distinct queries with cluster CIs, τ² at 10.5%, 17/33 refusal, q̂ provenance) |
| `soro-kev-4b.README.md` | `nisaralll/soro-kev-4b` | Same, plus: the 4B model was trained in `single_sequence` mode, so its JSD weight was 0 |
| `soro-universal-tool-harness.README.md` | `nisaralll/soro-universal-tool-harness` (dataset) | Flags `val_external_human.jsonl` as a duplicate of held-out; links DATA_LICENSES.md |

To publish one after review:

```bash
huggingface-cli upload nisaralll/soro-kev-0.8b docs/hf_cards/soro-kev-0.8b.README.md README.md
```

`releases/*/README.md` hold the same text.
