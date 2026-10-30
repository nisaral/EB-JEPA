# Soro-Kev dataset notes

## Known issue: duplicated "human" split

On the Hugging Face dataset `nisaralll/soro-universal-tool-harness`, `val_external_human.jsonl`
was found byte-identical (audit of 30 Oct 2026) to `val_external_heldout.jsonl`. It is a copy of the external held-out
split, not an independent set of human-written queries. Results labelled "human" should be
read as held-out results until a real human split is collected.

TODO(Keyush): either remove `val_external_human.jsonl` from the HF dataset or replace it with
genuinely human-written queries, then update this note and the dataset card.

This repo tracks only `val_external_human.jsonl` (no `val_external_heldout.jsonl`).
TODO(Keyush): confirm which file the README's "External Held-Out" tables were scored on.

## Upstream licenses

Some rows are converted from tau2-bench, BFCL and MCP benchmark sources. TODO(Keyush): list
each upstream source's license here; the Apache-2.0 tag on the dataset may not apply to
every upstream row.
