"""Data pipeline for Soro-Kev-Omni (text, image, audio decision rows).

Stages (see docs/OMNI_DATA.md):
  schema      one row format for every modality, with provenance and license
  licenses    source registry: what may be trained on, what is eval-only
  quality     dedupe, near-duplicate removal, train/eval leakage, balance, label audit
  text        collision mining, near-miss negatives, paraphrase capping
  teacher     permutation-averaged soft labels from open teachers (text / image / audio)
  generate    RLCD-style contrastive request writing, modality-necessity filter
  audio       TTS rendering, augmentation, streaming prefixes, Audio2Tool (eval)
  image       OS-Atlas UI-element pointing rows, image-dependent routing rows
  build       mixture targets, group-disjoint splits, shards, stats and card
"""

from soro.omni_data.schema import OmniRow, Option, read_rows, write_rows

__all__ = ["OmniRow", "Option", "read_rows", "write_rows"]
