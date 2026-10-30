"""Soro-Kev Neural Architecture: Single-Pass Causal Transformer + Dual Pointer Projection Head.

Implements the System-One decision model (based on jaredpalmer/kev / Qwen3.5 + Pointer Head)
with single-pass causal attention, exact token-span slicing, scaled dot-product attention,
and native PyTorch checkpoint loading/saving without external dependencies.
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_calibrated_pointer_loss(
    route_ce: torch.Tensor,
    brier: torch.Tensor,
    jsd: torch.Tensor,
    arg_ce: torch.Tensor,
    missing_ce: torch.Tensor,
    mutation_ce: torch.Tensor,
    intent_ce: torch.Tensor | None = None,
    jsd_weight: float = 0.30,
) -> torch.Tensor:
    """Multi-Task Calibrated Pointer Loss:
    L = L_route_CE + 0.25 L_Brier + jsd_weight * L_JSD + 0.50 L_arg_CE + 0.25 L_missing_CE + 0.25 L_mutation_CE + 0.40 L_intent_CE
    """
    if intent_ce is None:
        intent_ce = route_ce * 0.0
    return (
        route_ce
        + 0.25 * brier
        + jsd_weight * jsd
        + 0.50 * arg_ce
        + 0.25 * missing_ce
        + 0.25 * mutation_ce
        + 0.40 * intent_ce
    )


def compute_rlcd_loss(
    route_ce: torch.Tensor,
    brier: torch.Tensor,
    jsd: torch.Tensor,
    arg_ce: torch.Tensor,
    missing_ce: torch.Tensor,
    mutation_ce: torch.Tensor,
    intent_ce: torch.Tensor | None = None,
) -> torch.Tensor:
    """Backward compatibility alias for compute_calibrated_pointer_loss."""
    return compute_calibrated_pointer_loss(
        route_ce, brier, jsd, arg_ce, missing_ce, mutation_ce, intent_ce, jsd_weight=0.30
    )


class MockCausalBackbone(nn.Module):
    """Causal Transformer sequence backbone for smoke tests and CPU validation.
    
    Uses standard multi-head causal self-attention and positional embeddings so
    shuffling token order directly alters representations h_opt_i via attention,
    producing realistic non-zero permutation JSD (0.02 - 0.15).
    """

    def __init__(
        self,
        vocab_size: int = 2000,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 2,
            batch_first=True,
            dropout=0.0,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        # input_ids: (B, seq_len)
        B, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = self.embedding(input_ids) + self.pos_emb(positions)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=input_ids.device).bool()
        pad_mask = (attention_mask == 0) if attention_mask is not None else None
        h_seq = self.transformer(x, mask=causal_mask, src_key_padding_mask=pad_mask, is_causal=True)
        return self.proj(h_seq)


class SoroKevModel(nn.Module):
    """Soro-Kev: Full-Sequence Causal Backbone + Dual Linear Pointer Projection Head."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B",
        hidden_size: int = 1024,
        proj_dim: int = 256,
        use_mock: bool = False,
        torch_dtype: torch.dtype | None = None,
        allow_mock_fallback: bool = True,
        load_in_4bit: bool = False,
    ):
        super().__init__()
        self.model_name = model_name
        self.hidden_size = hidden_size
        self.proj_dim = proj_dim
        self.use_mock = use_mock
        self.tokenizer = None
        # Training may pass a short cap. Inference never uses a cap under 4096.
        self.inference_max_seq_len = 4096

        if use_mock:
            self.backbone = MockCausalBackbone(vocab_size=2000, hidden_size=128)
            backbone_dim = 128
        else:
            try:
                from transformers import AutoModel, AutoTokenizer
                from peft import LoraConfig, get_peft_model

                # Kaggle's image has torchao 0.10. Current PEFT raises on that
                # during every LoRA wrap, even when the backbone is plain fp16.
                def _torchao_unused() -> bool:
                    return False

                try:
                    import peft.import_utils as _peft_imports
                    _peft_imports.is_torchao_available = _torchao_unused
                except Exception:
                    pass
                try:
                    import peft.tuners.lora.torchao as _peft_torchao
                    _peft_torchao.is_torchao_available = _torchao_unused
                except Exception:
                    pass

                dtype = torch_dtype or (
                    torch.bfloat16
                    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                    else torch.float32
                )
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

                load_kw: dict[str, Any] = {"trust_remote_code": True}
                want_4bit = bool(load_in_4bit or ("4b" in model_name.lower() and torch.cuda.is_available()))
                if want_4bit:
                    try:
                        import bitsandbytes  # noqa: F401
                        from transformers import BitsAndBytesConfig
                        load_kw["quantization_config"] = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_compute_dtype=torch.float16,
                        )
                        load_kw["device_map"] = {"": torch.cuda.current_device()}
                    except Exception:
                        want_4bit = False
                if not want_4bit:
                    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                        load_kw["torch_dtype"] = torch.bfloat16
                    elif torch.cuda.is_available():
                        load_kw["torch_dtype"] = torch.float16
                    else:
                        load_kw["torch_dtype"] = dtype
                loaded = None
                for attn_name in ("sdpa", "eager"):
                    try:
                        loaded = AutoModel.from_pretrained(
                            model_name, attn_implementation=attn_name, **load_kw
                        )
                        break
                    except TypeError:
                        continue
                if loaded is None:
                    loaded = AutoModel.from_pretrained(model_name, **load_kw)
                use_4bit = bool(load_kw.get("quantization_config"))
                # Qwen3.5 ships a vision tower this router never uses. Drop it
                # even in 4-bit, or the T4 is full before the first step.
                if hasattr(loaded, "language_model"):
                    base = loaded._modules.pop("language_model")
                    for name in list(getattr(loaded, "_modules", {})):
                        del loaded._modules[name]
                    del loaded
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    base = loaded
                if use_4bit:
                    from peft import prepare_model_for_kbit_training
                    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
                text_cfg = getattr(base, "config", None)
                backbone_dim = getattr(text_cfg, "hidden_size", None) or getattr(
                    getattr(text_cfg, "text_config", None), "hidden_size", hidden_size
                )
                lora_cfg = LoraConfig(
                    r=16,
                    lora_alpha=32,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                    lora_dropout=0.05,
                    bias="none",
                )
                if hasattr(base, "config"):
                    base.config.use_cache = False
                if hasattr(base, "gradient_checkpointing_enable"):
                    base.gradient_checkpointing_enable()
                self.backbone = get_peft_model(base, lora_cfg)
            except Exception:
                if not allow_mock_fallback:
                    raise
                self.tokenizer = None
                self.backbone = MockCausalBackbone(vocab_size=2000, hidden_size=128)
                backbone_dim = 128

        # Dual pointer projection heads
        self.q_proj_head = nn.Linear(backbone_dim, proj_dim)
        self.opt_proj_head = nn.Linear(backbone_dim, proj_dim)

        # Temperature parameter initialized to 0.07 (standard contrastive pointer scale)
        self.log_temp = nn.Parameter(torch.tensor(math.log(0.07)))

        # Binary head for noul / boolean questions
        self.binary_head = nn.Linear(proj_dim, 2)
        # single=0, multi=1, none=2. multi must not auto-execute a mutation.
        self.intent_head = nn.Linear(proj_dim, 3)

    def _tokenize_text(self, text: str) -> list[int]:
        words = re.findall(r"\w+", text.lower())
        return [abs(hash(w)) % 2000 for w in words] or [0]

    def encode_tokens(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if self.tokenizer is not None:
            return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
        return self._tokenize_text(text)

    def _encode_choice(
        self,
        state: str,
        instructions: str,
        criteria: dict[str, str],
        max_seq_len: int,
    ) -> tuple[list[int], int, list[int], list[str]] | None:
        opt_keys = list(criteria.keys())
        q_prefix = f"State: {state}\nQuestion: {instructions}\nOptions:\n"
        if self.tokenizer is not None:
            prefix_toks = self.tokenizer.encode(q_prefix, add_special_tokens=True)
        else:
            prefix_toks = self._tokenize_text(q_prefix)
        q_span_end = len(prefix_toks) - 1
        all_toks = list(prefix_toks)
        opt_span_ends = []
        for idx, k in enumerate(opt_keys):
            spec = criteria[k]
            if len(spec) > 400:
                spec = spec[:400]
            opt_str = f"[{idx}] {k}: {spec}\n"
            if self.tokenizer is not None:
                opt_toks = self.tokenizer.encode(opt_str, add_special_tokens=False)
            else:
                opt_toks = self._tokenize_text(opt_str)
            all_toks.extend(opt_toks)
            opt_span_ends.append(len(all_toks) - 1)
        if len(all_toks) > max_seq_len:
            return None
        return all_toks, q_span_end, opt_span_ends, opt_keys

    def score_choice_sequence(
        self,
        state: str,
        instructions: str,
        criteria: dict[str, str],
        max_seq_len: int = 4096,
    ) -> tuple[torch.Tensor, list[str]]:
        """Pointer scores for a tool menu.

        Menus larger than 16 tools, or menus that do not fit in max_seq_len,
        are split into chunks of 6 plus abstain. The top two names from each
        chunk meet in a final championship pass. A row is not dropped.
        """
        opt_keys = list(criteria.keys())
        if not opt_keys:
            return torch.tensor([]), []
        floor = int(getattr(self, "inference_max_seq_len", 4096) or 4096)
        max_seq_len = max(int(max_seq_len), floor)
        abstain_text = criteria.get("__abstain__", "None of the listed tools fit the user request.")
        tools = [k for k in opt_keys if k != "__abstain__"]
        packed = self._encode_choice(state, instructions, criteria, max_seq_len)
        if len(tools) <= 16 and packed is not None:
            return self._pointer_logits(*packed)

        winners: list[str] = []
        for start in range(0, max(len(tools), 1), 6):
            chunk = tools[start : start + 6]
            if not chunk:
                break
            sub = {k: criteria[k] for k in chunk}
            sub["__abstain__"] = abstain_text
            sub_pack = self._encode_choice(state, instructions, sub, max_seq_len)
            if sub_pack is None:
                sub = {k: k for k in chunk}
                sub["__abstain__"] = abstain_text
                sub_pack = self._encode_choice(state, instructions, sub, max_seq_len)
            if sub_pack is None:
                winners.extend(chunk[:2])
                continue
            logits, keys = self._pointer_logits(*sub_pack)
            ranked = sorted(
                ((float(logits[i]), keys[i]) for i in range(len(keys)) if keys[i] != "__abstain__"),
                reverse=True,
            )
            winners.extend(name for _, name in ranked[:2])
        final_keys = list(dict.fromkeys(winners))
        final = {k: criteria.get(k, k) for k in final_keys}
        final["__abstain__"] = abstain_text
        final_pack = self._encode_choice(state, instructions, final, max_seq_len)
        if final_pack is None:
            final = {k: k for k in final_keys}
            final["__abstain__"] = abstain_text
            final_pack = self._encode_choice(state, instructions, final, max_seq_len)
        if final_pack is None:
            keys = list(final)
            return torch.zeros(len(keys)), keys
        # The decision logits are only this championship pass, not chunk abstain scores.
        return self._pointer_logits(*final_pack)

    def _pointer_logits(
        self,
        all_toks: list[int],
        q_span_end: int,
        opt_span_ends: list[int],
        opt_keys: list[str],
    ) -> tuple[torch.Tensor, list[str]]:

        device = next(self.parameters()).device
        input_ids = torch.tensor([all_toks], dtype=torch.long, device=device)

        # 2. Single forward pass through causal backbone
        if isinstance(self.backbone, MockCausalBackbone):
            h_seq = self.backbone(input_ids)  # (1, seq_len, hidden_size)
        else:
            outputs = self.backbone(input_ids=input_ids)
            h_seq = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]

        # 3. Exact token-span slicing
        h_q = h_seq[0, q_span_end, :].float()  # (hidden_size,)
        h_opts = torch.stack([h_seq[0, pos, :].float() for pos in opt_span_ends], dim=0)  # (K, hidden_size)

        # 4. Pointer Head projection + Scaled dot-product
        q_vec = self.q_proj_head(h_q.unsqueeze(0))  # (1, proj_dim)
        k_mat = self.opt_proj_head(h_opts)          # (K, proj_dim)

        scale = 1.0 / math.sqrt(self.proj_dim)
        temp = torch.exp(self.log_temp) + 1e-4
        raw_dots = torch.matmul(k_mat, q_vec.squeeze(0))
        logits = raw_dots * scale / temp  # (K,)
        intent_logits = self.intent_head(q_vec).squeeze(0)
        intent_names = ("single", "multi", "none")
        intent_idx = int(torch.argmax(intent_logits).item())
        self._last_route = {
            "raw_logits": raw_dots.detach(),
            "scaled_logits": logits.detach(),
            "keys": list(opt_keys),
            "intent_logits": intent_logits.detach(),
            "intent": intent_names[intent_idx] if intent_idx < len(intent_names) else "single",
            "temperature": float(temp.detach().item()),
        }
        return logits, opt_keys

    def score_binary(self, state: str) -> torch.Tensor:
        """Compute 2-class logits for noul boolean questions."""
        if self.tokenizer is not None:
            toks = self.tokenizer.encode(state, add_special_tokens=True)[:512]
        else:
            toks = self._tokenize_text(state)[:512]

        device = next(self.parameters()).device
        input_ids = torch.tensor([toks], dtype=torch.long, device=device)

        if isinstance(self.backbone, MockCausalBackbone):
            h_seq = self.backbone(input_ids)
        else:
            outputs = self.backbone(input_ids=input_ids)
            h_seq = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]

        h_last = h_seq[0, -1, :].float()
        q_vec = self.q_proj_head(h_last.unsqueeze(0))
        return self.binary_head(q_vec).squeeze(0)  # (2,) [no, yes]

    def forward_batch(
        self,
        batch_items: list[dict[str, Any]],
        max_seq_len: int = 1024,
    ) -> dict[str, Any]:
        """Multi-question packed sequence forward pass across B records.

        Packs all 4 non-permuted questions (tool_route, arg_mapping, missing_required,
        is_mutation) in Sequence 1, and encodes tool_route_perm in Sequence 2.
        Batches all 2*B sequences in ONE single backbone forward pass.
        """
        B = len(batch_items)
        if B == 0:
            return {"loss": torch.tensor(0.0), "metrics": {}}

        device = next(self.parameters()).device
        scale = 1.0 / math.sqrt(self.proj_dim)
        temp = torch.exp(self.log_temp) + 1e-4

        packed_tokens_list: list[list[int]] = []
        perm_tokens_list: list[list[int]] = []

        # Store metadata for span extraction per item
        meta_packed: list[dict[str, Any]] = []
        meta_perm: list[dict[str, Any]] = []
        kept_indices: list[int] = []

        for item_i, item in enumerate(batch_items):
            state = item["state"]
            q_dict = item["questions"]
            q_route = q_dict["tool_route"]
            q_perm = q_dict["tool_route_perm"]
            q_arg = q_dict.get("arg_mapping")
            q_miss = q_dict.get("missing_required")
            q_mut = q_dict.get("is_mutation")

            # --- Sequence 1: Packed Main Sequence ---
            prefix_text = f"State: {state}\n"
            toks = self.encode_tokens(prefix_text, add_special_tokens=True)

            # 1. tool_route
            r_keys = list(q_route["criteria"].keys())
            q1_text = f"Question [tool_route]: {q_route.get('instructions', '')}\nOptions:\n"
            toks.extend(self.encode_tokens(q1_text))
            q_end_route = len(toks) - 1
            opt_ends_route = []
            for idx, k in enumerate(r_keys):
                o_str = f"[{idx}] {k}: {q_route['criteria'][k]}\n"
                toks.extend(self.encode_tokens(o_str))
                opt_ends_route.append(len(toks) - 1)

            # 2. arg_mapping
            q_end_arg = None
            opt_ends_arg = []
            arg_keys = []
            if q_arg:
                arg_keys = list(q_arg["criteria"].keys())
                q2_text = f"\nQuestion [arg_mapping]: {q_arg.get('instructions', '')}\nOptions:\n"
                toks.extend(self.encode_tokens(q2_text))
                q_end_arg = len(toks) - 1
                for idx, k in enumerate(arg_keys):
                    o_str = f"[{idx}] {k}: {q_arg['criteria'][k]}\n"
                    toks.extend(self.encode_tokens(o_str))
                    opt_ends_arg.append(len(toks) - 1)

            # 3. missing_required
            q_end_miss = None
            if q_miss:
                q3_text = f"\nQuestion [missing_required]: {q_miss.get('instructions', '')}\n"
                toks.extend(self.encode_tokens(q3_text))
                q_end_miss = len(toks) - 1

            # 4. is_mutation
            q_end_mut = None
            if q_mut:
                q4_text = f"\nQuestion [is_mutation]: {q_mut.get('instructions', '')}\n"
                toks.extend(self.encode_tokens(q4_text))
                q_end_mut = len(toks) - 1

            if len(toks) > max_seq_len:
                continue

            packed_tokens_list.append(toks)
            kept_indices.append(item_i)
            meta_packed.append({
                "q_end_route": q_end_route,
                "opt_ends_route": opt_ends_route,
                "route_keys": r_keys,
                "route_gold": q_route["label"],
                "intent": {"single": 0, "multi": 1, "none": 2}.get(item.get("intent") or "single", 0),
                "q_end_arg": q_end_arg,
                "opt_ends_arg": opt_ends_arg,
                "arg_keys": arg_keys,
                "arg_gold": q_arg["label"] if q_arg else None,
                "q_end_miss": q_end_miss,
                "miss_gold": 1 if q_miss and q_miss.get("label") == "yes" else 0,
                "q_end_mut": q_end_mut,
                "mut_gold": 1 if q_mut and q_mut.get("label") == "yes" else 0,
            })

            # --- Sequence 2: Permuted Sequence ---
            p_keys = list(q_perm["criteria"].keys())
            q_perm_text = f"State: {state}\nQuestion [tool_route]: {q_perm.get('instructions', '')}\nOptions:\n"
            p_toks = self.encode_tokens(q_perm_text, add_special_tokens=True)
            q_end_perm = len(p_toks) - 1
            opt_ends_perm = []
            for idx, k in enumerate(p_keys):
                o_str = f"[{idx}] {k}: {q_perm['criteria'][k]}\n"
                p_toks.extend(self.encode_tokens(o_str))
                opt_ends_perm.append(len(p_toks) - 1)

            if len(p_toks) > max_seq_len:
                packed_tokens_list.pop()
                meta_packed.pop()
                kept_indices.pop()
                continue

            perm_tokens_list.append(p_toks)
            meta_perm.append({
                "q_end_perm": q_end_perm,
                "opt_ends_perm": opt_ends_perm,
                "perm_keys": p_keys,
                "perm_gold": q_perm["label"],
            })

        # Records that do not fit max_seq_len were dropped. Do not clamp spans.
        B = len(meta_packed)
        if B == 0:
            zero = sum((p.sum() * 0.0) for p in self.parameters())
            return {
                "loss": zero,
                "loss_route": 0.0,
                "loss_brier": 0.0,
                "loss_jsd": 0.0,
                "loss_arg": 0.0,
                "loss_missing": 0.0,
                "loss_mutation": 0.0,
                "metrics": {"route_acc": 0.0, "perm_flip_rate": 0.0, "arg_acc": 0.0, "missing_acc": 0.0, "mutation_acc": 0.0},
                "logits_route_batch": [],
                "kept_indices": [],
                "skipped": True,
            }

        # 4B on a T4 cannot afford the route sequence and its permutation together.
        # The permutation loss is zero for that step. The route loss still trains.
        single_sequence = bool(getattr(self, "single_sequence", False))
        all_sequences = packed_tokens_list if single_sequence else packed_tokens_list + perm_tokens_list
        max_len = max(len(s) for s in all_sequences)
        total_seqs = len(all_sequences)

        batch_input_ids = torch.zeros((total_seqs, max_len), dtype=torch.long, device=device)
        batch_attention_mask = torch.zeros((total_seqs, max_len), dtype=torch.long, device=device)

        for row_idx, s in enumerate(all_sequences):
            batch_input_ids[row_idx, :len(s)] = torch.tensor(s, dtype=torch.long, device=device)
            batch_attention_mask[row_idx, :len(s)] = 1

        # Single backbone forward pass!
        if isinstance(self.backbone, MockCausalBackbone):
            h_all = self.backbone(batch_input_ids, attention_mask=batch_attention_mask)
        else:
            outputs = self.backbone(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
            h_all = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]

        # Extract representations and compute logits & multi-task losses
        losses_route = []
        losses_jsd = []
        losses_brier = []
        losses_arg = []
        losses_miss = []
        losses_mut = []
        losses_intent = []

        route_hits = 0
        perm_flips = 0
        arg_hits = 0
        miss_hits = 0
        mut_hits = 0
        intent_hits = 0

        route_logits_list = []
        row_predictions: list[dict[str, Any]] = []

        for i in range(B):
            mp = meta_packed[i]
            h_packed = h_all[i]

            # 1. tool_route
            h_q1 = h_packed[mp["q_end_route"], :].float()
            h_opts1 = torch.stack([h_packed[pos, :].float() for pos in mp["opt_ends_route"]], dim=0)
            q1_vec = self.q_proj_head(h_q1.unsqueeze(0))
            k1_mat = self.opt_proj_head(h_opts1)
            logits_route = torch.matmul(k1_mat, q1_vec.squeeze(0)) * scale / temp
            route_logits_list.append(logits_route)
            intent_logits = self.intent_head(q1_vec).squeeze(0)
            intent_target = int(mp.get("intent", 0))
            p_intent = F.softmax(intent_logits, dim=-1)
            losses_intent.append(-torch.log(torch.clamp(p_intent[intent_target], min=1e-9)))
            if int(torch.argmax(intent_logits).item()) == intent_target:
                intent_hits += 1

            # Target route CE
            r_keys = mp["route_keys"]
            r_gold = mp["route_gold"]
            r_gold_idx = r_keys.index(r_gold) if r_gold in r_keys else 0
            p_route = F.softmax(logits_route, dim=-1)
            loss_r = -torch.log(torch.clamp(p_route[r_gold_idx], min=1e-9))
            losses_route.append(loss_r)

            pred_r_idx = int(torch.argmax(logits_route).item())
            if pred_r_idx == r_gold_idx:
                route_hits += 1

            # Brier loss
            num_opts = len(r_keys)
            gold_oh = torch.zeros(num_opts, device=device)
            gold_oh[r_gold_idx] = 1.0
            losses_brier.append(torch.mean((p_route - gold_oh) ** 2))

            # Permuted route & JSD
            if single_sequence:
                losses_jsd.append(torch.tensor(0.0, device=device))
                logits_perm = logits_route
                p_perm = p_route
                p_keys = r_keys
            else:
                mperm = meta_perm[i]
                h_perm = h_all[B + i]
                h_q_perm = h_perm[mperm["q_end_perm"], :].float()
                h_opts_perm = torch.stack([h_perm[pos, :].float() for pos in mperm["opt_ends_perm"]], dim=0)
                q_perm_vec = self.q_proj_head(h_q_perm.unsqueeze(0))
                k_perm_mat = self.opt_proj_head(h_opts_perm)
                logits_perm = torch.matmul(k_perm_mat, q_perm_vec.squeeze(0)) * scale / temp

                p_keys = mperm["perm_keys"]
                perm_map = {k: idx for idx, k in enumerate(p_keys)}
                p_perm = F.softmax(logits_perm, dim=-1)
                p_perm_aligned = torch.stack([p_perm[perm_map[k]] for k in r_keys])

                pred_perm_key = p_keys[int(torch.argmax(logits_perm).item())]
                if pred_perm_key != r_keys[pred_r_idx]:
                    perm_flips += 1

                p_safe = torch.clamp(p_route, min=1e-9)
                q_safe = torch.clamp(p_perm_aligned, min=1e-9)
                m_dist = 0.5 * (p_safe + q_safe)
                kl1 = torch.sum(p_safe * (torch.log(p_safe) - torch.log(m_dist)))
                kl2 = torch.sum(q_safe * (torch.log(q_safe) - torch.log(m_dist)))
                losses_jsd.append(torch.clamp(0.5 * (kl1 + kl2), min=0.0))

            # 2. arg_mapping
            if mp["q_end_arg"] is not None and mp["opt_ends_arg"]:
                h_q2 = h_packed[mp["q_end_arg"], :].float()
                h_opts2 = torch.stack([h_packed[pos, :].float() for pos in mp["opt_ends_arg"]], dim=0)
                q2_vec = self.q_proj_head(h_q2.unsqueeze(0))
                k2_mat = self.opt_proj_head(h_opts2)
                logits_arg = torch.matmul(k2_mat, q2_vec.squeeze(0)) * scale / temp
                arg_gold = mp["arg_gold"]
                arg_keys = mp["arg_keys"]
                arg_gold_idx = arg_keys.index(arg_gold) if arg_gold in arg_keys else 0
                p_arg = F.softmax(logits_arg, dim=-1)
                losses_arg.append(-torch.log(torch.clamp(p_arg[arg_gold_idx], min=1e-9)))
                if torch.argmax(logits_arg).item() == arg_gold_idx:
                    arg_hits += 1

            # 3. missing_required
            if mp["q_end_miss"] is not None:
                h_q3 = h_packed[mp["q_end_miss"], :].float()
                q3_vec = self.q_proj_head(h_q3.unsqueeze(0))
                logits_miss = self.binary_head(q3_vec).squeeze(0)  # (2,) [no, yes]
                miss_target = mp["miss_gold"]
                p_miss = F.softmax(logits_miss, dim=-1)
                losses_miss.append(-torch.log(torch.clamp(p_miss[miss_target], min=1e-9)))
                if torch.argmax(logits_miss).item() == miss_target:
                    miss_hits += 1

            # 4. is_mutation
            if mp["q_end_mut"] is not None:
                h_q4 = h_packed[mp["q_end_mut"], :].float()
                q4_vec = self.q_proj_head(h_q4.unsqueeze(0))
                logits_mut = self.binary_head(q4_vec).squeeze(0)  # (2,) [no, yes]
                mut_target = mp["mut_gold"]
                p_mut = F.softmax(logits_mut, dim=-1)
                losses_mut.append(-torch.log(torch.clamp(p_mut[mut_target], min=1e-9)))
                if torch.argmax(logits_mut).item() == mut_target:
                    mut_hits += 1

            answers: dict[str, Any] = {
                "tool_route": r_keys[pred_r_idx] if r_keys else None,
                "tool_route_perm": p_keys[int(torch.argmax(logits_perm).item())] if p_keys else None,
            }
            probabilities: dict[str, dict[str, float]] = {
                "tool_route": {key: float(p_route[idx]) for idx, key in enumerate(r_keys)},
                "tool_route_perm": {key: float(p_perm[idx]) for idx, key in enumerate(p_keys)},
            }
            if mp["q_end_arg"] is not None and mp.get("arg_keys"):
                answers["arg_mapping"] = mp["arg_keys"][int(torch.argmax(logits_arg).item())]
            if mp["q_end_miss"] is not None:
                answers["missing_required"] = "yes" if int(torch.argmax(logits_miss).item()) == 1 else "no"
                probabilities["missing_required"] = {
                    "no": float(p_miss[0]), "yes": float(p_miss[1]),
                }
            if mp["q_end_mut"] is not None:
                answers["is_mutation"] = "yes" if int(torch.argmax(logits_mut).item()) == 1 else "no"
                probabilities["is_mutation"] = {
                    "no": float(p_mut[0]), "yes": float(p_mut[1]),
                }
            row_predictions.append({"probabilities": probabilities, "answers": answers})

        loss_route_t = torch.stack(losses_route).mean() if losses_route else torch.tensor(0.0, device=device)
        loss_brier_t = torch.stack(losses_brier).mean() if losses_brier else torch.tensor(0.0, device=device)
        loss_jsd_t = torch.stack(losses_jsd).mean() if losses_jsd else torch.tensor(0.0, device=device)
        loss_arg_t = torch.stack(losses_arg).mean() if losses_arg else torch.tensor(0.0, device=device)
        loss_miss_t = torch.stack(losses_miss).mean() if losses_miss else torch.tensor(0.0, device=device)
        loss_mut_t = torch.stack(losses_mut).mean() if losses_mut else torch.tensor(0.0, device=device)
        loss_intent_t = torch.stack(losses_intent).mean() if losses_intent else torch.tensor(0.0, device=device)

        jsd_weight = 0.0 if single_sequence else 0.30
        total_loss = compute_calibrated_pointer_loss(
            loss_route_t, loss_brier_t, loss_jsd_t, loss_arg_t, loss_miss_t, loss_mut_t, loss_intent_t,
            jsd_weight=jsd_weight,
        )

        return {
            "loss": total_loss,
            "loss_route": float(loss_route_t.item()),
            "loss_brier": float(loss_brier_t.item()),
            "loss_jsd": float(loss_jsd_t.item()),
            "loss_arg": float(loss_arg_t.item()),
            "loss_missing": float(loss_miss_t.item()),
            "loss_mutation": float(loss_mut_t.item()),
            "loss_intent": float(loss_intent_t.item()),
            "parts": (loss_route_t, loss_brier_t, loss_jsd_t, loss_arg_t, loss_miss_t, loss_mut_t, loss_intent_t),
            "metrics": {
                "route_acc": float(route_hits / B),
                "perm_flip_rate": float(perm_flips / B),
                "arg_acc": float(arg_hits / max(len(losses_arg), 1)),
                "missing_acc": float(miss_hits / max(len(losses_miss), 1)),
                "mutation_acc": float(mut_hits / max(len(losses_mut), 1)),
                "intent_acc": float(intent_hits / max(len(losses_intent), 1)),
            },
            "logits_route_batch": route_logits_list,
            "kept_indices": kept_indices,
            "row_predictions": row_predictions,
            "skipped": False,
        }


def _flatten_head_weights(head_weights: dict[str, Any]) -> dict[str, torch.Tensor]:
    flat: dict[str, torch.Tensor] = {}
    for name, value in head_weights.items():
        if isinstance(value, torch.Tensor):
            flat[name] = value.detach().contiguous().cpu()
        else:
            for key, tensor in value.items():
                flat[f"{name}.{key}"] = tensor.detach().contiguous().cpu()
    return flat


def _unflatten_head_weights(flat: dict[str, torch.Tensor]) -> dict[str, Any]:
    nested: dict[str, Any] = {}
    for key, tensor in flat.items():
        if "." in key:
            name, sub = key.split(".", 1)
            nested.setdefault(name, {})[sub] = tensor
        else:
            nested[key] = tensor
    return nested


def save_head_safetensors(head_weights: dict[str, Any], path: Path | str) -> None:
    """Write pointer-head weights (nested state dicts) as a flat safetensors file."""
    from safetensors.torch import save_file

    save_file(_flatten_head_weights(head_weights), str(path))


def load_head_safetensors(path: Path | str, device: str = "cpu") -> dict[str, Any]:
    """Read a head.safetensors file back into the nested layout used by head.pt."""
    from safetensors.torch import load_file

    return _unflatten_head_weights(load_file(str(path), device=device))


def convert_head_pt_to_safetensors(checkpoint_dir: Path | str) -> Path:
    """Convert an existing checkpoint's head.pt to head.safetensors (head.pt is kept)."""
    path = Path(checkpoint_dir)
    weights = torch.load(path / "head.pt", map_location="cpu")
    out = path / "head.safetensors"
    save_head_safetensors(weights, out)
    return out


def save_soro_kev_checkpoint(
    model: SoroKevModel,
    out_dir: Path,
    calibrated_temp: float | None = None,
    q_hat: float | None = None,
) -> None:
    """Save pointer head weights, config, and calibrated temperature."""
    out_dir.mkdir(parents=True, exist_ok=True)
    head_weights = {
        "q_proj_head": model.q_proj_head.state_dict(),
        "opt_proj_head": model.opt_proj_head.state_dict(),
        "binary_head": model.binary_head.state_dict(),
        "intent_head": model.intent_head.state_dict(),
        "log_temp": model.log_temp.data,
    }
    torch.save(head_weights, out_dir / "head.pt")
    # Also write a pickle-free copy. Loading prefers head.safetensors when present.
    try:
        save_head_safetensors(head_weights, out_dir / "head.safetensors")
    except ImportError:
        pass

    # If real PEFT LoRA backbone is present, save adapter
    if hasattr(model.backbone, "save_pretrained") and not isinstance(model.backbone, MockCausalBackbone):
        try:
            model.backbone.save_pretrained(out_dir)
        except Exception:
            pass

    config = {
        "model_name": model.model_name,
        "proj_dim": model.proj_dim,
        "hidden_size": model.hidden_size,
        "use_mock": model.use_mock or isinstance(model.backbone, MockCausalBackbone),
        "learnable_temperature": float(torch.exp(model.log_temp).item()),
        "calibrated_q_hat": q_hat,
    }
    with (out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    temp_val = calibrated_temp if calibrated_temp is not None else float(torch.exp(model.log_temp).item())
    with (out_dir / "temperature.json").open("w", encoding="utf-8") as f:
        json.dump({"temperature": temp_val, "q_hat": q_hat}, f, indent=2)


def load_soro_kev_checkpoint(checkpoint_dir: Path | str, device: str = "cpu") -> tuple[SoroKevModel, float]:
    """Load SoroKevModel directly from checkpoint directory without external kev package."""
    path = Path(checkpoint_dir)
    head_file = path / "head.pt"
    head_st_file = path / "head.safetensors"
    config_file = path / "config.json"
    temp_file = path / "temperature.json"

    if not head_file.exists() and not head_st_file.exists():
        raise FileNotFoundError(f"Missing head.pt / head.safetensors in {path}")

    cfg: dict[str, Any] = {}
    if config_file.exists():
        with config_file.open(encoding="utf-8") as f:
            cfg = json.load(f)

    model = SoroKevModel(
        model_name=cfg.get("model_name", "Qwen/Qwen2.5-0.5B"),
        hidden_size=cfg.get("hidden_size", 1024),
        proj_dim=cfg.get("proj_dim", 256),
        use_mock=cfg.get("use_mock", True),
        allow_mock_fallback=True,
    )
    adapter = path / "adapter_model.safetensors"
    if adapter.exists() and not isinstance(model.backbone, MockCausalBackbone):
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        set_peft_model_state_dict(model.backbone, load_file(str(adapter)))

    if head_st_file.exists():
        weights = load_head_safetensors(head_st_file, device=device)
    else:
        weights = torch.load(head_file, map_location=device)
    model.q_proj_head.load_state_dict(weights["q_proj_head"])
    model.opt_proj_head.load_state_dict(weights["opt_proj_head"])
    if "binary_head" in weights:
        model.binary_head.load_state_dict(weights["binary_head"])
    if "intent_head" in weights and hasattr(model, "intent_head"):
        model.intent_head.load_state_dict(weights["intent_head"])
    if "log_temp" in weights:
        model.log_temp.data = weights["log_temp"]

    # A training config may record the 512-token T4 cap. Inference keeps 4096.
    model.inference_max_seq_len = max(int(cfg.get("max_seq_len", 4096) or 4096), 4096)

    temperature = float(torch.exp(model.log_temp).item())
    q_hat = None
    if temp_file.exists():
        try:
            with temp_file.open(encoding="utf-8") as f:
                t_data = json.load(f)
                temperature = float(t_data.get("temperature", temperature))
                q_hat = t_data.get("q_hat")
        except Exception:
            pass

    if q_hat is None and config_file.exists():
        try:
            with config_file.open(encoding="utf-8") as f:
                c_data = json.load(f)
                q_hat = c_data.get("calibrated_q_hat") or c_data.get("calibration", {}).get("q_hat")
        except Exception:
            pass

    model.q_hat = float(q_hat) if q_hat is not None else None

    quantized = "4b" in str(getattr(model, "model_name", "")).lower()
    if quantized and torch.cuda.is_available() and str(device).startswith("cuda"):
        for head_name in ("q_proj_head", "opt_proj_head", "binary_head", "intent_head"):
            head = getattr(model, head_name, None)
            if head is not None:
                head.to(device)
        model.log_temp.data = model.log_temp.data.to(device)
    else:
        model.to(device)
    model.eval()
    return model, temperature


def _from_checkpoint(cls, checkpoint_dir: Path | str, device: str = "cpu"):
    model, _temperature = load_soro_kev_checkpoint(checkpoint_dir, device=device)
    return model


SoroKevModel.from_checkpoint = classmethod(_from_checkpoint)
