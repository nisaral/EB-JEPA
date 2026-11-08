"""Universal Probabilistic Pointer (UPP): the next Soro-Kev decision architecture.

One causal backbone reads a menu of options and an input; a pointer head scores
every option in a single prefill. UPP extends the Kev/Jev pointer with switchable
parts so each can be ablated alone (see docs/OMNI_PLAN.md):

Layout (``layout``)
  * ``query_first``   Kev: request, then options (options see the request).
  * ``catalog_first`` options, then request; the catalog KV cache is reused.
  * ``block``         each option is an isolated block with its own positions
                      (Block-Attention). Exact order invariance, per-catalog cache.
                      Needs full-attention backbones (not Gated DeltaNet).
Head (``head``)
  * ``dot``   Kev scaled dot product.
  * ``gauss`` diagonal-Gaussian match with a KL to N(0, I) (VAE-style).
  * ``rbf``   distance-aware pointer: spectral-normalized projections and an
              RBF score, SNGP-style. Far-from-everything inputs get low scores.
Extras
  * ``refiner``     2-layer order-equivariant set transformer over [query; options].
  * ``ensemble=K``  K independent head members; probabilities are averaged and
                    member disagreement is an uncertainty signal.
  * ``exits``       extra pointer readouts at fractional depths (early exit).
  * ``think_slots`` learned pause embeddings before the readout token.

Every head also reports an energy score (logsumexp of logits) and a scalar
uncertainty ``u`` used for normalized conformal sets.

Defaults (query_first + dot) reproduce the Soro-Kev v0.2 pointer.
Image and audio enter as embeddings spliced into the request span once a
multimodal backbone (Gemma 4 E2B) is plugged in.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from soro.routing.kev_model import MockCausalBackbone, SoroKevModel

ROUTE_INSTRUCTIONS = "Which single tool or skill should run for this request? Use only the listed candidates."


@dataclass
class UPPConfig:
    layout: str = "query_first"  # query_first | catalog_first | block
    head: str = "dot"  # dot | gauss | rbf
    refiner: bool = False
    ensemble: int = 1
    exits: tuple[float, ...] = field(default_factory=tuple)  # e.g. (0.5, 0.75)
    think_slots: int = 0
    proj_dim: int = 256
    # LayerNorm on backbone states before the head. Some backbones (Qwen2.5) emit
    # very large hidden values; without it a T=0.07 dot head starts saturated.
    pre_norm: bool = True
    kl_weight: float = 1e-3
    brier_weight: float = 0.25
    exit_weight: float = 0.5
    max_option_chars: int = 400

    def name(self) -> str:
        parts = [self.layout, self.head]
        if self.refiner:
            parts.append("refiner")
        if self.ensemble > 1:
            parts.append(f"ens{self.ensemble}")
        if self.exits:
            parts.append("exits" + "-".join(f"{e:g}" for e in self.exits))
        if self.think_slots:
            parts.append(f"slots{self.think_slots}")
        return "+".join(parts)


class _Member(nn.Module):
    """One pointer: projections, optional refiner, and a score function."""

    def __init__(self, d_model: int, cfg: UPPConfig):
        super().__init__()
        self.cfg = cfg
        p = cfg.proj_dim
        q_in, k_in = nn.Linear(d_model, p), nn.Linear(d_model, p)
        if cfg.head == "rbf":
            sn = torch.nn.utils.parametrizations.spectral_norm
            q_in, k_in = sn(q_in), sn(k_in)
        self.q_in, self.k_in = q_in, k_in
        self.refiner = None
        if cfg.refiner:
            self.type_emb = nn.Parameter(torch.zeros(2, p))
            layer = nn.TransformerEncoderLayer(
                d_model=p, nhead=4, dim_feedforward=2 * p, dropout=0.0, batch_first=True, norm_first=True
            )
            self.refiner = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        if cfg.head == "gauss":
            self.q_logvar, self.k_logvar = nn.Linear(p, p), nn.Linear(p, p)
            for lin in (self.q_logvar, self.k_logvar):
                nn.init.zeros_(lin.weight)
                nn.init.zeros_(lin.bias)
            self.log_temp = nn.Parameter(torch.tensor(0.0))
        elif cfg.head == "rbf":
            self.log_temp = nn.Parameter(torch.tensor(0.0))  # log length-scale^2
        else:
            self.log_temp = nn.Parameter(torch.tensor(math.log(0.07)))

    def forward(self, hq: torch.Tensor, hk: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        q, k = self.q_in(hq.unsqueeze(0)), self.k_in(hk)
        if self.refiner is not None:
            tokens = torch.cat([q + self.type_emb[0], k + self.type_emb[1]], dim=0).unsqueeze(0)
            out = self.refiner(tokens).squeeze(0)
            q, k = out[:1], out[1:]
        p = self.cfg.proj_dim
        temp = torch.exp(self.log_temp) + 1e-4
        aux: dict[str, torch.Tensor] = {}
        if self.cfg.head == "gauss":
            lv_q, lv_k = self.q_logvar(q).clamp(-6, 6), self.k_logvar(k).clamp(-6, 6)
            var = torch.exp(lv_q) + torch.exp(lv_k)
            score = -0.5 * (((q - k) ** 2) / var + torch.log(var)).sum(-1)
            logits = score / math.sqrt(p) / temp
            aux["kl"] = 0.5 * (q ** 2 + torch.exp(lv_q) - 1 - lv_q).mean() + 0.5 * (k ** 2 + torch.exp(lv_k) - 1 - lv_k).mean()
            aux["u"] = torch.exp(lv_q).mean()
        elif self.cfg.head == "rbf":
            d2 = ((q - k) ** 2).sum(-1) / p
            logits = -0.5 * d2 / temp * math.sqrt(p)
            aux["u"] = d2.min()  # distance to the nearest option
        else:
            logits = (k @ q.squeeze(0)) / math.sqrt(p) / temp
        return logits, aux


class UPPHead(nn.Module):
    """Pointer head with ``ensemble`` members. Returns log of mean member probabilities."""

    def __init__(self, d_model: int, cfg: UPPConfig):
        super().__init__()
        self.cfg = cfg
        self.norm = nn.LayerNorm(d_model) if cfg.pre_norm else nn.Identity()
        self.members = nn.ModuleList(_Member(d_model, cfg) for _ in range(max(1, cfg.ensemble)))

    def forward(self, h_q: torch.Tensor, h_opts: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        hq, hk = self.norm(h_q.float()), self.norm(h_opts.float())
        outs = [m(hq, hk) for m in self.members]
        aux: dict[str, torch.Tensor] = {}
        if len(outs) == 1:
            logits, aux = outs[0]
        else:
            probs = torch.stack([F.softmax(lg, -1) for lg, _ in outs])  # (M, K)
            mean = probs.mean(0)
            logits = torch.log(mean.clamp_min(1e-9))
            aux["u"] = probs.var(0).sum() + 1e-6  # member disagreement
            kls = [a["kl"] for _, a in outs if "kl" in a]
            if kls:
                aux["kl"] = torch.stack(kls).mean()
            aux["member_logits"] = torch.stack([lg for lg, _ in outs])
        raw = outs[0][0] if len(outs) == 1 else aux["member_logits"].mean(0)
        aux["energy"] = torch.logsumexp(raw, -1)
        if "u" not in aux:
            aux["u"] = 1.0 - F.softmax(logits, -1).max()  # fallback: 1 - top prob
        return logits, aux


class UPPModel(nn.Module):
    """Backbone (LoRA) + UPP head. Reuses Soro-Kev's tested backbone loader."""

    def __init__(
        self,
        cfg: UPPConfig,
        model_name: str = "Qwen/Qwen2.5-0.5B",
        use_mock: bool = False,
        load_in_4bit: bool = False,
        torch_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        base = SoroKevModel(
            model_name=model_name,
            use_mock=use_mock,
            allow_mock_fallback=use_mock,
            load_in_4bit=load_in_4bit,
            torch_dtype=torch_dtype,
        )
        self.model_name = model_name
        self.backbone = base.backbone
        self.tokenizer = base.tokenizer
        self._mock_tok = base._tokenize_text
        d_model = base.q_proj_head.in_features
        del base
        self.head = UPPHead(d_model, cfg)
        self.slots = None
        if cfg.think_slots > 0:
            emb = self._embed_layer().weight
            init = emb.detach().float().mean(0, keepdim=True).repeat(cfg.think_slots, 1)
            self.slots = nn.Parameter(init + 0.02 * torch.randn_like(init))
        if cfg.layout == "block" and not self.is_mock and self._has_linear_attention():
            raise ValueError(
                f"layout='block' needs full-attention layers; {model_name} has linear-attention "
                "(Gated DeltaNet) layers that cannot isolate blocks. Use Qwen3 or Gemma 4."
            )

    # ------------------------------------------------------------------ plumbing
    @property
    def is_mock(self) -> bool:
        return isinstance(self.backbone, MockCausalBackbone)

    def _has_linear_attention(self) -> bool:
        cfg = getattr(self.backbone, "config", None)
        types = getattr(cfg, "layer_types", None) or getattr(getattr(cfg, "text_config", None), "layer_types", None)
        return bool(types) and any("linear" in str(t) for t in types)

    def _embed_layer(self) -> nn.Embedding:
        if self.is_mock:
            return self.backbone.embedding
        return self.backbone.get_input_embeddings()

    def _tok(self, text: str, special: bool = False) -> list[int]:
        if self.tokenizer is not None:
            return self.tokenizer.encode(text, add_special_tokens=special)
        return self._mock_tok(text)

    def _exit_layers(self) -> list[int]:
        if not self.cfg.exits or self.is_mock:
            return []
        n = int(getattr(getattr(self.backbone, "config", None), "num_hidden_layers", 0) or 0)
        return sorted({max(1, min(n - 1, round(f * n))) for f in self.cfg.exits}) if n else []

    def _hidden(self, embeds, mask, positions=None, past=None, use_cache=False):
        """Returns (last hidden, list of exit hiddens, past)."""
        if self.is_mock:
            b = self.backbone
            L = embeds.shape[1]
            pos = positions if positions is not None else torch.arange(L, device=embeds.device).unsqueeze(0)
            x = embeds + b.pos_emb(pos.clamp(max=b.pos_emb.num_embeddings - 1))
            if mask.dim() == 4:  # additive (B,1,L,L) -> (B*heads, L, L)
                heads = b.transformer.layers[0].self_attn.num_heads
                attn = mask[:, 0].repeat_interleave(heads, dim=0).to(x.dtype)
                h = b.transformer(x, mask=attn)
            else:
                causal = nn.Transformer.generate_square_subsequent_mask(L, device=embeds.device).bool()
                h = b.transformer(x, mask=causal, src_key_padding_mask=(mask == 0), is_causal=True)
            return b.proj(h), [], None
        exits = self._exit_layers()
        out = self.backbone(
            inputs_embeds=embeds, attention_mask=mask, position_ids=positions, past_key_values=past,
            use_cache=use_cache, output_hidden_states=bool(exits),
        )
        h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        hs = [out.hidden_states[i] for i in exits] if exits else []
        return h, hs, getattr(out, "past_key_values", None)

    # ------------------------------------------------------------------ layout
    def build(self, state: str, criteria: dict[str, str]) -> dict[str, Any]:
        """Token ids (slot marker -1), positions, block ids, query index and option span ends."""
        cfg = self.cfg
        keys = list(criteria)
        opt_text = [f"[{i}] {k}: {str(criteria[k])[: cfg.max_option_chars]}\n" for i, k in enumerate(keys)]
        question = f"Question [tool_route]: {ROUTE_INSTRUCTIONS}\n"
        request = self._tok(f"State: {state}\n") + self._tok(question) + [-1] * cfg.think_slots + self._tok("Answer:")
        ids: list[int] = []
        opt_ends: list[int] = []
        blocks: list[int] = []  # -1 prefix, i option i, -2 request
        positions: list[int] = []
        if cfg.layout == "block":
            prefix = self._tok("Options:\n", special=True)
            ids += prefix
            blocks += [-1] * len(prefix)
            positions += list(range(len(prefix)))
            p0 = len(prefix)
            longest = 0
            for i, t in enumerate(opt_text):
                toks = self._tok(t.replace(f"[{i}] ", "", 1))  # no index: order-free text
                ids += toks
                blocks += [i] * len(toks)
                positions += list(range(p0, p0 + len(toks)))
                opt_ends.append(len(ids) - 1)
                longest = max(longest, len(toks))
            catalog_len = len(ids)
            r0 = p0 + longest
            ids += request
            blocks += [-2] * len(request)
            positions += list(range(r0, r0 + len(request)))
        elif cfg.layout == "catalog_first":
            ids += self._tok("Options:\n", special=True)
            for t in opt_text:
                ids += self._tok(t)
                opt_ends.append(len(ids) - 1)
            catalog_len = len(ids)
            ids += request
            positions = list(range(len(ids)))
        else:
            ids += self._tok(f"State: {state}\n", special=True) + self._tok(question)
            ids += [-1] * cfg.think_slots
            ids += self._tok("Options:\n")
            q_marker = len(ids) - 1
            for t in opt_text:
                ids += self._tok(t)
                opt_ends.append(len(ids) - 1)
            catalog_len = 0
            positions = list(range(len(ids)))
            return {"ids": ids, "q_idx": q_marker, "opt_ends": opt_ends, "keys": keys,
                    "catalog_len": catalog_len, "positions": positions, "blocks": None}
        return {"ids": ids, "q_idx": len(ids) - 1, "opt_ends": opt_ends, "keys": keys,
                "catalog_len": catalog_len, "positions": positions, "blocks": blocks or None}

    @staticmethod
    def _block_allowed(blocks: list[int], L: int) -> torch.Tensor:
        """(L, L) bool: token i may attend token j. Options see the prefix and their own block."""
        n = len(blocks)
        b = torch.tensor(blocks)
        i_b, j_b = b.view(n, 1), b.view(1, n)
        causal = torch.ones(n, n, dtype=torch.bool).tril()
        same = i_b == j_b
        to_prefix = j_b == -1
        from_request = i_b == -2
        allowed = causal & (same | to_prefix | from_request)
        out = torch.zeros(L, L, dtype=torch.bool)
        out[:n, :n] = allowed
        idx = torch.arange(L)
        out[idx, idx] = True  # padding rows attend to themselves (no NaNs)
        return out

    def _embed(self, ids: list[int], device: torch.device) -> torch.Tensor:
        emb = self._embed_layer()
        x = emb(torch.tensor([max(i, 0) for i in ids], dtype=torch.long, device=device))
        if self.slots is not None:
            slot_pos = [j for j, i in enumerate(ids) if i == -1]
            if slot_pos:
                x = x.clone()
                x[slot_pos] = self.slots.to(x.dtype)
        return x

    def _batch(self, packs: list[dict[str, Any]], device: torch.device):
        L = max(len(p["ids"]) for p in packs)
        emb_dim = self._embed_layer().weight.shape[1]
        dtype = self._embed_layer().weight.dtype
        embeds = torch.zeros(len(packs), L, emb_dim, device=device, dtype=dtype)
        positions = torch.zeros(len(packs), L, dtype=torch.long, device=device)
        block = self.cfg.layout == "block"
        if block:
            neg = torch.finfo(dtype if dtype.is_floating_point else torch.float32).min
            mask = torch.full((len(packs), 1, L, L), neg, device=device, dtype=dtype)
        else:
            mask = torch.zeros(len(packs), L, dtype=torch.long, device=device)
        for b, pk in enumerate(packs):
            n = len(pk["ids"])
            embeds[b, :n] = self._embed(pk["ids"], device)
            positions[b, :n] = torch.tensor(pk["positions"], device=device)
            if n < L:
                positions[b, n:] = positions[b, n - 1] + 1
            if block:
                mask[b, 0][self._block_allowed(pk["blocks"], L).to(device)] = 0.0
            else:
                mask[b, :n] = 1
        return embeds, mask, positions

    # ------------------------------------------------------------------ forward
    def forward_batch(self, records: list[dict[str, Any]], max_seq_len: int = 1024) -> dict[str, Any]:
        device = next(self.head.parameters()).device
        packs, golds = [], []
        for r in records:
            q = r["questions"]["tool_route"]
            pk = self.build(r["state"], q["criteria"])
            if len(pk["ids"]) > max_seq_len:
                continue
            packs.append(pk)
            golds.append(pk["keys"].index(q["label"]) if q["label"] in pk["keys"] else 0)
        if not packs:
            zero = sum(p.sum() * 0.0 for p in self.parameters() if p.requires_grad)
            return {"loss": zero, "skipped": True, "logits": [], "kept": 0}
        embeds, mask, positions = self._batch(packs, device)
        h, exit_hs, _ = self._hidden(embeds, mask, positions)
        losses, logits_out, us = [], [], []
        n_exit = len(exit_hs)
        for b, pk in enumerate(packs):
            g = golds[b]
            logits, aux = self.head(h[b, pk["q_idx"]], h[b, pk["opt_ends"]])
            losses.append(self._row_loss(logits, aux, g))
            for hs in exit_hs:
                lg_e, aux_e = self.head(hs[b, pk["q_idx"]], hs[b, pk["opt_ends"]])
                losses.append(self.cfg.exit_weight / n_exit * self._row_loss(lg_e, aux_e, g))
            logits_out.append(logits)
            us.append(float(aux["u"].detach()))
        loss = torch.stack(losses).sum() / len(packs)
        acc = sum(int(torch.argmax(lg).item() == g) for lg, g in zip(logits_out, golds)) / len(golds)
        return {"loss": loss, "skipped": False, "logits": logits_out, "kept": len(packs), "acc": acc,
                "u": sum(us) / len(us)}

    def _row_loss(self, logits: torch.Tensor, aux: dict[str, torch.Tensor], g: int) -> torch.Tensor:
        p = F.softmax(logits, -1)
        oh = torch.zeros_like(p)
        oh[g] = 1.0
        loss = -torch.log(p[g].clamp_min(1e-9)) + self.cfg.brier_weight * ((p - oh) ** 2).mean()
        if "member_logits" in aux:  # train each ensemble member on its own CE too
            loss = loss + F.cross_entropy(aux["member_logits"], torch.full((aux["member_logits"].shape[0],), g,
                                          device=logits.device))
        if "kl" in aux:
            loss = loss + self.cfg.kl_weight * aux["kl"]
        return loss

    @torch.no_grad()
    def score(self, state: str, criteria: dict[str, str]) -> tuple[dict[str, float], dict[str, Any]]:
        """Probabilities over ``criteria`` for one request, plus uncertainty and exit info."""
        device = next(self.head.parameters()).device
        pk = self.build(state, criteria)
        embeds, mask, positions = self._batch([pk], device)
        h, exit_hs, _ = self._hidden(embeds, mask, positions)
        logits, aux = self.head(h[0, pk["q_idx"]], h[0, pk["opt_ends"]])
        p = F.softmax(logits, -1).float().cpu().tolist()
        info: dict[str, Any] = {"u": float(aux["u"]), "energy": float(aux["energy"])}
        if exit_hs:
            info["exits"] = []
            for hs in exit_hs:
                lg, _ = self.head(hs[0, pk["q_idx"]], hs[0, pk["opt_ends"]])
                pe = F.softmax(lg, -1).float().cpu().tolist()
                info["exits"].append({k: pe[i] for i, k in enumerate(pk["keys"])})
        return {k: p[i] for i, k in enumerate(pk["keys"])}, info

    # ------------------------------------------------------------------ catalog cache
    @torch.no_grad()
    def encode_catalog(self, criteria: dict[str, str]) -> dict[str, Any]:
        """Run the options once (catalog_first or block). Reuse for many requests."""
        if self.cfg.layout not in ("catalog_first", "block") or self.is_mock:
            raise ValueError("catalog caching needs layout catalog_first/block and a real backbone")
        device = next(self.head.parameters()).device
        pk = self.build("", criteria)
        n = pk["catalog_len"]
        cat = dict(pk, ids=pk["ids"][:n], positions=pk["positions"][:n],
                   blocks=pk["blocks"][:n] if pk["blocks"] else None)
        embeds, mask, positions = self._batch([cat], device)
        h, _, past = self._hidden(embeds, mask, positions, use_cache=True)
        return {"criteria": dict(criteria), "past": past, "h_opts": h[0, pk["opt_ends"]].clone(),
                "keys": pk["keys"], "catalog_len": n}

    @torch.no_grad()
    def score_cached(self, state: str, cache: dict[str, Any]) -> tuple[dict[str, float], dict[str, Any]]:
        device = next(self.head.parameters()).device
        pk = self.build(state, cache["criteria"])
        n = cache["catalog_len"]
        x = self._embed(pk["ids"][n:], device).unsqueeze(0)
        positions = torch.tensor([pk["positions"][n:]], device=device)
        mask = torch.ones(1, len(pk["ids"]), dtype=torch.long, device=device)  # request sees everything
        h, _, _ = self._hidden(x, mask, positions, past=copy.deepcopy(cache["past"]), use_cache=True)
        logits, aux = self.head(h[0, -1], cache["h_opts"])
        p = F.softmax(logits, -1).float().cpu().tolist()
        return {k: p[i] for i, k in enumerate(cache["keys"])}, {"u": float(aux["u"])}

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def config_dict(self) -> dict[str, Any]:
        return {"model_name": self.model_name, **asdict(self.cfg)}
