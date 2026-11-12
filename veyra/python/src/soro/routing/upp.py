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
Option compression and interaction (block layout)
  * ``gists=g``       each option is compressed into g learned gist tokens inside its
                      own block; the request attends to the gists only, so a tool's
                      cache is g tokens and large catalogs fit in one pass. Per-tool
                      caches are exact and composable in any order.
  * ``gist_distill``  weight of a self-distillation loss: the gist path is trained to
                      match the same model reading the full option text.
  * ``interaction``   ``cross`` (request attends to options) or ``late`` (no
                      cross-attention; ColBERT-style MaxSim between request tokens and
                      option gists; a tool's cache is g vectors).
  * ``option_pool``   option readout: ``last`` token, ``mean`` or learned ``attn`` pooling.
  * ``gist_budgets``  Matryoshka gists: train with a random budget b per step so the first
                      b gists of every option are a usable summary on their own. Gist j only
                      attends to gists < j, so prefixes are nested by construction. One cache
                      then serves a coarse pass over the whole catalog (1 gist per tool) and a
                      fine pass over the top-M tools (all gists): retrieval and listwise
                      reranking in one model (``score_two_level``).
Request modality (``media=True``, omni backbones such as Gemma 4)
  * The catalog stays text and is cached once; the request may carry speech or an image,
    entered as the backbone's own soft tokens (``soro.routing.media``) inside the request
    block. All block-layout exactness properties hold for these requests too.

Every head also reports an energy score (logsumexp of logits) and a scalar
uncertainty ``u`` used for normalized conformal sets.

Defaults (query_first + dot) reproduce the Soro-Kev v0.2 pointer.
Image and audio enter as embeddings spliced into the request span once a
multimodal backbone (Gemma 4 E2B) is plugged in.
"""

from __future__ import annotations

import contextlib
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
    refiner_layers: int = 1
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
    gists: int = 0
    gist_distill: float = 0.0
    interaction: str = "cross"  # cross | late
    option_pool: str = "last"  # last | mean | attn
    # Block layout: the request starts at this fixed position, so a tool's cache never
    # depends on which other tools share the catalog (options are truncated to fit).
    request_offset: int = 640
    gist_budgets: tuple[int, ...] = field(default_factory=tuple)  # e.g. (1, 2, 4, 8)
    # Readout. ``vec``: last request token vs one vector per option (needs the request token to
    # carry the answer). ``xattn``: each option vector cross-attends to all request token states
    # in a small post-backbone head (options never see each other, so scores stay exactly order
    # equivariant and the catalog cache is unchanged).
    readout: str = "vec"  # vec | xattn
    option_index: bool = True  # prepend "[i] " to options in non-block layouts
    option_isolation: bool = False  # in query_first: mask option-to-option attn & share start position
    cost_weight: float = 0.0  # S2 exact expected asymmetric action cost weight (0 = off)
    teacher_kl_weight: float = 0.0  # Soft-label KL weight against row["teacher"]["probs"] (0 = off)

    def name(self) -> str:
        parts = [self.layout, self.head]
        if self.refiner:
            parts.append("refiner" if self.refiner_layers == 1 else f"refiner{self.refiner_layers}")
        if self.ensemble > 1:
            parts.append(f"ens{self.ensemble}")
        if self.exits:
            parts.append("exits" + "-".join(f"{e:g}" for e in self.exits))
        if self.think_slots:
            parts.append(f"slots{self.think_slots}")
        if self.gists:
            parts.append(f"gist{self.gists}")
        if self.gist_distill:
            parts.append("gsd")
        if self.gist_budgets:
            parts.append("mrl")
        if self.interaction != "cross":
            parts.append(self.interaction)
        if self.option_pool != "last":
            parts.append(f"pool-{self.option_pool}")
        if self.readout != "vec":
            parts.append(self.readout)
        if not self.option_index:
            parts.append("noidx")
        if self.option_isolation:
            parts.append("iso")
        if self.cost_weight > 0:
            parts.append(f"cost{self.cost_weight:g}")
        if self.teacher_kl_weight > 0:
            parts.append(f"tkl{self.teacher_kl_weight:g}")
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
            self.refiner = nn.TransformerEncoder(
                layer, num_layers=max(1, cfg.refiner_layers), enable_nested_tensor=False
            )
            self.ref_norm = nn.LayerNorm(p)
            self.ref_gate = nn.Parameter(torch.tensor(0.1))
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
            ctx = torch.amp.autocast("cuda", enabled=False) if q.is_cuda else contextlib.nullcontext()
            with ctx:
                tokens = torch.cat([q.float() + self.type_emb[0], k.float() + self.type_emb[1]], dim=0).unsqueeze(0)
                delta = self.ref_norm(self.refiner(tokens).squeeze(0) - tokens.squeeze(0))
                gate = torch.tanh(self.ref_gate)
                q, k = q.float() + gate * delta[:1], k.float() + gate * delta[1:]
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


class AttnPool(nn.Module):
    """Learned-query attention pooling over a span of token states (NV-Embed-style)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(d_model))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h: torch.Tensor) -> torch.Tensor:  # (n, d) -> (d,)
        hn = self.norm(h.float())
        w = torch.softmax(hn @ self.query / math.sqrt(hn.shape[-1]), dim=0)
        return (w.unsqueeze(-1) * h.float()).sum(0)


class XAttnHead(nn.Module):
    """Options read the request: per-option cross-attention over request token states.

    Each option is scored independently of the others (no option-option attention), so
    permuting the menu permutes the logits exactly. Logits are a linear readout, no temperature."""

    def __init__(self, d_model: int, cfg: UPPConfig, layers: int = 2):
        super().__init__()
        self.cfg = cfg
        p = cfg.proj_dim
        self.norm = nn.LayerNorm(d_model)
        self.q_in, self.k_in = nn.Linear(d_model, p), nn.Linear(d_model, p)
        self.attn = nn.ModuleList(nn.MultiheadAttention(p, 4, batch_first=True) for _ in range(layers))
        self.ln = nn.ModuleList(nn.LayerNorm(p) for _ in range(2 * layers))
        self.ff = nn.ModuleList(nn.Sequential(nn.Linear(p, 2 * p), nn.GELU(), nn.Linear(2 * p, p))
                                for _ in range(layers))
        self.out = nn.Linear(p, 1)

    def forward(self, h_req: torch.Tensor, h_opts: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """h_req: (Lq, d) request token states; h_opts: (K, d) -> logits (K,)."""
        r = self.q_in(self.norm(h_req.float())).unsqueeze(0)  # (1, Lq, p)
        o = self.k_in(self.norm(h_opts.float())).unsqueeze(1)  # (K, 1, p): one query per option
        mem = r.expand(o.shape[0], -1, -1)
        for i, attn in enumerate(self.attn):
            x = self.ln[2 * i](o)
            o = o + attn(x, mem, mem, need_weights=False)[0]
            o = o + self.ff[i](self.ln[2 * i + 1](o))
        logits = self.out(o.squeeze(1)).squeeze(-1)
        return logits, {"energy": torch.logsumexp(logits, -1), "u": 1.0 - F.softmax(logits, -1).max()}


class LateHead(nn.Module):
    """ColBERT-style late interaction: mean over request tokens of max similarity to option gists."""

    def __init__(self, d_model: int, cfg: UPPConfig):
        super().__init__()
        self.cfg = cfg
        self.norm = nn.LayerNorm(d_model) if cfg.pre_norm else nn.Identity()
        self.q_in = nn.Linear(d_model, cfg.proj_dim)
        self.k_in = nn.Linear(d_model, cfg.proj_dim)
        self.log_temp = nn.Parameter(torch.tensor(math.log(0.05)))

    def forward(self, h_req: torch.Tensor, h_gists: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """h_req: (Lq, d) request token states; h_gists: (K, g, d) -> logits (K,)."""
        q = F.normalize(self.q_in(self.norm(h_req.float())), dim=-1)
        k = F.normalize(self.k_in(self.norm(h_gists.float())), dim=-1)
        sim = torch.einsum("qp,kgp->kqg", q, k)  # (K, Lq, g)
        score = sim.max(-1).values.mean(-1)
        logits = score / (torch.exp(self.log_temp) + 1e-4)
        return logits, {"energy": torch.logsumexp(logits, -1), "u": 1.0 - F.softmax(logits, -1).max()}


class UPPModel(nn.Module):
    """Backbone (LoRA) + UPP head. Reuses Soro-Kev's tested backbone loader."""

    GIST_BASE = -100  # token-id markers: -1 think slot, -100 - j gist j

    def __init__(
        self,
        cfg: UPPConfig,
        model_name: str = "Qwen/Qwen2.5-0.5B",
        use_mock: bool = False,
        load_in_4bit: bool = False,
        torch_dtype: torch.dtype | None = None,
        media: bool = False,
    ):
        super().__init__()
        if cfg.interaction == "late" and cfg.layout != "block":
            raise ValueError("interaction='late' needs layout='block'")
        if cfg.gists and cfg.layout != "block":
            raise ValueError("gists need layout='block'")
        self.cfg = cfg
        base = SoroKevModel(
            model_name=model_name,
            use_mock=use_mock,
            allow_mock_fallback=use_mock,
            load_in_4bit=load_in_4bit,
            torch_dtype=torch_dtype,
            keep_media=media and not use_mock,
        )
        self.model_name = model_name
        self.media_encoder = None
        if media and not use_mock:
            if base.media_model is None:
                raise ValueError(f"media=True needs an omni backbone with audio/vision towers; {model_name} has none")
            from soro.routing.media import MediaEncoder

            self.media_encoder = MediaEncoder(base.media_model, model_name)
        self._media_device = None
        self.backbone = base.backbone
        self.tokenizer = base.tokenizer
        self._mock_tok = base._tokenize_text
        d_model = base.q_proj_head.in_features
        del base
        if cfg.interaction == "late":
            self.head = LateHead(d_model, cfg)
        elif cfg.readout == "xattn":
            self.head = XAttnHead(d_model, cfg)
        else:
            self.head = UPPHead(d_model, cfg)
        self.pool = AttnPool(d_model) if cfg.option_pool == "attn" else None
        emb = self._embed_layer().weight
        # learned tokens start at the embedding layer's real scale (Gemma multiplies by sqrt(d))
        scale = float(getattr(self._embed_layer(), "embed_scale", 1.0) or 1.0)
        mean_emb = emb.detach().float().mean(0, keepdim=True) * scale
        self.slots = None
        if cfg.think_slots > 0:
            self.slots = nn.Parameter(mean_emb.repeat(cfg.think_slots, 1)
                                      + 0.02 * torch.randn(cfg.think_slots, emb.shape[1]))
        self.gist_emb = None
        if cfg.gists > 0:
            self.gist_emb = nn.Parameter(mean_emb.repeat(cfg.gists, 1) + 0.02 * torch.randn(cfg.gists, emb.shape[1]))
        if cfg.option_isolation and cfg.layout == "catalog_first":
            raise ValueError("option_isolation is for layout='query_first' (layout='block' already isolates options)")
        if (cfg.layout == "block" or cfg.option_isolation) and not self.is_mock and self._has_linear_attention():
            raise ValueError(
                f"layout='block' / option_isolation needs full-attention layers; {model_name} has linear-attention "
                "(Gated DeltaNet) layers that cannot isolate blocks. Use Qwen3 or Gemma 4."
            )

    # ------------------------------------------------------------------ plumbing
    @property
    def is_mock(self) -> bool:
        return isinstance(self.backbone, MockCausalBackbone)

    @property
    def _device(self) -> torch.device:
        return next(self.head.parameters()).device

    def _has_linear_attention(self) -> bool:
        cfg = getattr(self.backbone, "config", None)
        types = getattr(cfg, "layer_types", None) or getattr(getattr(cfg, "text_config", None), "layer_types", None)
        return bool(types) and any("linear" in str(t) for t in types)

    def _embed_layer(self) -> nn.Embedding:
        if self.is_mock:
            return self.backbone.embedding
        return self.backbone.get_input_embeddings()

    MEDIA = -3  # request positions filled with media soft tokens

    def _pad_id(self) -> int:
        tok = self.tokenizer
        pid = getattr(tok, "pad_token_id", None) if tok is not None else None
        return int(pid) if pid is not None else 0

    def _ple_model(self) -> Any:
        """The text model when it uses per-layer embeddings (Gemma 4 E2B/E4B), else None.

        Those models look token ids up a second time for every layer. Given only embeddings
        they try to invert the embedding table, which fails for gists, think slots, media and
        padding; so UPP computes the per-layer inputs from clean token ids itself."""
        if self.is_mock:
            return None
        m = self.backbone.get_base_model() if hasattr(self.backbone, "get_base_model") else self.backbone
        if getattr(m, "hidden_size_per_layer_input", None) and hasattr(m, "get_per_layer_inputs"):
            return m
        return None

    def _clean_ids(self, ids: list[int]) -> list[int]:
        """Token ids for per-layer lookups: markers (slots, gists, media) read as padding."""
        pad = self._pad_id()
        return [i if i >= 0 else pad for i in ids]

    def _resolve_media(self, media: Any) -> dict[str, Any] | None:
        if not media:
            return None
        if isinstance(media, dict) and "embeds" in media:
            return media
        if self.media_encoder is None:
            raise ValueError("this UPPModel was built without media=True")
        dev = self._device
        if self._media_device != dev:
            self.media_encoder.to(dev)
            self._media_device = dev
        return self.media_encoder.encode(media)

    def _tok(self, text: str, special: bool = False) -> list[int]:
        if self.tokenizer is not None:
            return self.tokenizer.encode(text, add_special_tokens=special)
        return self._mock_tok(text)

    def _exit_layers(self) -> list[int]:
        if not self.cfg.exits or self.is_mock:
            return []
        n = int(getattr(getattr(self.backbone, "config", None), "num_hidden_layers", 0) or 0)
        return sorted({max(1, min(n - 1, round(f * n))) for f in self.cfg.exits}) if n else []

    def _hidden(self, embeds, mask, positions=None, past=None, use_cache=False, ids=None):
        """Returns (last hidden, list of exit hiddens, past). ``ids``: (B, L) clean token ids."""
        if self.is_mock:
            b = self.backbone
            L = embeds.shape[1]
            pos = positions if positions is not None else torch.arange(L, device=embeds.device).unsqueeze(0)
            x = embeds + b.pos_emb(pos.clamp(max=b.pos_emb.num_embeddings - 1))
            if mask.dim() == 4:  # additive (B,1,L,L) -> (B*heads, L, L)
                heads = b.transformer.layers[0].self_attn.num_heads
                h = b.transformer(x, mask=mask[:, 0].repeat_interleave(heads, dim=0).to(x.dtype))
            else:
                causal = nn.Transformer.generate_square_subsequent_mask(L, device=embeds.device).bool()
                h = b.transformer(x, mask=causal, src_key_padding_mask=(mask == 0), is_causal=True)
            return b.proj(h), [], None
        exits = self._exit_layers()
        extra: dict[str, Any] = {}
        ple = self._ple_model()
        if ple is not None:
            if ids is None:
                raise ValueError("per-layer-embedding backbones need token ids alongside embeddings")
            extra["per_layer_inputs"] = ple.get_per_layer_inputs(ids.to(embeds.device), None)
        out = self.backbone(
            inputs_embeds=embeds, attention_mask=mask, position_ids=positions, past_key_values=past,
            use_cache=use_cache, output_hidden_states=bool(exits), **extra,
        )
        h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        hs = [out.hidden_states[i] for i in exits] if exits else []
        return h, hs, getattr(out, "past_key_values", None)

    # ------------------------------------------------------------------ layout
    def _option_tokens(self, i: int, key: str, text: str, budget: int | None) -> list[int]:
        body = f"{key}: {str(text)[: self.cfg.max_option_chars]}\n"
        if self.cfg.layout != "block" and self.cfg.option_index:
            body = f"[{i}] " + body
        toks = self._tok(body)
        return toks[:budget] if budget is not None else toks

    def _media_tokens(self, media: dict[str, Any] | None) -> list[int]:
        return (self._tok("Input: ") + list(media["tokens"]) + self._tok("\n")) if media else []

    def _request_tokens(self, state: str, media: dict[str, Any] | None = None) -> list[int]:
        question = f"Question [tool_route]: {ROUTE_INSTRUCTIONS}\n"
        return (self._media_tokens(media) + self._tok(f"State: {state}\n") + self._tok(question)
                + [-1] * self.cfg.think_slots + self._tok("Answer:"))

    def build(self, state: str, criteria: dict[str, str], media: dict[str, Any] | None = None) -> dict[str, Any]:
        """Token ids (markers: -1 slot, -100-j gist j), positions, blocks and readout indices."""
        cfg = self.cfg
        keys = list(criteria)
        question = f"Question [tool_route]: {ROUTE_INSTRUCTIONS}\n"
        ids: list[int] = []
        positions: list[int] = []
        blocks: list[int] = []  # -1 prefix, i option i, -2 request
        is_gist: list[bool] = []
        opt_tokens: list[list[int]] = []
        opt_gists: list[list[int]] = []
        if cfg.layout == "block":
            prefix = self._tok("Options:\n", special=True)
            ids += prefix
            positions += list(range(len(prefix)))
            blocks += [-1] * len(prefix)
            is_gist += [False] * len(prefix)
            p0 = len(prefix)
            budget = max(8, cfg.request_offset - p0 - cfg.gists)
            for i, k in enumerate(keys):
                toks = self._option_tokens(i, k, criteria[k], budget)
                start = len(ids)
                ids += toks + [self.GIST_BASE - j for j in range(cfg.gists)]
                n_all = len(toks) + cfg.gists
                positions += list(range(p0, p0 + n_all))
                blocks += [i] * n_all
                is_gist += [False] * len(toks) + [True] * cfg.gists
                opt_tokens.append(list(range(start, start + len(toks))))
                opt_gists.append(list(range(start + len(toks), start + n_all)))
            catalog_len = len(ids)
            request = self._request_tokens(state, media)
            r_start = len(ids)
            ids += request
            positions += list(range(cfg.request_offset, cfg.request_offset + len(request)))
            blocks += [-2] * len(request)
            is_gist += [False] * len(request)
            req_idx = list(range(r_start, len(ids)))
            q_idx = len(ids) - 1
        elif cfg.layout == "catalog_first":
            ids += self._tok("Options:\n", special=True)
            for i, k in enumerate(keys):
                start = len(ids)
                ids += self._option_tokens(i, k, criteria[k], None)
                opt_tokens.append(list(range(start, len(ids))))
            catalog_len = len(ids)
            r_start = len(ids)
            ids += self._request_tokens(state, media)
            req_idx = list(range(r_start, len(ids)))
            positions = list(range(len(ids)))
            q_idx = len(ids) - 1
        else:
            if media:
                ids += self._tok("", special=True) + self._media_tokens(media) + self._tok(f"State: {state}\n")
            else:
                ids += self._tok(f"State: {state}\n", special=True)
            ids += self._tok(question)
            ids += [-1] * cfg.think_slots
            ids += self._tok("Options:\n")
            q_idx = len(ids) - 1
            req_idx = list(range(0, q_idx + 1))
            if cfg.option_isolation:
                p_opt = len(ids)
                positions = list(range(p_opt))
                blocks = [-1] * p_opt
                for i, k in enumerate(keys):
                    toks = self._option_tokens(i, k, criteria[k], None)
                    start = len(ids)
                    ids += toks
                    positions += list(range(p_opt, p_opt + len(toks)))
                    blocks += [i] * len(toks)
                    opt_tokens.append(list(range(start, len(ids))))
            else:
                for i, k in enumerate(keys):
                    start = len(ids)
                    ids += self._option_tokens(i, k, criteria[k], None)
                    opt_tokens.append(list(range(start, len(ids))))
                positions = list(range(len(ids)))
            catalog_len = 0
        if cfg.gists:
            opt_idx = opt_gists
        elif cfg.option_pool == "last":
            opt_idx = [[t[-1]] for t in opt_tokens]
        else:
            opt_idx = opt_tokens
        return {"ids": ids, "positions": positions, "blocks": blocks or None, "is_gist": is_gist or None,
                "q_idx": q_idx, "req_idx": req_idx, "opt_idx": opt_idx, "opt_tokens": opt_tokens,
                "opt_gists": opt_gists, "keys": keys, "catalog_len": catalog_len,
                "media": media["embeds"] if media else None}

    def _block_allowed(self, pk: dict[str, Any], L: int, mode: str) -> torch.Tensor:
        """(L, L) bool. mode: gist (request sees gists), full (request sees option tokens), none (late)."""
        blocks, gist = pk["blocks"], pk.get("visible_gist") or pk["is_gist"]
        n = len(blocks)
        b = torch.tensor(blocks)
        g = torch.tensor(gist[:n])
        i_b, j_b = b.view(n, 1), b.view(1, n)
        causal = torch.ones(n, n, dtype=torch.bool).tril()
        same = i_b == j_b
        to_prefix = j_b == -1
        from_request = i_b == -2
        if mode == "none":
            # late interaction: the request is encoded completely alone (not even the shared
            # prefix), so its states are identical whether or not a catalog is in the batch
            req_sees = torch.zeros(1, n, dtype=torch.bool)
            to_prefix = to_prefix & ~from_request
        elif mode == "gist" and self.cfg.gists:
            req_sees = g.view(1, n)
        else:  # full: every option token (and gists)
            req_sees = j_b >= 0
        allowed = causal & (same | to_prefix | (from_request & req_sees))
        out = torch.zeros(L, L, dtype=torch.bool)
        out[:n, :n] = allowed
        idx = torch.arange(L)
        out[idx, idx] = True
        return out

    def _query_iso_allowed(self, pk: dict[str, Any], L: int) -> torch.Tensor:
        """(L, L) bool for query_first + option_isolation: options see prefix and themselves only."""
        blocks = pk["blocks"]
        n = len(blocks)
        b = torch.tensor(blocks)
        i_b, j_b = b.view(n, 1), b.view(1, n)
        causal = torch.ones(n, n, dtype=torch.bool).tril()
        allowed = causal & ((i_b == j_b) | (j_b == -1))
        out = torch.zeros(L, L, dtype=torch.bool)
        out[:n, :n] = allowed
        idx = torch.arange(L)
        out[idx, idx] = True
        return out

    def _with_budget(self, pk: dict[str, Any], budget: int | None) -> dict[str, Any]:
        """Same row, but the request may only read the first ``budget`` gists of each option."""
        if not self.cfg.gists or budget is None or budget >= self.cfg.gists:
            return pk
        vis = [False] * len(pk["ids"])
        for gi in pk["opt_gists"]:
            for j in gi[:budget]:
                vis[j] = True
        return dict(pk, visible_gist=vis, opt_idx=[gi[:budget] for gi in pk["opt_gists"]])

    def _mode(self, teacher: bool = False) -> str:
        if self.cfg.interaction == "late":
            return "none"
        if teacher:
            return "full"
        return "gist" if self.cfg.gists else "full"

    def _embed(self, ids: list[int], device: torch.device, media: torch.Tensor | None = None) -> torch.Tensor:
        emb = self._embed_layer()
        x = emb(torch.tensor([max(i, 0) for i in ids], dtype=torch.long, device=device))
        slot_pos = [j for j, i in enumerate(ids) if i == -1]
        gist_pos = [(j, self.GIST_BASE - i) for j, i in enumerate(ids) if i <= self.GIST_BASE]
        media_pos = [j for j, i in enumerate(ids) if i == self.MEDIA]
        if media_pos:
            if media is None or media.shape[0] != len(media_pos):
                got = None if media is None else tuple(media.shape)
                raise ValueError(f"{len(media_pos)} media slots but media embeddings {got}")
            x = x.clone()
            x[media_pos] = media.to(device=x.device, dtype=x.dtype)
        if slot_pos or gist_pos:
            x = x.clone()
            if slot_pos and self.slots is not None:
                x[slot_pos] = self.slots.to(x.dtype)
            if gist_pos and self.gist_emb is not None:
                x[[p for p, _ in gist_pos]] = self.gist_emb[[gg for _, gg in gist_pos]].to(x.dtype)
        return x

    def _batch(self, packs: list[dict[str, Any]], device: torch.device, mode: str):
        L = max(len(p["ids"]) for p in packs)
        emb_dim = self._embed_layer().weight.shape[1]
        dtype = self._embed_layer().weight.dtype
        embeds = torch.zeros(len(packs), L, emb_dim, device=device, dtype=dtype)
        positions = torch.zeros(len(packs), L, dtype=torch.long, device=device)
        tok_ids = torch.full((len(packs), L), self._pad_id(), dtype=torch.long, device=device)
        use_4d = self.cfg.layout == "block" or (self.cfg.layout == "query_first" and self.cfg.option_isolation)
        if use_4d:
            neg = torch.finfo(dtype if dtype.is_floating_point else torch.float32).min
            mask = torch.full((len(packs), 1, L, L), neg, device=device, dtype=dtype)
        else:
            mask = torch.zeros(len(packs), L, dtype=torch.long, device=device)
        for b, pk in enumerate(packs):
            n = len(pk["ids"])
            embeds[b, :n] = self._embed(pk["ids"], device, pk.get("media"))
            tok_ids[b, :n] = torch.tensor(self._clean_ids(pk["ids"]), device=device)
            positions[b, :n] = torch.tensor(pk["positions"], device=device)
            if n < L:
                positions[b, n:] = positions[b, n - 1] + 1
            if self.cfg.layout == "block":
                mask[b, 0][self._block_allowed(pk, L, mode).to(device)] = 0.0
            elif use_4d:
                mask[b, 0][self._query_iso_allowed(pk, L).to(device)] = 0.0
            else:
                mask[b, :n] = 1
        return embeds, mask, positions, tok_ids

    # ------------------------------------------------------------------ readout
    def _pool(self, h: torch.Tensor) -> torch.Tensor:
        if h.shape[0] == 1:
            return h[0]
        if self.pool is not None:
            return self.pool(h)
        return h.float().mean(0)

    def _score(self, h: torch.Tensor, pk: dict[str, Any], opt_idx: list[list[int]] | None = None):
        """Logits for one row from its hidden states (L, d)."""
        opt_idx = opt_idx or pk["opt_idx"]
        if self.cfg.interaction == "late":
            g = max(len(ix) for ix in opt_idx)
            gist = torch.stack([h[ix + [ix[-1]] * (g - len(ix))] for ix in opt_idx])
            return self.head(h[pk["req_idx"]], gist)
        opts = torch.stack([self._pool(h[ix]) for ix in opt_idx])
        if self.cfg.readout == "xattn":
            return self.head(h[pk["req_idx"]], opts)
        return self.head(h[pk["q_idx"]], opts)

    # ------------------------------------------------------------------ forward
    def forward_batch(self, records: list[dict[str, Any]], max_seq_len: int = 1024) -> dict[str, Any]:
        device = self._device
        packs, golds, kept_recs = [], [], []
        for r in records:
            q = r["questions"]["tool_route"]
            pk = self.build(r["state"], q["criteria"], media=self._resolve_media(r.get("media")))
            if len(pk["ids"]) > max_seq_len:
                continue
            packs.append(pk)
            golds.append(pk["keys"].index(q["label"]) if q["label"] in pk["keys"] else 0)
            kept_recs.append(r)
        if not packs:
            zero = sum(p.sum() * 0.0 for p in self.parameters() if p.requires_grad)
            return {"loss": zero, "skipped": True, "logits": [], "kept": 0}
        if self.cfg.gist_budgets and self.training:
            b = self.cfg.gist_budgets[int(torch.randint(len(self.cfg.gist_budgets), (1,)))]
            packs = [self._with_budget(pk, b) for pk in packs]
        embeds, mask, positions, tok_ids = self._batch(packs, device, self._mode())
        h, exit_hs, _ = self._hidden(embeds, mask, positions, ids=tok_ids)
        teacher_logits = None
        if self.cfg.gist_distill > 0 and self.cfg.gists and self.cfg.interaction == "cross":
            # Same model, request reading the full option text: the gist path learns to match it.
            with torch.no_grad():
                t_embeds, t_mask, t_pos, t_ids = self._batch(packs, device, self._mode(teacher=True))
                th, _, _ = self._hidden(t_embeds, t_mask, t_pos, ids=t_ids)
                teacher_logits = [self._score(th[b], pk, [[t[-1]] for t in pk["opt_tokens"]])[0].detach()
                                  for b, pk in enumerate(packs)]
        losses, logits_out, us = [], [], []
        n_exit = len(exit_hs)
        for b, pk in enumerate(packs):
            g = golds[b]
            costs_dict = ((kept_recs[b].get("meta") or {}).get("s2") or {}).get("costs")
            cost_vec = (
                [float(costs_dict.get(k, 1.0)) for k in pk["keys"]]
                if (self.cfg.cost_weight > 0 and costs_dict)
                else None
            )
            t_probs = (kept_recs[b].get("teacher") or {}).get("probs")
            teacher_vec: list[float] | None = None
            if self.cfg.teacher_kl_weight > 0 and isinstance(t_probs, dict):
                raw_tp = [max(0.0, float(t_probs.get(k, 0.0))) for k in pk["keys"]]
                s_tp = sum(raw_tp)
                if s_tp > 1e-8:
                    teacher_vec = [v / s_tp for v in raw_tp]
            logits, aux = self._score(h[b], pk)
            row = self._row_loss(logits, aux, g, cost_vec=cost_vec, teacher_probs=teacher_vec)
            if teacher_logits is not None:
                pt = F.softmax(teacher_logits[b], -1)
                row = row + self.cfg.gist_distill * F.kl_div(F.log_softmax(logits, -1), pt, reduction="sum")
            losses.append(row)
            for hs in exit_hs:
                lg_e, aux_e = self._score(hs[b], pk)
                losses.append(
                    self.cfg.exit_weight
                    / n_exit
                    * self._row_loss(lg_e, aux_e, g, cost_vec=cost_vec, teacher_probs=teacher_vec)
                )
            logits_out.append(logits)
            us.append(float(aux["u"].detach()))
        loss = torch.stack(losses).sum() / len(packs)
        acc = sum(int(torch.argmax(lg).item() == g) for lg, g in zip(logits_out, golds)) / len(golds)
        return {"loss": loss, "skipped": False, "logits": logits_out, "kept": len(packs), "acc": acc,
                "u": sum(us) / len(us), "keys": [pk["keys"] for pk in packs]}

    def _row_loss(
        self,
        logits: torch.Tensor,
        aux: dict[str, torch.Tensor],
        g: int,
        cost_vec: list[float] | None = None,
        teacher_probs: list[float] | None = None,
    ) -> torch.Tensor:
        p = F.softmax(logits, -1)
        oh = torch.zeros_like(p)
        oh[g] = 1.0
        loss = -torch.log(p[g].clamp_min(1e-9)) + self.cfg.brier_weight * ((p - oh) ** 2).mean()
        if cost_vec is not None and self.cfg.cost_weight > 0:
            c = torch.as_tensor(cost_vec, dtype=p.dtype, device=logits.device)
            loss = loss + self.cfg.cost_weight * (p * c).sum()
        if teacher_probs is not None and self.cfg.teacher_kl_weight > 0:
            pt = torch.as_tensor(teacher_probs, dtype=torch.float32, device=logits.device).clamp_min(1e-9)
            pt = pt / pt.sum()
            loss = loss + self.cfg.teacher_kl_weight * F.kl_div(F.log_softmax(logits.float(), -1), pt, reduction="sum")
        if "member_logits" in aux:
            loss = loss + F.cross_entropy(aux["member_logits"], torch.full((aux["member_logits"].shape[0],), g,
                                          device=logits.device))
        if "kl" in aux:
            loss = loss + self.cfg.kl_weight * aux["kl"]
        return loss

    @torch.no_grad()
    def score(self, state: str, criteria: dict[str, str], budget: int | None = None,
              media: Any = None) -> tuple[dict[str, float], dict[str, Any]]:
        """Probabilities over ``criteria`` for one request, plus uncertainty and exit info.

        ``budget`` limits how many gists per option the request reads (Matryoshka models)."""
        device = self._device
        pk = self._with_budget(self.build(state, criteria, media=self._resolve_media(media)), budget)
        embeds, mask, positions, tok_ids = self._batch([pk], device, self._mode())
        h, exit_hs, _ = self._hidden(embeds, mask, positions, ids=tok_ids)
        logits, aux = self._score(h[0], pk)
        p = F.softmax(logits, -1).float().cpu().tolist()
        info: dict[str, Any] = {"u": float(aux["u"]), "energy": float(aux["energy"])}
        if exit_hs:
            info["exits"] = []
            for hs in exit_hs:
                lg, _ = self._score(hs[0], pk)
                pe = F.softmax(lg, -1).float().cpu().tolist()
                info["exits"].append({k: pe[i] for i, k in enumerate(pk["keys"])})
        return {k: p[i] for i, k in enumerate(pk["keys"])}, info

    @torch.no_grad()
    def batch_score(
        self,
        items: list[tuple[str, dict[str, str], Any]],
        budget: int | None = None,
        batch_size: int = 8,
    ) -> list[tuple[dict[str, float], dict[str, Any]]]:
        """Batched inference counterpart of ``score`` over (state, criteria, media) tuples."""
        device = self._device
        out: list[tuple[dict[str, float], dict[str, Any]]] = []
        mode = self._mode()
        for start in range(0, len(items), max(1, batch_size)):
            chunk = items[start : start + max(1, batch_size)]
            packs = [
                self._with_budget(self.build(st, crit, media=self._resolve_media(med)), budget)
                for st, crit, med in chunk
            ]
            embeds, mask, positions, tok_ids = self._batch(packs, device, mode)
            h, exit_hs, _ = self._hidden(embeds, mask, positions, ids=tok_ids)
            for b, pk in enumerate(packs):
                logits, aux = self._score(h[b], pk)
                p = F.softmax(logits, -1).float().cpu().tolist()
                info: dict[str, Any] = {"u": float(aux["u"]), "energy": float(aux["energy"])}
                if exit_hs:
                    info["exits"] = []
                    for hs in exit_hs:
                        lg, _ = self._score(hs[b], pk)
                        pe = F.softmax(lg, -1).float().cpu().tolist()
                        info["exits"].append({k: pe[i] for i, k in enumerate(pk["keys"])})
                out.append(({k: p[i] for i, k in enumerate(pk["keys"])}, info))
        return out

    # ------------------------------------------------------------------ caching
    @staticmethod
    def _select(past: Any, keep: list[int]) -> Any:
        """A new cache holding only the given key/value positions of ``past``."""
        from transformers import DynamicCache

        out = DynamicCache()
        for li, layer in enumerate(past.layers):
            idx = torch.tensor(keep, device=layer.keys.device)
            out.update(layer.keys.index_select(2, idx), layer.values.index_select(2, idx), li)
        return out

    def _prefix_len(self, pk: dict[str, Any]) -> int:
        return next(i for i, b in enumerate(pk["blocks"]) if b != -1)

    @torch.no_grad()
    def encode_catalog(self, criteria: dict[str, str]) -> dict[str, Any]:
        """Run the options once (catalog_first or block) and keep what requests need.

        With gists only the prefix and the gist positions are kept (prefix + K*g
        tokens). Late interaction keeps only the gist vectors, no KV at all.
        """
        if self.cfg.layout not in ("catalog_first", "block") or self.is_mock:
            raise ValueError("catalog caching needs layout catalog_first/block and a real backbone")
        device = self._device
        pk = self.build("", criteria)
        n = pk["catalog_len"]
        cat = dict(pk, ids=pk["ids"][:n], positions=pk["positions"][:n],
                   blocks=pk["blocks"][:n] if pk["blocks"] else None,
                   is_gist=pk["is_gist"][:n] if pk["is_gist"] else None)
        embeds, mask, positions, tok_ids = self._batch([cat], device, self._mode())
        h, _, past = self._hidden(embeds, mask, positions, use_cache=True, ids=tok_ids)
        opt_states = [h[0, ix].clone() for ix in pk["opt_idx"]]
        prefix_len = self._prefix_len(pk) if pk["blocks"] else 0
        if self.cfg.interaction == "late":
            past = None
        elif self.cfg.gists:
            keep = list(range(prefix_len)) + [i for ix in pk["opt_gists"] for i in ix]
            past = self._select(past, keep)
        return {"criteria": dict(criteria), "past": past, "opt_states": opt_states, "keys": pk["keys"],
                "catalog_len": n, "prefix_len": prefix_len,
                "cached_tokens": past.get_seq_length() if past is not None else 0}

    @torch.no_grad()
    def _score_request(self, state: str, cat: dict[str, Any], fresh: bool = False, media: Any = None,
                       ) -> tuple[dict[str, float], dict[str, Any]]:
        """Run only the request against a prepared catalog (cached KV or late-interaction vectors).

        The cache is not copied: the request's keys/values are appended and then cropped away,
        so the catalog cache is unchanged afterwards. ``fresh`` marks a throwaway sub-cache."""
        device = self._device
        m = self._resolve_media(media)
        request = self._request_tokens(state, m)
        x = self._embed(request, device, m["embeds"] if m else None).unsqueeze(0)
        req_ids = torch.tensor([self._clean_ids(request)], dtype=torch.long, device=device)
        start = self.cfg.request_offset if self.cfg.layout == "block" else cat["catalog_len"]
        pos = torch.arange(start, start + len(request), device=device).unsqueeze(0)
        if self.cfg.interaction == "late":
            h, _, _ = self._hidden(x, torch.ones(1, len(request), dtype=torch.long, device=device), pos,
                                   ids=req_ids)
            g = max(s.shape[0] for s in cat["opt_states"])
            gist = torch.stack([torch.cat([s, s[-1:].expand(g - s.shape[0], -1)]) for s in cat["opt_states"]])
            logits, aux = self.head(h[0], gist)
        else:
            mask = torch.ones(1, cat["cached_tokens"] + len(request), dtype=torch.long, device=device)
            past = cat["past"]
            try:
                h, _, _ = self._hidden(x, mask, pos, past=past, use_cache=True, ids=req_ids)
            finally:
                if not fresh:
                    # drop the request's entries; catalog cache unchanged (negative crop = remove that
                    # many tokens; positive crop is deprecated in transformers >= 5.18)
                    extra_tokens = past.get_seq_length() - cat["cached_tokens"]
                    if extra_tokens > 0:
                        past.crop(-extra_tokens)
            opts = torch.stack([self._pool(s) for s in cat["opt_states"]])
            logits, aux = self.head(h[0] if self.cfg.readout == "xattn" else h[0, -1], opts)
        p = F.softmax(logits, -1).float().cpu().tolist()
        return {k: p[i] for i, k in enumerate(cat["keys"])}, {"u": float(aux["u"])}

    def score_cached(self, state: str, cache: dict[str, Any], media: Any = None,
                     ) -> tuple[dict[str, float], dict[str, Any]]:
        return self._score_request(state, cache, media=media)

    # ------------------------------------------------------------------ per-tool composable cache
    @torch.no_grad()
    def encode_tool(self, key: str, text: str) -> dict[str, Any]:
        """Encode one tool in isolation; its cache does not depend on any other tool."""
        if not (self.cfg.layout == "block" and (self.cfg.gists or self.cfg.interaction == "late")) or self.is_mock:
            raise ValueError("per-tool caches need layout='block' with gists (or late interaction)")
        c = self.encode_catalog({key: text})
        return {"key": key, "past": c["past"], "opt_state": c["opt_states"][0], "prefix_len": c["prefix_len"]}

    def compose(self, tools: list[dict[str, Any]]) -> dict[str, Any]:
        """Join per-tool caches (any order): the shared prefix once, then each tool's gists."""
        from transformers import DynamicCache

        past = None
        if self.cfg.interaction != "late":
            past = DynamicCache()
            pl = tools[0]["prefix_len"]
            for li in range(len(tools[0]["past"].layers)):
                ks = [tools[0]["past"].layers[li].keys[:, :, :pl]] + [t["past"].layers[li].keys[:, :, pl:] for t in tools]
                vs = [tools[0]["past"].layers[li].values[:, :, :pl]] + [t["past"].layers[li].values[:, :, pl:] for t in tools]
                past.update(torch.cat(ks, 2), torch.cat(vs, 2), li)
        return {"past": past, "opt_states": [t["opt_state"] for t in tools], "keys": [t["key"] for t in tools],
                "catalog_len": None, "prefix_len": tools[0]["prefix_len"],
                "cached_tokens": past.get_seq_length() if past is not None else 0}

    def score_composed(self, state: str, catalog: dict[str, Any], media: Any = None,
                       ) -> tuple[dict[str, float], dict[str, Any]]:
        return self._score_request(state, catalog, media=media)

    def _sub_catalog(self, cat: dict[str, Any], tool_ids: list[int], budget: int) -> dict[str, Any]:
        """View of a gist catalog with only ``tool_ids`` and their first ``budget`` gists."""
        g, pl = self.cfg.gists, cat["prefix_len"]
        keep = list(range(pl)) + [pl + i * g + j for i in tool_ids for j in range(budget)]
        past = self._select(cat["past"], keep) if cat["past"] is not None else None
        return {"past": past, "opt_states": [cat["opt_states"][i][:budget] for i in tool_ids],
                "keys": [cat["keys"][i] for i in tool_ids], "catalog_len": cat.get("catalog_len"),
                "prefix_len": pl, "cached_tokens": len(keep) if past is not None else 0}

    def score_two_level(self, state: str, cat: dict[str, Any], coarse: int = 1, top_m: int = 16,
                        fine: int | None = None, media: Any = None) -> tuple[dict[str, float], dict[str, Any]]:
        """Coarse pass over every tool with ``coarse`` gists, then a fine pass over the top-M with
        ``fine`` gists, both read from the same cache. ``__abstain__`` always reaches the fine pass.
        With coarse = fine = gists and top_m >= catalog size this equals ``score_cached``."""
        if not self.cfg.gists or self.cfg.interaction == "late":
            raise ValueError("two-level scoring needs a cross-interaction gist model")
        fine = fine or self.cfg.gists
        n = len(cat["keys"])
        media = self._resolve_media(media)  # encode once for both passes
        if coarse >= self.cfg.gists:
            p_coarse, _ = self._score_request(state, cat, media=media)
        else:
            p_coarse, _ = self._score_request(state, self._sub_catalog(cat, list(range(n)), coarse), fresh=True,
                                              media=media)
        order = sorted(range(n), key=lambda i: -p_coarse[cat["keys"][i]])
        top = order[:top_m]
        if "__abstain__" in cat["keys"]:
            ab = cat["keys"].index("__abstain__")
            if ab not in top:
                top = top[: max(0, top_m - 1)] + [ab]
        top = sorted(top)  # menu order does not matter (order-invariant), keep catalog order
        p_fine, info = self._score_request(state, self._sub_catalog(cat, top, fine), fresh=True, media=media)
        info.update({"coarse_kept": len(top), "coarse_tokens": cat["prefix_len"] + n * coarse,
                     "fine_tokens": cat["prefix_len"] + len(top) * fine})
        return p_fine, info

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def config_dict(self) -> dict[str, Any]:
        return {"model_name": self.model_name, **asdict(self.cfg)}

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_dir: Any,
        use_mock: bool = False,
        load_in_4bit: bool = False,
        torch_dtype: torch.dtype | None = None,
        media: bool = False,
    ) -> "UPPModel":
        from dataclasses import fields
        from pathlib import Path

        ckpt = Path(ckpt_dir)
        try:
            state = torch.load(ckpt / "head.pt", map_location="cpu", weights_only=True)
        except Exception:
            state = torch.load(ckpt / "head.pt", map_location="cpu", weights_only=False)
        cfg_raw = dict(state["cfg"])
        model_name = cfg_raw.pop("model_name", "Qwen/Qwen3-0.6B-Base")
        valid_fields = {f.name for f in fields(UPPConfig)}
        cfg_clean: dict[str, Any] = {}
        for k, v in cfg_raw.items():
            if k in valid_fields:
                if k in ("exits", "gist_budgets") and isinstance(v, list):
                    v = tuple(v)
                cfg_clean[k] = v
        model = cls(
            UPPConfig(**cfg_clean),
            model_name=model_name,
            use_mock=use_mock,
            load_in_4bit=load_in_4bit,
            torch_dtype=torch_dtype,
            media=media,
        )
        model.head.load_state_dict(state["head"])
        if "pool" in state and model.pool is not None:
            model.pool.load_state_dict(state["pool"])
        if "slots" in state and model.slots is not None:
            with torch.no_grad():
                model.slots.copy_(state["slots"].to(model.slots.dtype))
        if "gist_emb" in state and model.gist_emb is not None:
            with torch.no_grad():
                model.gist_emb.copy_(state["gist_emb"].to(model.gist_emb.dtype))
        adapter_dir = ckpt / "adapter"
        if not model.is_mock and adapter_dir.exists():
            from peft import set_peft_model_state_dict

            sf = adapter_dir / "adapter_model.safetensors"
            if sf.exists():
                from safetensors.torch import load_file

                adapter_sd = load_file(str(sf))
            else:
                adapter_sd = torch.load(adapter_dir / "adapter_model.bin", map_location="cpu", weights_only=True)
            res = set_peft_model_state_dict(model.backbone, adapter_sd)
            if getattr(res, "unexpected_keys", None):
                raise RuntimeError(f"Unexpected LoRA adapter keys: {res.unexpected_keys[:5]}")
        return model
