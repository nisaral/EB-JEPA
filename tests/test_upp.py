"""Unit tests for the Universal Probabilistic Pointer (no weights needed)."""
from __future__ import annotations

import torch
from soro.routing.upp import UPPConfig, UPPHead, UPPModel

D = 32


def _head(**kw) -> UPPHead:
    torch.manual_seed(0)
    return UPPHead(D, UPPConfig(proj_dim=16, **kw)).eval()


def test_refiner_is_order_equivariant():
    head = _head(head="gauss", refiner=True)
    h_q, h_opts = torch.randn(D), torch.randn(5, D)
    perm = torch.tensor([3, 0, 4, 1, 2])
    with torch.no_grad():
        a, _ = head(h_q, h_opts)
        b, _ = head(h_q, h_opts[perm])
    assert torch.allclose(a[perm], b, atol=1e-5)


def test_gauss_prefers_matching_option_and_reports_variance():
    head = _head(head="gauss")
    h_q = torch.randn(D)
    h_opts = torch.stack([torch.randn(D), h_q.clone(), torch.randn(D)])
    with torch.no_grad():
        logits, aux = head(h_q, h_opts)
    assert int(torch.argmax(logits)) == 1
    assert {"kl", "u", "energy"} <= set(aux)
    assert float(aux["u"]) > 0


def test_dot_head_reports_energy_and_fallback_uncertainty():
    logits, aux = _head(head="dot")(torch.randn(D), torch.randn(4, D))
    assert logits.shape == (4,) and {"energy", "u"} <= set(aux)


def test_rbf_head_top_logit_is_the_nearest_option():
    head = _head(head="rbf")
    member = head.members[0]
    with torch.no_grad():
        logits, aux = head(torch.randn(D), torch.randn(4, D))
        temp = float(torch.exp(member.log_temp)) + 1e-4
    # logit = -0.5 * d2 * sqrt(p) / temp, and u is the smallest d2, so the top logit matches u.
    expected_top = -0.5 * float(aux["u"]) * (16 ** 0.5) / temp
    assert abs(float(logits.max()) - expected_top) < 1e-4 and float(aux["u"]) >= 0


def test_ensemble_averages_members_and_reports_disagreement():
    head = _head(head="dot", ensemble=3)
    with torch.no_grad():
        logits, aux = head(torch.randn(D), torch.randn(4, D))
    assert aux["member_logits"].shape == (3, 4)
    assert abs(float(torch.softmax(logits, -1).sum()) - 1) < 1e-5 and float(aux["u"]) > 0


def test_block_layout_is_order_invariant_on_mock():
    torch.manual_seed(0)
    model = UPPModel(UPPConfig(layout="block", head="rbf", refiner=True), use_mock=True).eval()
    crit = {"get_weather": "weather", "send_email": "email someone", "__abstain__": "none"}
    a, _ = model.score("User Request: rain tomorrow?", crit)
    b, _ = model.score("User Request: rain tomorrow?", dict(reversed(list(crit.items()))))
    assert max(abs(a[k] - b[k]) for k in a) < 1e-5


def test_layouts_place_query_and_slots():
    crit = {"a": "tool a", "b": "tool b", "__abstain__": "none"}
    cat = UPPModel(UPPConfig(layout="catalog_first", think_slots=3), use_mock=True)
    pk = cat.build("User Request: hi", crit)
    assert pk["ids"].count(-1) == 3
    assert max(i for ix in pk["opt_idx"] for i in ix) < pk["catalog_len"] <= pk["q_idx"] == len(pk["ids"]) - 1
    blk = UPPModel(UPPConfig(layout="block"), use_mock=True).build("User Request: hi", crit)
    starts = [blk["positions"][blk["blocks"].index(i)] for i in range(3)]
    assert len(set(starts)) == 1  # every option block starts at the same position
    qf = UPPModel(UPPConfig(layout="query_first"), use_mock=True)
    pk2 = qf.build("User Request: hi", crit)
    assert pk2["q_idx"] < min(i for ix in pk2["opt_idx"] for i in ix) and pk2["catalog_len"] == 0


def test_forward_batch_and_score_run_on_mock():
    model = UPPModel(UPPConfig(layout="catalog_first", head="gauss", refiner=True, think_slots=2), use_mock=True)
    rec = {"state": "User Request: weather in Paris",
           "questions": {"tool_route": {"criteria": {"get_weather": "weather", "__abstain__": "none"},
                                        "label": "get_weather"}}}
    out = model.forward_batch([rec, rec])
    assert not out["skipped"] and out["kept"] == 2
    out["loss"].backward()
    assert model.slots.grad is not None
    probs, info = model.score(rec["state"], rec["questions"]["tool_route"]["criteria"])
    assert abs(sum(probs.values()) - 1.0) < 1e-5 and info["u"] > 0


REC = {"state": "User Request: weather in Paris",
       "questions": {"tool_route": {"criteria": {"get_weather": "weather forecast for a city",
                                                 "send_email": "send an email", "__abstain__": "none"},
                                    "label": "get_weather"}}}


def test_gist_mask_hides_raw_option_tokens_from_request():
    m = UPPModel(UPPConfig(layout="block", gists=2), use_mock=True)
    pk = m.build(REC["state"], REC["questions"]["tool_route"]["criteria"])
    allowed = m._block_allowed(pk, len(pk["ids"]), "gist")
    q = pk["q_idx"]
    raw = [i for t in pk["opt_tokens"] for i in t]
    gists = [i for g in pk["opt_gists"] for i in g]
    assert not allowed[q, raw].any() and allowed[q, gists].all()
    full = m._block_allowed(pk, len(pk["ids"]), "full")
    assert full[q, raw].all()
    none = m._block_allowed(pk, len(pk["ids"]), "none")
    assert not none[q, raw + gists].any()
    # options never see each other
    a, b = pk["opt_tokens"][0][-1], pk["opt_tokens"][1][0]
    assert not allowed[b, a] and not allowed[a, b]


def test_block_positions_do_not_depend_on_other_tools():
    m = UPPModel(UPPConfig(layout="block", gists=2), use_mock=True)
    crit = REC["questions"]["tool_route"]["criteria"]
    alone = m.build("x", {"get_weather": crit["get_weather"]})
    joint = m.build("x", crit)
    pos_alone = [alone["positions"][i] for i in alone["opt_gists"][0]]
    pos_joint = [joint["positions"][i] for i in joint["opt_gists"][0]]
    assert pos_alone == pos_joint
    assert joint["positions"][joint["q_idx"] - 1] >= m.cfg.request_offset


def test_late_interaction_and_gist_distill_train_on_mock():
    torch.manual_seed(0)
    late = UPPModel(UPPConfig(layout="block", gists=2, interaction="late"), use_mock=True)
    out = late.forward_batch([REC, REC])
    out["loss"].backward()
    assert late.gist_emb.grad is not None and not out["skipped"]
    probs, _ = late.score(REC["state"], REC["questions"]["tool_route"]["criteria"])
    assert abs(sum(probs.values()) - 1) < 1e-5
    gsd = UPPModel(UPPConfig(layout="block", gists=2, gist_distill=1.0, option_pool="attn"), use_mock=True)
    out = gsd.forward_batch([REC])
    out["loss"].backward()
    assert gsd.gist_emb.grad is not None and gsd.pool is not None  # attn pooling applies to gists too
    assert gsd.pool.query.grad is not None


def test_matryoshka_budget_masks_later_gists_and_trains():
    torch.manual_seed(0)
    m = UPPModel(UPPConfig(layout="block", gists=4, gist_budgets=(1, 2, 4)), use_mock=True)
    pk = m.build(REC["state"], REC["questions"]["tool_route"]["criteria"])
    b1 = m._with_budget(pk, 1)
    allowed = m._block_allowed(b1, len(pk["ids"]), "gist")
    q = pk["q_idx"]
    first = [g[0] for g in pk["opt_gists"]]
    later = [i for g in pk["opt_gists"] for i in g[1:]]
    assert allowed[q, first].all() and not allowed[q, later].any()
    assert all(len(ix) == 1 for ix in b1["opt_idx"])
    m.train()
    out = m.forward_batch([REC, REC])
    out["loss"].backward()
    assert m.gist_emb.grad is not None
    p1, _ = m.eval().score(REC["state"], REC["questions"]["tool_route"]["criteria"], budget=1)
    assert abs(sum(p1.values()) - 1) < 1e-5


def _media(model: UPPModel, n: int = 5, seed: int = 0) -> dict:
    """A pre-encoded spoken/image request: soft tokens wrapped by two ordinary ids."""
    g = torch.Generator().manual_seed(seed)
    d = model._embed_layer().weight.shape[1]
    return {"tokens": [7] + [UPPModel.MEDIA] * n + [8], "embeds": torch.randn(n, d, generator=g)}


def test_media_request_is_placed_in_request_block_and_stays_order_invariant():
    torch.manual_seed(0)
    m = UPPModel(UPPConfig(layout="block", gists=2), use_mock=True).eval()
    crit = REC["questions"]["tool_route"]["criteria"]
    media = _media(m)
    pk = m.build("", crit, media=media)
    slots = [j for j, i in enumerate(pk["ids"]) if i == UPPModel.MEDIA]
    assert len(slots) == 5 and all(pk["blocks"][j] == -2 for j in slots)  # inside the request block
    assert min(pk["positions"][j] for j in slots) >= m.cfg.request_offset
    a, _ = m.score("", crit, media=media)
    b, _ = m.score("", dict(reversed(list(crit.items()))), media=media)
    assert max(abs(a[k] - b[k]) for k in a) < 1e-5
    c, _ = m.score("", crit, media=_media(m, seed=1))  # the media content reaches the decision
    assert max(abs(a[k] - c[k]) for k in a) > 1e-6


def test_media_rows_train_and_marker_ids_read_as_padding():
    m = UPPModel(UPPConfig(layout="block", gists=2), use_mock=True)
    rec = dict(REC, state="", media=_media(m))
    out = m.forward_batch([rec, REC])
    assert not out["skipped"] and out["kept"] == 2
    out["loss"].backward()
    pk = m.build("", REC["questions"]["tool_route"]["criteria"], media=rec["media"])
    _, _, _, ids = m._batch([pk], torch.device("cpu"), m._mode())
    assert int(ids.min()) >= 0  # gists and media never reach a per-layer lookup as negative ids
    assert all(ids[0, j].item() == m._pad_id() for j, i in enumerate(pk["ids"]) if i < 0)


def test_media_slot_count_must_match_embeddings():
    import pytest

    m = UPPModel(UPPConfig(layout="block"), use_mock=True)
    bad = dict(_media(m), embeds=torch.zeros(3, m._embed_layer().weight.shape[1]))
    with pytest.raises(ValueError):
        m.score("", {"a": "tool a", "__abstain__": "none"}, media=bad)
    with pytest.raises(ValueError):  # raw paths need media=True
        m.score("", {"a": "tool a", "__abstain__": "none"}, media={"audio": "x.wav"})


def test_xattn_readout_is_order_equivariant_and_trains_on_mock():
    from soro.routing.upp import XAttnHead
    torch.manual_seed(0)
    head = XAttnHead(D, UPPConfig(proj_dim=16)).eval()
    h_req, h_opts = torch.randn(7, D), torch.randn(5, D)
    perm = torch.tensor([3, 0, 4, 1, 2])
    with torch.no_grad():
        a, _ = head(h_req, h_opts)
        b, _ = head(h_req, h_opts[perm])
    assert torch.allclose(a[perm], b, atol=1e-5)
    for kw in ({}, {"gists": 2, "gist_budgets": (1, 2), "gist_distill": 1.0}):
        m = UPPModel(UPPConfig(layout="block", readout="xattn", **kw), use_mock=True)
        out = m.forward_batch([REC, REC])
        out["loss"].backward()
        assert m.head.out.weight.grad is not None
        m.eval()
        crit = REC["questions"]["tool_route"]["criteria"]
        p, _ = m.score(REC["state"], crit)
        q, _ = m.score(REC["state"], dict(reversed(list(crit.items()))))
        assert max(abs(p[k] - q[k]) for k in p) < 1e-5


def test_option_index_false_removes_bracket_prefix():
    with_idx = UPPModel(UPPConfig(layout="query_first", option_index=True), use_mock=True)
    no_idx = UPPModel(UPPConfig(layout="query_first", option_index=False), use_mock=True)
    crit = {"get_weather": "weather", "send_email": "email"}
    pk_with = with_idx.build("User Request: hi", crit)
    pk_no = no_idx.build("User Request: hi", crit)
    assert len(pk_no["opt_tokens"][0]) < len(pk_with["opt_tokens"][0])
    assert pk_no["q_idx"] == pk_with["q_idx"]
    # Identical tool text produces identical token lengths regardless of position index.
    pk_same = no_idx.build("User Request: hi", {"a": "same", "b": "same"})
    assert len(pk_same["opt_tokens"][0]) == len(pk_same["opt_tokens"][1])


def test_aligned_order_loss_penalizes_permutation_mismatch():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "veyra" / "training"))
    from p0_upp_ablation import _aligned_order_loss, _reverse_menu

    rev = _reverse_menu(REC)
    assert list(rev["questions"]["tool_route"]["criteria"]) == list(reversed(list(REC["questions"]["tool_route"]["criteria"])))

    keys_a = ["t1", "t2", "t3"]
    keys_b = ["t3", "t2", "t1"]
    lg_a = torch.tensor([2.0, 0.5, -1.0], requires_grad=True)
    lg_b_match = torch.tensor([-1.0, 0.5, 2.0], requires_grad=True)
    lg_b_flip = torch.tensor([2.0, 0.5, -1.0], requires_grad=True)
    loss_same = _aligned_order_loss([lg_a], [keys_a], [lg_b_match], [keys_b], logit_weight=0.15)
    loss_flip = _aligned_order_loss([lg_a], [keys_a], [lg_b_flip], [keys_b], logit_weight=0.15)
    assert float(loss_same) < 1e-6
    assert float(loss_flip) > 0.5
    loss_flip.backward()
    # Both permutation passes receive gradients (non-detached symmetric JSD + centered-logit MSE).
    assert lg_a.grad is not None and float(lg_a.grad.abs().sum()) > 0
    assert lg_b_flip.grad is not None and float(lg_b_flip.grad.abs().sum()) > 0


def test_query_first_option_isolation_exact_order_invariance_and_batch_score():
    import random

    torch.manual_seed(0)
    m = UPPModel(
        UPPConfig(layout="query_first", head="dot", option_index=False, option_isolation=True),
        use_mock=True,
    ).eval()
    crit = {
        "get_weather": "weather forecast for a city",
        "send_email": "send an email message to recipient",
        "search_files": "search local files by glob",
        "__abstain__": "none of the listed tools fit",
    }
    pk = m.build(REC["state"], crit)
    # All options must share the exact same start position right after q_idx, and mask out other options.
    starts = [pk["positions"][t[0]] for t in pk["opt_tokens"]]
    assert len(set(starts)) == 1 and starts[0] == pk["q_idx"] + 1
    allowed = m._query_iso_allowed(pk, len(pk["ids"]))
    assert allowed[pk["opt_tokens"][0][-1], : pk["q_idx"] + 1].all()
    assert not allowed[pk["opt_tokens"][0][-1], pk["opt_tokens"][1][0]]
    assert not allowed[pk["opt_tokens"][1][-1], pk["opt_tokens"][0][0]]

    # Exact permutation invariance across 5 random menu orders + reversed order.
    base_probs, _ = m.score(REC["state"], crit)
    keys = list(crit)
    rng = random.Random(123)
    for _ in range(5):
        perm = list(keys)
        rng.shuffle(perm)
        shuf_probs, _ = m.score(REC["state"], {k: crit[k] for k in perm})
        assert max(abs(base_probs[k] - shuf_probs[k]) for k in keys) < 1e-5

    # Batched scoring matches unbatched scoring even with different prompt lengths / right-padding.
    crit2 = {"only_tool": "short", "__abstain__": "none"}
    batched = m.batch_score([(REC["state"], crit, None), ("User Request: short ping", crit2, None)], batch_size=2)
    p2, _ = m.score("User Request: short ping", crit2)
    assert max(abs(batched[0][0][k] - base_probs[k]) for k in keys) < 1e-5
    assert max(abs(batched[1][0][k] - p2[k]) for k in crit2) < 1e-5

    # Gradient flows in training mode.
    m.train()
    out = m.forward_batch([REC, REC])
    out["loss"].backward()
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in m.trainable_parameters())


def test_from_checkpoint_and_eval_upp_full_roundtrip(tmp_path):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "veyra" / "training"))
    from eval_upp_full import _score_split_shard, summarize_by_split, wilson_ci

    torch.manual_seed(0)
    m = UPPModel(
        UPPConfig(layout="query_first", head="dot", option_index=False, option_isolation=True),
        use_mock=True,
    ).eval()
    ckpt = tmp_path / "ckpt_A"
    ckpt.mkdir()
    torch.save({"head": m.head.state_dict(), "cfg": m.config_dict()}, ckpt / "head.pt")

    torch.manual_seed(0)
    loaded = UPPModel.from_checkpoint(ckpt, use_mock=True).eval()
    loaded.backbone.load_state_dict(m.backbone.state_dict())
    assert loaded.cfg == m.cfg
    p_orig, _ = m.score(REC["state"], REC["questions"]["tool_route"]["criteria"])
    p_load, _ = loaded.score(REC["state"], REC["questions"]["tool_route"]["criteria"])
    assert max(abs(p_orig[k] - p_load[k]) for k in p_orig) < 1e-6

    lo, hi = wilson_ci(85, 100)
    assert 0.76 < lo < 0.85 < hi < 0.92

    row_col = dict(REC, id="r1", tags=["collision"])
    row_nm = dict(REC, id="r2", tags=["near_miss"], questions={"tool_route": {"criteria": REC["questions"]["tool_route"]["criteria"], "label": "__abstain__"}})
    scored = _score_split_shard(loaded, "zeroshot", [row_col, row_nm, REC, dict(REC, id="r4")], flip_orders=5, batch_size=2, seed=42, shard_idx=0, shard_total=1)
    assert len(scored) == 4 and all(r["flipped"] == 0 for r in scored)
    summary = summarize_by_split(scored)["zeroshot"]
    assert summary["flip_rate"] == 0.0 and summary["subsets"]["collision"]["n"] == 1 and summary["subsets"]["near_miss"]["n"] == 1
