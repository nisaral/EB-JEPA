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


def test_upp_s2_expected_cost_loss_and_shrink_preserves_action_cards():
    import random
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "veyra" / "training"))
    from p0_upp_ablation import _shrink_menu

    crit = {
        "send_email": "send an email",
        "t1": "tool 1", "t2": "tool 2", "t3": "tool 3", "t4": "tool 4",
        "__answer__": "answer directly",
        "__ask__": "ask clarifying question",
        "__escalate__": "escalate",
        "__abstain__": "abstain",
    }
    rec = {
        "id": "s2-1",
        "state": "User Request: send email now",
        "questions": {"tool_route": {"criteria": crit, "label": "__ask__"}},
        "meta": {"s2": {"costs": {k: (0.0 if k == "__ask__" else (1.5 if k.startswith(("send", "t")) else 0.6)) for k in crit}}},
    }
    shrunk = _shrink_menu(rec, k=6, rng=random.Random(0))
    scrit = shrunk["questions"]["tool_route"]["criteria"]
    assert {"__answer__", "__ask__", "__escalate__", "__abstain__"} <= set(scrit) and len(scrit) <= 6

    m = UPPModel(
        UPPConfig(layout="query_first", head="dot", option_index=False, option_isolation=True, cost_weight=0.5),
        use_mock=True,
    ).train()
    out = m.forward_batch([rec, rec])
    assert not out["skipped"] and float(out["loss"]) > 0
    out["loss"].backward()
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in m.trainable_parameters())


def test_isolated_options_with_stabilized_refiner_preserves_invariance_and_initial_scale():
    import random

    torch.manual_seed(0)
    m_base = UPPModel(
        UPPConfig(layout="query_first", head="dot", option_index=False, option_isolation=True, refiner=False),
        use_mock=True,
    ).eval()
    torch.manual_seed(0)
    m_ref = UPPModel(
        UPPConfig(layout="query_first", head="dot", option_index=False, option_isolation=True, refiner=True, refiner_layers=1),
        use_mock=True,
    ).eval()

    hq, hk = torch.randn(128), torch.randn(16, 128)
    with torch.no_grad():
        lg0, _ = m_base.head(hq, hk)
        lgr, _ = m_ref.head(hq, hk)
    # Stabilized residual refiner must NOT blow up initial logits (old refiner jumped to ~60).
    assert float(lgr.abs().max()) < max(20.0, float(lg0.abs().max()) * 1.5)

    crit = {
        "datetime_add_to_date": "add days or months to a calendar date",
        "calculate_date_difference": "compute days between two calendar dates",
        "send_email": "send an email message to a recipient",
        "__abstain__": "none of the listed tools fit",
    }
    p0, _ = m_ref.score(REC["state"], crit)
    keys = list(crit)
    rng = random.Random(19)
    for _ in range(5):
        perm = list(keys)
        rng.shuffle(perm)
        pk, _ = m_ref.score(REC["state"], {k: crit[k] for k in perm})
        assert max(abs(p0[k] - pk[k]) for k in keys) < 1e-5

    # Batched right-padded scoring must match unbatched scoring too.
    crit2 = {"only_tool": "short description", "__abstain__": "none"}
    batched = m_ref.batch_score([(REC["state"], crit, None), ("User Request: ping", crit2, None)], batch_size=2)
    assert max(abs(batched[0][0][k] - p0[k]) for k in keys) < 1e-5

    # Verify gradients reach refiner parameters cleanly.
    m_ref.train()
    out = m_ref.forward_batch([REC, REC])
    out["loss"].backward()
    mem = m_ref.head.members[0]
    assert mem.ref_gate.grad is not None and mem.type_emb.grad is not None
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in mem.refiner.parameters())


def test_compare_upp_eval_mcnemar_and_exact_pairing():
    import sys
    from pathlib import Path

    tr_dir = Path(__file__).resolve().parents[1] / "veyra" / "training"
    if str(tr_dir) not in sys.path:
        sys.path.insert(0, str(tr_dir))
    from compare_upp_eval import compare_pair, mcnemar_exact

    # 12 discordant pairs: 10 A-only, 2 B-only -> exact 2-sided p = 2*(C(12,0)+C(12,1)+C(12,2))/4096 = 158/4096
    mc = mcnemar_exact(10, 2)
    assert abs(mc["p_exact"] - (158.0 / 4096.0)) < 1e-12
    assert mc["discordant"] == 12

    rows_a = [{"split": "collisions", "orig_idx": i, "id": f"c{i}", "ok": int(i < 15), "flipped": int(i == 0)} for i in range(20)]
    rows_b = [{"split": "collisions", "orig_idx": i, "id": f"c{i}", "ok": int(i < 5 or i >= 15), "flipped": 0} for i in range(20)]
    stats = compare_pair(rows_a, rows_b)
    col = stats["collisions"]
    assert col["n"] == 20 and col["n10_a_only"] == 10 and col["n01_b_only"] == 5 and abs(col["delta_pts"] - (-25.0)) < 1e-9
    assert col["flips_a"] == 1 and col["flips_b"] == 0


def test_s2_teacher_kl_init_ckpt_and_eval_ask_or_exact(tmp_path):
    import sys
    from argparse import Namespace
    from pathlib import Path

    tr_dir = Path(__file__).resolve().parents[1] / "veyra" / "training"
    if str(tr_dir) not in sys.path:
        sys.path.insert(0, str(tr_dir))
    from eval_upp_full import _score_split_shard, summarize_by_split
    from p0_upp_ablation import train_one

    # 1. teacher_kl_weight computes soft KL when teacher.probs present and cleanly skips when absent.
    torch.manual_seed(0)
    m = UPPModel(
        UPPConfig(layout="query_first", head="dot", option_index=False, option_isolation=True,
                  refiner=True, cost_weight=0.5, teacher_kl_weight=0.25),
        use_mock=True,
    ).train()
    crit = {"get_weather": "weather", "send_email": "email", "__ask__": "ask", "__abstain__": "none"}
    rec_teacher = {
        "id": "t1",
        "state": "User Request: weather in Pune",
        "questions": {"tool_route": {"criteria": crit, "label": "get_weather"}},
        "meta": {"s2": {"costs": {"get_weather": 0.0, "send_email": 1.0, "__ask__": 0.45, "__abstain__": 0.8}}},
        "teacher": {"model": "mock", "probs": {"get_weather": 0.8, "__ask__": 0.15, "send_email": 0.03, "__abstain__": 0.02}},
    }
    rec_plain = {
        "id": "t2",
        "state": "User Request: unclear request",
        "questions": {"tool_route": {"criteria": crit, "label": "__ask__"}},
    }
    out = m.forward_batch([rec_teacher, rec_plain])
    assert not out["skipped"] and float(out["loss"]) > 0
    out["loss"].backward()
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in m.trainable_parameters())

    # 2. --init-ckpt reloads checkpoint config/weights, applies S2 cost & teacher_kl, and saves ckpt_A.
    ckpt_in = tmp_path / "s1_ckpt"
    ckpt_in.mkdir()
    torch.save({"head": m.head.state_dict(), "cfg": m.config_dict()}, ckpt_in / "head.pt")

    out_dir = tmp_path / "s2_run"
    args = Namespace(
        base_model="mock", smoke=True, load_in_4bit=False, no_grad_ckpt=True,
        init_ckpt=str(ckpt_in), lr=1e-3, max_steps=2, batch=2, max_seq_len=512,
        seed=42, log_every=10, s1=False, save_ckpt=True, out=str(out_dir),
    )
    cfg = UPPConfig(layout="query_first", head="dot", option_index=False, option_isolation=True,
                    refiner=True, cost_weight=0.5, teacher_kl_weight=0.25)
    res = train_one("A", cfg, args, [rec_teacher, rec_plain], {"sel": [rec_teacher, rec_plain]})
    assert (out_dir / "ckpt_A" / "head.pt").exists()
    assert "sel" in res["eval"]

    # 3. eval_upp_full tracks ask_or_exact and action:* breakdowns.
    m.eval()
    rows = _score_split_shard(m, "when2call", [rec_teacher, rec_plain], flip_orders=3, batch_size=2, seed=7, shard_idx=0, shard_total=1)
    summ = summarize_by_split(rows)["when2call"]
    assert "ask_or_exact" in summ and summ["ask_or_exact"] >= summ["acc"]
    assert "action:tool" in summ["subsets"] and "action:ask" in summ["subsets"]

    # 4. label_s2_teacher shard merge & validation works end-to-end.
    from label_s2_teacher import attach_teacher, merge_and_validate
    from soro.omni_data.schema import OmniRow, Option, read_rows, write_rows

    r1 = OmniRow(
        id="lbl-1", task="tool_route", modality="text",
        options=[Option("a", "tool a"), Option("__ask__", "ask", kind="special"), Option("__abstain__", "none", kind="special")],
        label="a", source="soro_train", license="own", group="g1", text="run tool a on foo",
    )
    r2 = OmniRow(
        id="lbl-2", task="tool_route", modality="text",
        options=[Option("b", "tool b"), Option("__ask__", "ask", kind="special"), Option("__abstain__", "none", kind="special")],
        label="__ask__", source="when2call", license="cc-by-4.0", group="g2", text="run tool b",
    )
    s0 = attach_teacher(r1, {"a": 0.9, "__ask__": 0.05, "__abstain__": 0.05}, "mock-teacher", rotations=4)
    s1 = attach_teacher(r2, {"b": 0.9, "__ask__": 0.05, "__abstain__": 0.05}, "mock-teacher", rotations=4)
    p0_path, p1_path = tmp_path / "sh0.jsonl", tmp_path / "sh1.jsonl"
    write_rows([s0], p0_path)
    write_rows([s1], p1_path)
    merged_path = tmp_path / "merged_teacher.jsonl"
    rep = merge_and_validate([p0_path, p1_path], merged_path)
    assert rep["total"] == 2 and rep["teacher_coverage"] == 1.0
    assert rep["label_audit"]["flagged"] == 1 and rep["label_audit"]["checked"] == 2
    assert len(list(read_rows(merged_path))) == 2

    filtered_path = tmp_path / "merged_filtered.jsonl"
    frep = merge_and_validate([p0_path, p1_path], filtered_path, filter_mode="exact", strip_probs=True)
    frows = list(read_rows(filtered_path))
    assert frep["kept"] == 1 and frep["dropped"] == 1
    assert [r.id for r in frows] == ["lbl-1"] and "probs" not in (frows[0].teacher or {})


def test_eval_conformal_action_g2_g3():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "veyra" / "training"))
    from eval_conformal_action import build_report, format_markdown, k_bucket, lac_quantile, set_as_action_predict

    assert k_bucket(4) == "k_le_5" and k_bucket(6) == "k_6_20" and k_bucket(20) == "k_6_20" and k_bucket(25) == "k_gt_20"
    assert abs(lac_quantile([0.1, 0.2, 0.3, 0.4, 0.5], 0.2) - 0.5) < 1e-9

    # Set-as-action: |S_tool|=0 -> abstain, |S_tool|=1 -> act, |S_tool|>=2 -> ask
    p_abs = {"t1": 0.25, "t2": 0.20, "__ask__": 0.15, "__abstain__": 0.35, "__answer__": 0.05}
    p_act = {"t1": 0.85, "t2": 0.05, "__ask__": 0.05, "__abstain__": 0.03, "__answer__": 0.02}
    p_ask = {"t1": 0.45, "t2": 0.42, "__ask__": 0.08, "__abstain__": 0.03, "__answer__": 0.02}
    d_abs, s_abs = set_as_action_predict(p_abs, q_tool=0.65)  # needs p >= 0.35
    d_act, s_act = set_as_action_predict(p_act, q_tool=0.65)
    d_ask, s_ask = set_as_action_predict(p_ask, q_tool=0.65)
    assert d_abs == "__abstain__" and s_abs == []
    assert d_act == "t1" and s_act == ["t1"]
    assert d_ask == "__ask__" and s_ask == ["t1", "t2"]

    def _mk(split: str, idx: int, probs: dict[str, float], gold: str) -> dict:
        top = max(probs, key=probs.get)
        return {
            "split": split,
            "id": f"{split}-{idx}",
            "gold": gold,
            "gold_kind": "abstain" if gold == "__abstain__" else ("ask" if gold == "__ask__" else ("answer" if gold == "__answer__" else "tool")),
            "top": top,
            "ok": int(top == gold),
            "flipped": 0,
            "probs": probs,
            "u": 0.1,
        }

    # Build 7-option (K in [6,20]) calibration + held-out eval rows
    def _k7_probs(vals: dict[str, float]) -> dict[str, float]:
        base = {f"t{i}": 0.01 for i in range(1, 5)}
        base.update({"__ask__": 0.01, "__abstain__": 0.01, "__answer__": 0.01})
        base.update(vals)
        s = sum(base.values())
        return {k: v / s for k, v in base.items()}

    rows = []
    for i in range(30):
        rows.append(_mk("s2_selection", i, _k7_probs({"t1": 0.85}), "t1"))
        rows.append(_mk("s2_selection", 100 + i, _k7_probs({"t1": 0.42, "t2": 0.40, "__ask__": 0.12}), "__ask__"))
        rows.append(_mk("s2_selection", 200 + i, _k7_probs({"t1": 0.15, "__abstain__": 0.70}), "__abstain__"))
    for i in range(24):
        rows.append(_mk("when2call_mcq", i, _k7_probs({"t1": 0.85}), "t1"))
        rows.append(_mk("when2call_mcq", 100 + i, _k7_probs({"t1": 0.42, "t2": 0.40, "__ask__": 0.12}), "__ask__"))
        rows.append(_mk("when2call_mcq", 200 + i, _k7_probs({"t1": 0.15, "__abstain__": 0.70}), "__abstain__"))

    rep = build_report(rows, cal_split="s2_selection")
    assert "k_6_20" in rep["g2_k_buckets"]["when2call_mcq"]
    assert rep["g2_conformal"]["by_alpha"]["0.10"]["splits"]["when2call_mcq"]["transfer_lac"]["coverage"] >= 0.85
    assert rep["g3_set_as_action"]["tuned_tool_set"]["splits"]["when2call_mcq"]["acc"] >= 0.9
    md = format_markdown(rep)
    assert "G2 & G3 Evaluation" in md and "ALL_EVAL_K_6_20" in md
