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
    assert max(pk["opt_ends"]) < pk["catalog_len"] <= pk["q_idx"] == len(pk["ids"]) - 1
    blk = UPPModel(UPPConfig(layout="block"), use_mock=True).build("User Request: hi", crit)
    starts = [blk["positions"][blk["blocks"].index(i)] for i in range(3)]
    assert len(set(starts)) == 1  # every option block starts at the same position
    qf = UPPModel(UPPConfig(layout="query_first"), use_mock=True)
    pk2 = qf.build("User Request: hi", crit)
    assert pk2["q_idx"] < min(pk2["opt_ends"]) and pk2["catalog_len"] == 0


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
