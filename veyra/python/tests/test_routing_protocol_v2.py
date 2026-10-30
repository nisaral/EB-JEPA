import pytest

from soro.routing import eval_v2
from soro.routing.eval_v2 import load_frozen, validate_frozen, agreement
from soro.routing.overlap import illegal_overlap
from soro.routing.protocol_v2 import PROTOCOL_ID


def test_v2_protocol_id():
    assert PROTOCOL_ID == "soro-routing-protocol-v2"


_needs_harvest = pytest.mark.skipif(
    not (eval_v2.DATA / "mcp_tools_harvest.json").exists(),
    reason="frozen v2 data (mcp_tools_harvest.json) is not committed to the repo",
)


@_needs_harvest
def test_v2_queries_have_zero_gold_name_overlap():
    _, queries, _ = load_frozen()
    gate = validate_frozen(queries)
    assert gate["illegal_overlap"] == []
    assert gate["cluster_gaps"] == []
    for q in queries:
        assert not illegal_overlap(q["gold"], q["text"])


@_needs_harvest
def test_v2_annotator_disagreement_is_nonzero():
    _, queries, gold_b = load_frozen()
    agr = agreement(queries, gold_b)
    assert agr["n"] == 80
    assert agr["disagreement_share"] > 0
    assert agr["raw_agreement"] < 1.0
