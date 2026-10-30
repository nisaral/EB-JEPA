"""Smoke test against the published Hugging Face weights.

Opt-in: set ``SORO_RUN_PRETRAINED=1``. Skips (with a reason) when the Hub
is unreachable or the repo is absent, so offline runs stay green. Uses the
normal Hugging Face cache (``HF_HOME``), which CI caches between runs.
"""
from __future__ import annotations

import os

import pytest

pytestmark = [pytest.mark.network, pytest.mark.slow]

MODEL_ID = os.environ.get("SORO_PRETRAINED_ID", "nisaralll/soro-kev-0.8b")
EXPECTED_Q_HAT = 0.9518


@pytest.fixture(scope="module")
def harness():
    if os.environ.get("SORO_RUN_PRETRAINED") != "1":
        pytest.skip("set SORO_RUN_PRETRAINED=1 to download and test the HF weights")
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(MODEL_ID)
    except Exception as exc:  # offline, rate-limited, repo missing, ...
        pytest.skip(f"cannot fetch {MODEL_ID} from the Hugging Face Hub: {type(exc).__name__}: {exc}")
    from soro import SoroHarness

    h = SoroHarness.from_pretrained(MODEL_ID, device="cpu")

    @h.tool(description="Check customer current plan and remaining data balance")
    def check_plan(customer_id: str):
        return {"status": "active", "id": customer_id}

    @h.tool(description="Get the current weather for a city")
    def get_weather(city: str):
        return {"city": city}

    return h


def test_checkpoint_q_hat(harness):
    assert harness._scorer.q_hat is not None
    assert abs(harness._scorer.q_hat - EXPECTED_Q_HAT) < 1e-3


def test_routes_and_is_deterministic(harness):
    r1 = harness.route("What plan is user 123 on?")
    r2 = harness.route("What plan is user 123 on?")
    assert r1.selected_tool == "check_plan"
    assert (r1.action, r1.selected_tool) == (r2.action, r2.selected_tool)


def test_router_api_on_real_weights(harness):
    import soro_boro

    Router = getattr(soro_boro, "Router", None)
    if Router is None:
        pytest.skip("soro_boro.Router not available")

    router = Router.from_pretrained(MODEL_ID, device="cpu")
    d = router.route(
        "weather in Paris?",
        tools=[
            {"name": "get_weather", "description": "Get the current weather for a city"},
            {"name": "check_plan", "description": "Check customer current plan"},
        ],
    )
    assert d.tool == "get_weather"
