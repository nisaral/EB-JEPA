"""CI smoke test for the heuristic (no-weights) SoroHarness path."""
from soro import SoroHarness

harness = SoroHarness()


@harness.tool(description="Check customer current plan and remaining data balance")
def get_user_plan(customer_id: str):
    return {"status": "active", "id": customer_id}


res = harness.route("What plan is user 123 on?")
print("Route result:", res.action, res.selected_tool, res.q_hat)
out = harness.execute("get_user_plan", {"user_id": "123"})
assert out.data["id"] == "123", "Semantic argument repair failed"
print("Heuristic harness smoke test passed")
