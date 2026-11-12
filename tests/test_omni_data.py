"""Tests for the Soro-Kev-Omni data pipeline (no models, no downloads)."""
from __future__ import annotations

import json

import numpy as np
import pytest
from soro.omni_data import audio, build, image, licenses, quality
from soro.omni_data.generate import agree_filter, necessity_filter, parse_request
from soro.omni_data.schema import ABSTAIN, NEED_INPUT, OmniRow, Option, from_soro_record, read_rows, special_option, write_rows
from soro.omni_data.teacher import fit_temperature, soft_target
from soro.omni_data.text import ToolPool, filter_near_misses, make_near_misses, mine_collisions


def _row(i, text, label="get_weather", tools=("get_weather", "send_email", "list_files"), pkg="p1", **kw):
    opts = [Option(k, f"{k.replace('_', ' ')} tool") for k in tools] + [special_option(ABSTAIN)]
    return OmniRow(id=str(i), task="tool_route", modality="text", options=opts, label=label, source="soro_train",
                   license="own", group=f"g{i}", text=text, package=pkg, **kw)


def test_schema_roundtrip_and_validation(tmp_path):
    r = _row(1, "will it rain tomorrow")
    assert r.validate() == []
    write_rows([r], tmp_path / "a.jsonl")
    back = next(read_rows(tmp_path / "a.jsonl"))
    assert back == r and back.to_upp_record()["questions"]["tool_route"]["label"] == "get_weather"
    bad = _row(2, "x", label="nope")
    assert any("label" in e for e in bad.validate())
    with pytest.raises(ValueError):
        Option("el0", "button", kind="ui_element", bbox=[0.5, 0.5, 0.1, 0.9])


def test_from_soro_record_splits_context_and_request():
    rec = {"id": "x", "package": "pk", "state": "Agent: hi\nUser Request: book a cab",
           "questions": {"tool_route": {"criteria": {"book_cab": "Book a cab", ABSTAIN: "none"}, "label": "book_cab"}}}
    r = from_soro_record(rec, "soro_train", "own")
    assert r.text == "book a cab" and r.context == "Agent: hi" and r.validate() == []


def test_license_gate_moves_noncommercial_rows_out_of_train():
    rows = [_row(1, "a"), _row(2, "b")]
    rows[1].license = "cc-by-nc-4.0"
    kept, moved = licenses.gate(rows)
    assert moved == 1 and kept[1].split == "test" and kept[0].split == "train"
    assert licenses.REGISTRY["audio2tool"].use == "eval"


def test_dedupe_exact_and_near_duplicates():
    rows = [_row(1, "what is the weather in pune tomorrow morning please"),
            _row(2, "what is the weather in pune tomorrow morning please"),       # exact (same menu/label)
            _row(3, "what is the weather in pune tomorrow morning please now"),   # near
            _row(4, "send an email to my manager about the report")]
    rows[3].label = "send_email"
    kept, stats = quality.dedupe(rows, near_threshold=0.7)
    assert stats["exact_dupes"] == 1 and stats["near_dupes"] == 1 and len(kept) == 2


def test_dedupe_keeps_audio_twin_of_the_same_request():
    text = _row(1, "will it rain tomorrow")
    spoken = _row(2, "")
    spoken.modality = "audio"
    spoken.transcript = "will it rain tomorrow"
    spoken.audio = "a.wav"
    spoken.group = text.group
    kept, stats = quality.dedupe([text, spoken], near_threshold=0.7)
    assert len(kept) == 2 and stats.get("near_dupes", 0) == 0


def test_leakage_finds_shared_queries_and_packages():
    train = [_row(1, "list every file in my downloads folder", pkg="files")]
    ev = [_row(9, "list every file in my downloads folder", pkg="files"), _row(10, "totally different ask", pkg="x")]
    rep = quality.leakage(train, {"test": ev})["test"]
    assert rep["exact_query_overlap"] == 1 and rep["n_shared_packages"] == 1


def test_collision_mining_keeps_label_and_menu_size():
    rows = [_row(i, f"request {i}", tools=("get_weather", f"tool_{i}", f"other_{i}")) for i in range(5)]
    rows.append(_row(99, "x", tools=("get_weather_forecast", "get_weather_alerts", "zzz")))
    pool = ToolPool(rows)
    out = mine_collisions(rows[:5], pool, per_row=2, rate=1.0)
    assert out
    for r in out:
        assert r.label == "get_weather" and len(r.options) == 4 and "collision" in r.tags
        assert {"get_weather_forecast", "get_weather_alerts"} & {o.key for o in r.options}
        assert r.validate() == []


def test_near_miss_requires_teacher_verification():
    rows = [_row(1, "weather please"), _row(2, "y", tools=("get_weather_forecast", "a_b", "c_d"))]
    nm = make_near_misses(rows[:1], ToolPool(rows), rate=1.0)
    assert len(nm) == 1 and nm[0].label == ABSTAIN and "get_weather" not in {o.key for o in nm[0].options}
    kept, dropped = filter_near_misses(nm)
    assert kept == [] and dropped == 1
    nm[0].teacher = {"probs": {o.key: (0.9 if o.key == ABSTAIN else 0.1 / 3) for o in nm[0].options}}
    kept, dropped = filter_near_misses(nm)
    assert len(kept) == 1 and "teacher_verified" in kept[0].tags


def test_teacher_temperature_and_soft_targets():
    rows = []
    for i in range(20):
        r = _row(i, f"q{i}")
        r.teacher = {"probs": {"get_weather": 0.97, "send_email": 0.01, "list_files": 0.01, ABSTAIN: 0.01}}
        if i % 4 == 0:
            r.label = "send_email"
        rows.append(r)
    t = fit_temperature(rows)
    assert t > 1.0  # overconfident teacher gets softened
    st = soft_target(rows[1], lam=0.5, temperature=t)
    assert abs(sum(st.values()) - 1) < 1e-6 and st["get_weather"] > 0.5


def test_generation_parsing_and_filters():
    assert parse_request('Sure: "Can you check if it will rain in Pune?"') == "Can you check if it will rain in Pune?"
    assert parse_request("ok") is None
    assert necessity_filter({"a": 0.4, "b": 0.35, ABSTAIN: 0.25}, {"a": 0.9, "b": 0.05, ABSTAIN: 0.05}, "a")
    assert not necessity_filter({"a": 0.9, "b": 0.1}, {"a": 0.95, "b": 0.05}, "a")  # text alone was enough
    r = _row(1, "q", tags=["needs_teacher"])
    r.teacher = {"probs": {"get_weather": 0.1, "send_email": 0.8, "list_files": 0.05, ABSTAIN: 0.05}}
    assert agree_filter([r]) == ([], 1)


def test_audio_wav_roundtrip_and_augment():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 1, audio.SR, dtype=np.float32)
    x = 0.5 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
    noisy = audio.add_noise(x, 10.0, rng)
    snr = 20 * np.log10(audio.rms(x) / audio.rms(noisy - x))
    assert abs(snr - 10.0) < 0.5
    assert len(audio.speed(x, 1.1)) == round(len(x) / 1.1)
    y, rec = audio.augment(x, rng)
    assert y.dtype == np.float32 and np.abs(y).max() <= 1.0 and "gain" in rec
    assert [f for f, _ in audio.prefixes(x)] == [0.4, 0.6, 0.8]


def test_audio_wav_io(tmp_path):
    x = np.linspace(-0.5, 0.5, 1600, dtype=np.float32)
    audio.write_wav(tmp_path / "a.wav", x)
    y, sr = audio.read_wav(tmp_path / "a.wav")
    assert sr == audio.SR and np.abs(x - y).max() < 1e-3


def test_render_rows_with_fake_tts(tmp_path):
    fake = lambda text, voice: np.sin(np.linspace(0, 50, 8000)).astype(np.float32) * 0.3  # noqa: E731
    rows = audio.render_rows([_row(1, "turn on the lights")], tmp_path, fake, ["af_heart"], streaming_rate=1.0)
    assert rows[0].modality == "audio" and rows[0].transcript == "turn on the lights" and rows[0].text == ""
    assert any("streaming" in r.tags for r in rows) and all(r.validate() == [] for r in rows)


AUDIO2TOOL_SAMPLE = {
    "id": 1, "tier": "tier1_direct", "query_idx": 2100, "query": "Be a dear and turn on the lights",
    "domain": "smart_home", "category": "lighting", "tool_id": 70, "tool_name": "setLighting",
    "functions": [{"tool_id": 70, "domain": "smart_home", "category": "lighting", "tool_name": "setLightState",
                   "signature": "setLightState(deviceId: String, state: OnOff)",
                   "description": "Turns a lighting device on or off."}],
    "path": "tier1_direct_audios/query_00001/a.wav", "duration": 2.7, "additional_tool_calls": None,
}


def test_audio2tool_matches_gold_by_tool_id_and_is_eval_only():
    other = json.loads(json.dumps(AUDIO2TOOL_SAMPLE))
    other.update(id=2, query_idx=5, tool_id=71, functions=[{**other["functions"][0], "tool_id": 71,
                                                            "tool_name": "setThermostat",
                                                            "signature": "setThermostat(t: Int)",
                                                            "description": "Sets temperature."}])
    rows = audio.audio2tool_rows([AUDIO2TOOL_SAMPLE, other], "/data/a2t")
    r = rows[0]
    assert r.label == "setLightState" and r.split == "test" and r.license == "cc-by-nc-4.0"
    assert "setThermostat" in {o.key for o in r.options} and r.validate() == []


def test_os_atlas_rows_build_menus_absent_and_twins():
    recs = []
    for img in ("s1.png", "s2.png"):
        for j, name in enumerate(["Settings", "Search", "Profile", "Settings"]):
            recs.append({"img_filename": img, "instruction": name if img == "s1.png" else f"{name} {img}",
                         "bbox": [0.1 * j, 0.1, 0.1 * j + 0.05, 0.2]})
    rows = image.os_atlas_rows(recs, "/imgs", "test", "apache-2.0", absent_rate=0.0, seed=1)
    assert rows and all(r.validate() == [] for r in rows)
    assert any("twin" in r.tags for r in rows)
    assert all(r.task == "ui_point" and r.image.endswith(".png") for r in rows)
    absent = image.os_atlas_rows(recs, "/imgs", "test", "apache-2.0", absent_rate=1.0, seed=1)
    assert any(r.label == ABSTAIN and "absent" in r.tags for r in absent)


def test_build_end_to_end(tmp_path):
    train = [_row(i, f"please check weather number {i} in city {i * 7}") for i in range(40)]
    train += [_row(100 + i, f"nothing fits here {i} at all {i * 3}", label=ABSTAIN) for i in range(10)]
    unverified = _row(500, "generated thing", tags=["needs_teacher"])
    ev = [_row(900, "please check weather number 3 in city 21")]  # leaks from train
    ev[0].split = "test"
    write_rows(train + [unverified], tmp_path / "train.jsonl")
    write_rows(ev, tmp_path / "eval.jsonl")
    cfg = {"inputs": [{"path": str(tmp_path / "train.jsonl")}, {"path": str(tmp_path / "eval.jsonl"), "split": "test"}],
           "targets": {"text/positive": 30, "text/abstain": 10}, "selection_frac": 0.2}
    rep = build.build(cfg, tmp_path / "out")
    assert rep["dropped_unverified"] == 1
    assert rep["eval_rows_dropped_for_leakage"] == 1
    assert rep["mix"]["text/positive"]["taken"] == 30
    tr = {r.group for r in read_rows(tmp_path / "out" / "train.jsonl")}
    se = {r.group for r in read_rows(tmp_path / "out" / "selection.jsonl")}
    assert tr and se and not (tr & se)


def test_lookalikes_skip_same_tool_spellings():
    rows = [_row(1, "a", tools=("send_email", "x_y", "z_w")),
            _row(2, "b", tools=("send-email", "send_sms", "q_r"))]
    pool = ToolPool(rows)
    assert pool.same_tool("send_email", "send-email")
    looks = pool.lookalikes("send_email", k=5, min_sim=0.0)
    assert "send-email" not in looks and "send_sms" in looks


def test_generic_requests_are_tagged():
    rows = [_row(i, "book a table for me and my friend", pkg=f"pkg{i}") for i in range(4)]
    rows.append(_row(9, "unique ask", pkg="pkg9"))
    rep = quality.generic_requests(rows, min_packages=3)
    assert rep["generic_requests"] == 1 and rep["rows_tagged"] == 4
    assert "generic_request" not in rows[-1].tags


def test_build_keeps_verified_copy_of_unverified_row(tmp_path):
    raw = _row(1, "book a table for me and my friend", tags=["generic_request", "needs_teacher"])
    ver = OmniRow.from_json(raw.to_json())
    ver.tags = ["generic_request", "teacher_verified"]
    others = [_row(10 + i, f"distinct request number {i} about weather {i * 13}") for i in range(10)]
    write_rows([raw] + others, tmp_path / "base.jsonl")
    write_rows([ver], tmp_path / "ver.jsonl")
    cfg = {"inputs": [{"path": str(tmp_path / "base.jsonl")}, {"path": str(tmp_path / "ver.jsonl")}],
           "targets": {}, "selection_frac": 0.0}
    build.build(cfg, tmp_path / "out")
    kept = [r for r in read_rows(tmp_path / "out" / "train.jsonl") if r.id == "1"]
    assert len(kept) == 1 and "teacher_verified" in kept[0].tags


def test_dedupe_does_not_cap_distinct_ui_screens_on_generic_el_keys():
    rows = []
    for i in range(5):
        opts = [Option("el0", "top-left"), Option("el1", "bottom-right"), special_option(ABSTAIN)]
        rows.append(OmniRow(
            id=f"ui-{i}", task="ui_point", modality="image", options=opts, label="el0",
            source="os_atlas", license="apache-2.0", group=f"osa-screen-{i}",
            text=f"Tap item {i}", image=f"/imgs/screen_{i}.png", package="os_atlas/aw_mobile",
        ))
    kept, stats = quality.dedupe(rows, max_per_group_label=2)
    assert len(kept) == 5 and stats.get("capped", 0) == 0


def test_s2_wrap_and_asymmetric_costs():
    from soro.omni_data.s2_actions import action_cost, wrap_s2_row
    from soro.omni_data.schema import ACTION_CARDS

    r = _row(1, "delete the backup file", label="delete_backup", tools=("delete_backup", "send_email"))
    s2 = wrap_s2_row(r, shuffle_seed=0)
    keys = {o.key for o in s2.options}
    assert set(ACTION_CARDS) <= keys and s2.validate() == []
    costs = s2.meta["s2"]["costs"]
    assert set(costs) == keys and costs["delete_backup"] == 0.0
    # False/unsafe tool execution costs more than cautious deferral (ask/abstain/escalate).
    assert action_cost("__ask__", "delete_backup") > action_cost("delete_backup", "__ask__")
    assert action_cost("__escalate__", "delete_backup", risk_tier="high") == 2.0


def test_s2_when2call_and_agentabstain_converters():
    from soro.omni_data.s2_actions import (
        from_agentabstain_rows,
        from_when2call_mcq,
        from_when2call_pref,
        parse_agentabstain_env_source,
    )
    from soro.omni_data.schema import ABSTAIN, ANSWER, ASK, ESCALATE

    tools = ['{"name": "get_weather", "description": "Get city weather", "parameters": {"type": "dict", "required": ["city"], "properties": {"city": {"type": "string"}}}}']
    pref_recs = [
        {"tools": tools, "messages": [{"role": "user", "content": "Weather in Pune?"}],
         "chosen_response": {"role": "assistant", "content": '<TOOLCALL>[{"name": "get_weather", "arguments": {"city": "Pune"}}]</TOOLCALL>'}},
        {"tools": tools, "messages": [{"role": "user", "content": "Weather?"}],
         "chosen_response": {"role": "assistant", "content": "Which city would you like the weather for?"}},
        {"tools": tools, "messages": [{"role": "user", "content": "Book a flight to Paris"}],
         "chosen_response": {"role": "assistant", "content": "Apologies, I am unable to book flights."}},
        {"tools": tools, "messages": [{"role": "user", "content": "What is 2+2?"}],
         "chosen_response": {"role": "assistant", "content": "Four."}},
    ]
    pref_rows = from_when2call_pref(pref_recs)
    assert [r.label for r in pref_rows] == ["get_weather", ASK, ABSTAIN, ANSWER]
    assert all(r.validate() == [] for r in pref_rows)

    mcq_recs = [
        {"uuid": "1", "question": "q1", "correct_answer": "tool_call", "target_tool": tools[0], "tools": tools, "source": "live"},
        {"uuid": "2", "question": "q2", "correct_answer": "request_for_info", "target_tool": None, "tools": tools, "source": "live"},
        {"uuid": "3", "question": "q3", "correct_answer": "cannot_answer", "target_tool": None, "tools": tools, "source": "live"},
    ]
    mcq_rows = from_when2call_mcq(mcq_recs)
    assert [r.label for r in mcq_rows] == ["get_weather", ASK, ABSTAIN] and all(r.split == "test" for r in mcq_rows)

    env_src = '''
class DemoEnv:
    def _register_tools(self):
        @self.mcp.tool()
        def send_email(recipient: str, body: str) -> dict:
            """Send an email to recipient."""
            return {}
        @self.mcp.tool()
        def read_inbox() -> dict:
            """Read messages in inbox."""
            return {}
'''
    parsed = parse_agentabstain_env_source(env_src)
    assert set(parsed) == {"send_email", "read_inbox"}
    aa_tasks = [
        {"pair_id": "missing_critical_parameter/1", "category": "missing_critical_parameter", "task_type": "act",
         "phase": "pre_execution", "action_type": "operational", "instruction": "Email alex@x.com hi",
         "system_prompt": "sys", "execution_dag": {"nodes": [{"tool": "demo.send_email"}]}, "environments": ["demo"]},
        {"pair_id": "missing_critical_parameter/1", "category": "missing_critical_parameter", "task_type": "abstain",
         "phase": "pre_execution", "action_type": "operational", "instruction": "Email hi",
         "system_prompt": "sys", "execution_dag": {"nodes": []}, "environments": ["demo"]},
        {"pair_id": "insufficient_tool_capability/2", "category": "insufficient_tool_capability", "task_type": "act",
         "phase": "pre_execution", "action_type": "operational", "instruction": "Send an email",
         "system_prompt": "sys", "execution_dag": {"nodes": [{"tool": "demo.send_email"}]}, "environments": ["demo"]},
        {"pair_id": "insufficient_tool_capability/2", "category": "insufficient_tool_capability", "task_type": "abstain",
         "phase": "pre_execution", "action_type": "operational", "instruction": "Send an email",
         "system_prompt": "sys", "execution_dag": {"nodes": []}, "environments": ["demo"]},
        {"pair_id": "high_stakes_action/3", "category": "high_stakes_action", "task_type": "abstain",
         "phase": "pre_execution", "action_type": "operational", "instruction": "Delete all emails",
         "system_prompt": "sys", "execution_dag": {"nodes": []}, "environments": ["demo"]},
        {"pair_id": "critical_tool_failure/4", "category": "critical_tool_failure", "task_type": "abstain",
         "phase": "runtime", "action_type": "informational", "instruction": "Read inbox",
         "system_prompt": "sys", "execution_dag": {"nodes": []}, "environments": ["demo"]},
    ]
    aa_rows = from_agentabstain_rows(aa_tasks, {"demo": parsed}, include_runtime=False)
    assert len(aa_rows) == 5
    by_id = {r.id: r for r in aa_rows}
    assert by_id["aa-missing_critical_parameter_1-act"].label == "send_email"
    assert by_id["aa-missing_critical_parameter_1-abstain"].label == ASK
    insuff = by_id["aa-insufficient_tool_capability_2-abstain"]
    assert insuff.label == ABSTAIN and "send_email" not in {o.key for o in insuff.options}
    assert by_id["aa-high_stakes_action_3-abstain"].label == ESCALATE


def test_s2_deterministic_builders_conformal_and_stage_smoke(tmp_path):
    import sys
    from pathlib import Path
    import torch
    from soro.omni_data.s2_actions import (
        action_group_conformal,
        build_answer_from_soro,
        build_ask_from_soro,
        build_escalate_from_soro,
        expected_cost_loss,
    )
    from soro.omni_data.schema import ANSWER, ASK, ESCALATE

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "veyra" / "omni_data"))
    import stage_s2

    base = [
        _row(i, f"Send an email report to user{i}@example.com right now", label="delete_old_logs",
             tools=("delete_old_logs", "send_email", "get_weather"))
        for i in range(10)
    ]
    asks = build_ask_from_soro(base, limit=5, seed=1)
    escs = build_escalate_from_soro(base, limit=5, seed=2)
    anss = build_answer_from_soro(base, limit=5, seed=3)
    assert asks and all(r.label == ASK and r.validate() == [] for r in asks)
    assert escs and all(r.label == ESCALATE and r.validate() == [] for r in escs)
    assert anss and all(r.label == ANSWER and r.validate() == [] for r in anss)

    logits = torch.tensor([10.0, 0.0, 0.0], requires_grad=True)
    cost = expected_cost_loss(logits, [0.0, 1.5, 0.45])
    assert float(cost) < 1e-3
    cost.backward()
    assert logits.grad is not None

    items = [({"get_weather": 0.9, ASK: 0.05, ESCALATE: 0.02, ANSWER: 0.02, ABSTAIN: 0.01}, "get_weather") for _ in range(10)]
    items += [({"get_weather": 0.05, ASK: 0.85, ESCALATE: 0.05, ANSWER: 0.03, ABSTAIN: 0.02}, ASK) for _ in range(10)]
    conf = action_group_conformal(items, alpha=0.1)
    assert conf["overall"]["coverage"] >= 0.9 and {"tool", "ask"} <= set(conf["by_action"])

    rep = stage_s2._smoke_build(tmp_path)
    assert set(rep["train_classes"]) == {"tool", "abstain", "ask", "escalate", "answer"}


def test_s2_clean_filters_schema_star_and_done_target():
    from soro.omni_data.s2_actions import (
        _format_tool_schema,
        build_abstain_cross_domain,
        build_ambiguous_ask,
        from_when2call_pref,
        wrap_s2_row,
    )
    from soro.omni_data.schema import ANSWER, ASK, ESCALATE

    # Missing 'required' list falls back to 'optional' in type string (When2Call pref/sft format).
    tool_no_req = {
        "name": "send_msg",
        "description": "Send message",
        "parameters": {
            "properties": {
                "recipient": {"type": "string"},
                "cc": {"type": "string, optional"},
            }
        },
    }
    sig = _format_tool_schema(tool_no_req)
    assert "recipient*:string" in sig and "cc:string, optional" in sig and "cc*" not in sig

    t_json = json.dumps(tool_no_req)
    pref_clean_recs = [
        # Real missing required param ASK (rejected tool uses 'recipient' not present in query)
        {
            "tools": [t_json],
            "messages": [{"role": "user", "content": "Send a message right now"}],
            "chosen_response": {"role": "assistant", "content": "Who should I send the message to?"},
            "rejected_response": {"role": "assistant", "content": '<TOOLCALL>[{"name": "send_msg", "arguments": {"recipient": "Alice"}}]</TOOLCALL>'},
        },
        # False pagination ASK -> dropped under clean_s2=True
        {
            "tools": [t_json],
            "messages": [{"role": "user", "content": "List messages"}],
            "chosen_response": {"role": "assistant", "content": "Which page number would you like?"},
            "rejected_response": {"role": "assistant", "content": '<TOOLCALL>[{"name": "send_msg", "arguments": {"recipient": "Bob"}}]</TOOLCALL>'},
        },
        # Pure LLM ABSTAIN -> dropped under clean_s2=True
        {
            "tools": [t_json],
            "messages": [{"role": "user", "content": "Given [3, 1, 4, 2], find the largest 2 numbers."}],
            "chosen_response": {"role": "assistant", "content": "Sorry, I cannot solve this."},
            "rejected_response": {"role": "assistant", "content": '<TOOLCALL>[{"name": "missing_algo", "arguments": {}}]</TOOLCALL>'},
        },
    ]
    cleaned = from_when2call_pref(pref_clean_recs, clean_s2=True)
    assert len(cleaned) == 1 and cleaned[0].label == ASK

    # Ambiguous multi-tool ASK & cross-domain ABSTAIN builders
    r_amb = _row(
        10,
        "Manage the invoice database please now",
        label="create_invoice",
        tools=("create_invoice", "delete_invoice", "get_weather"),
        pkg="fin",
    )
    amb = build_ambiguous_ask([r_amb], limit=5, seed=0)
    assert len(amb) == 1 and amb[0].label == ASK and amb[0].validate() == []

    r_xab = _row(
        11,
        "Check the flight departure status in Mumbai today",
        label="get_flight_status",
        tools=("get_flight_status", "book_hotel", "cancel_trip"),
        pkg="travel",
    )
    unrelated_pool = [
        Option("compile_shader", "Compile a GPU shader kernel"),
        Option("prune_orchard", "Schedule apple orchard pruning"),
        Option("tune_piano", "Record piano acoustic frequencies"),
        Option("brew_espresso", "Control espresso boiler temperature"),
        Option("calc_orbit", "Compute satellite orbital inclination"),
    ]
    xab = build_abstain_cross_domain([r_xab], unrelated_pool, limit=5, seed=0)
    assert len(xab) == 1 and xab[0].label == ABSTAIN and "get_flight_status" not in {o.key for o in xab[0].options}

    # wrap_s2_row strips __done__ from normal rows but keeps __done__ when target == "__done__" (zeroshot eval)
    r_done_target = _row(12, "Thank you, everything looks complete!", label="__done__", tools=("__done__", "get_weather"))
    s2_done = wrap_s2_row(r_done_target, shuffle_seed=0)
    assert s2_done.label == "__done__" and "__done__" in {o.key for o in s2_done.options}
    r_other_target = _row(13, "What is the weather today in Pune?", label="get_weather", tools=("__done__", "get_weather"))
    s2_other = wrap_s2_row(r_other_target, shuffle_seed=0)
    assert "__done__" not in {o.key for o in s2_other.options} and {ANSWER, ASK, ESCALATE, ABSTAIN} <= {o.key for o in s2_other.options}
