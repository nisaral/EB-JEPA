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
