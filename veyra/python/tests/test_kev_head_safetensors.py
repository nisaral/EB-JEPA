import pytest
import torch

pytest.importorskip("safetensors")

from soro.routing.kev_model import (
    SoroKevModel,
    convert_head_pt_to_safetensors,
    load_soro_kev_checkpoint,
    save_soro_kev_checkpoint,
)


def test_head_safetensors_roundtrip(tmp_path):
    model = SoroKevModel(use_mock=True)
    save_soro_kev_checkpoint(model, tmp_path, calibrated_temp=0.1, q_hat=0.9)
    assert (tmp_path / "head.pt").exists()
    assert (tmp_path / "head.safetensors").exists()
    loaded, _ = load_soro_kev_checkpoint(tmp_path)
    for name in ("q_proj_head", "opt_proj_head"):
        a = getattr(model, name).state_dict()
        b = getattr(loaded, name).state_dict()
        for k in a:
            assert torch.equal(a[k], b[k])


def test_convert_legacy_head_pt(tmp_path):
    model = SoroKevModel(use_mock=True)
    save_soro_kev_checkpoint(model, tmp_path)
    (tmp_path / "head.safetensors").unlink()
    out = convert_head_pt_to_safetensors(tmp_path)
    assert out.exists()
    (tmp_path / "head.pt").unlink()
    loaded, _ = load_soro_kev_checkpoint(tmp_path)
    assert torch.equal(model.q_proj_head.weight, loaded.q_proj_head.weight)
