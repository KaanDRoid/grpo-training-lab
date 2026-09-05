"""Model defaults must work in a clone without downloaded weights."""

from pathlib import Path
import runpy
import shutil

import pytest


@pytest.mark.parametrize("local_weights", [False, True])
def test_model_paths_in_fresh_checkout(tmp_path, local_weights):
    source = Path(__file__).resolve().parents[1] / "lora_fp16_config.py"
    copied_config = tmp_path / source.name
    shutil.copyfile(source, copied_config)
    model_names = ["Qwen2.5-0.5B-Instruct", "Qwen2.5-1.5B-Instruct"]
    if local_weights:
        for name in model_names:
            (tmp_path / name).mkdir()

    config = runpy.run_path(str(copied_config))["MODEL"]
    expected = [str(tmp_path / name) if local_weights else f"Qwen/{name}" for name in model_names]
    assert [config.small_model, config.model] == expected
