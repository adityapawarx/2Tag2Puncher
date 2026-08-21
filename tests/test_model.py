from __future__ import annotations

import pytest

from tagpuncher.model import TagPuncherModel


def test_predict_ranks_the_right_tag_first(tiny_model):
    [tags] = tiny_model.predict(["the star and the galaxy seen through a telescope"], top_k=2)
    assert tags[0].tag == "Astronomy"
    assert tags[0].score >= tags[-1].score


def test_save_and_load_roundtrip(tiny_model, tmp_path):
    artifact = tiny_model.save(tmp_path / "artifact")
    reloaded = TagPuncherModel.load(artifact)

    assert reloaded.tags == tiny_model.tags
    assert reloaded.manifest.version == "test"
    assert reloaded.manifest.n_labels == 3

    text = ["the guitarist tuned the strings"]
    assert reloaded.predict(text)[0][0].tag == tiny_model.predict(text)[0][0].tag


def test_loading_a_non_artifact_directory_explains_itself(tmp_path):
    with pytest.raises(FileNotFoundError, match="not a TagPuncher artifact"):
        TagPuncherModel.load(tmp_path)
