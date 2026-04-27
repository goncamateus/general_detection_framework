from pathlib import Path

from gdf.datasets.local import LocalDatasetSource


def test_local_source_resolve(tmp_dataset: Path):
    src = LocalDatasetSource(tmp_dataset)
    resolved = src.resolve()
    assert resolved == tmp_dataset


def test_local_source_validate(tmp_dataset: Path):
    src = LocalDatasetSource(tmp_dataset)
    assert src.validate() is True


def test_local_source_validate_missing():
    src = LocalDatasetSource("/nonexistent/path")
    assert src.validate() is False


def test_local_source_get_classes(tmp_dataset: Path):
    src = LocalDatasetSource(tmp_dataset)
    classes = src.get_class_names()
    assert sorted(classes) == ["bird", "cat", "dog"]
