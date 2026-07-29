from pathlib import Path


def test_cli_version():
    from typer.testing import CliRunner

    from gdf.cli.app import app

    runner = CliRunner()
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "gdf" in result.output.lower()


def test_cli_info():
    from typer.testing import CliRunner

    from gdf.cli.app import app

    runner = CliRunner()
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0


def test_cli_train_missing_args():
    from typer.testing import CliRunner

    from gdf.cli.app import app

    runner = CliRunner()
    result = runner.invoke(app, ["train"])
    assert result.exit_code != 0


def _seg_dataset(tmp_path: Path, label_row: str) -> Path:
    (tmp_path / "data.yaml").write_text("nc: 1\nnames: [plume]\n")
    labels = tmp_path / "train" / "labels"
    labels.mkdir(parents=True)
    # Roboflow exports have no trailing newline — the sniff must handle that.
    (labels / "frame.txt").write_text(label_row)
    return tmp_path


def test_resolve_task_sniffs_polygon_labels(tmp_path: Path):
    from gdf.cli.train import _resolve_task

    root = _seg_dataset(tmp_path, "0 0.1 0.1 0.2 0.2 0.3 0.15 0.2 0.05")
    assert _resolve_task("auto", root) == "segment"


def test_resolve_task_sniffs_bbox_labels(tmp_path: Path):
    from gdf.cli.train import _resolve_task

    root = _seg_dataset(tmp_path, "0 0.5 0.5 0.2 0.2")
    assert _resolve_task("auto", root) == "detect"


def test_resolve_task_without_data_yaml_is_cls(tmp_path: Path):
    from gdf.cli.train import _resolve_task

    (tmp_path / "train" / "cat").mkdir(parents=True)
    assert _resolve_task("auto", tmp_path) == "cls"


def test_resolve_task_explicit_wins_over_sniff(tmp_path: Path):
    from gdf.cli.train import _resolve_task

    root = _seg_dataset(tmp_path, "0 0.5 0.5 0.2 0.2")
    assert _resolve_task("segment", root) == "segment"
