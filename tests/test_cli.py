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
