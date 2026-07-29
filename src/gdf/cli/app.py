from __future__ import annotations

import typer

from gdf import __version__

app = typer.Typer(
    name="gdf",
    help="General Detection Framework — YOLO classification with ONNX/TensorRT deployment",
    add_completion=False,
    rich_markup_mode="rich",
)


def version_callback(value: bool) -> None:
    if value:
        typer.echo(f"gdf {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", "-v", callback=version_callback, is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    pass


from gdf.cli.train import train_cmd  # noqa: E402
from gdf.cli.export import export_cmd  # noqa: E402
from gdf.cli.predict import predict_cmd  # noqa: E402
from gdf.cli.track import track_cmd  # noqa: E402
from gdf.cli.webcam import webcam_cmd  # noqa: E402
from gdf.cli.info import info_cmd  # noqa: E402

app.command(name="train")(train_cmd)
app.command(name="export")(export_cmd)
app.command(name="predict")(predict_cmd)
app.command(name="track")(track_cmd)
app.command(name="webcam")(webcam_cmd)
app.command(name="info")(info_cmd)


if __name__ == "__main__":  # `python -m gdf.cli.app`, used by Dockerfile.jetson
    app()
