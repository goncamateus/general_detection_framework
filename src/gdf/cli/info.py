from __future__ import annotations

import platform

import typer
from rich.console import Console
from rich.table import Table

console = Console()


def info_cmd() -> None:
    from gdf import __version__
    from gdf.models.registry import list_available_models
    from gdf.utils.device import get_device_info

    device = get_device_info()

    table = Table(title="GDF Environment Info", show_lines=True)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("GDF Version", __version__)
    table.add_row("Python", platform.python_version())
    table.add_row("Platform", platform.platform())
    table.add_row("Device", device.name)
    table.add_row("CUDA Available", str(device.cuda_available))
    table.add_row("CUDA Version", device.cuda_version)
    table.add_row("GPU Count", str(device.device_count))
    table.add_row("TensorRT Available", str(device.trt_available))
    table.add_row("TensorRT Version", device.trt_version)

    console.print(table)

    models_table = Table(title="Available Models", show_lines=True)
    models_table.add_column("Version", style="cyan")
    models_table.add_column("Sizes", style="green")

    for version, sizes in list_available_models().items():
        models_table.add_row(version, ", ".join(sizes))

    console.print(models_table)

    try:
        import ultralytics
        console.print(f"[dim]Ultralytics: {ultralytics.__version__}[/dim]")
    except ImportError:
        console.print("[yellow]Ultralytics not installed[/yellow]")
