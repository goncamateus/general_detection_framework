from __future__ import annotations

from pathlib import Path

from gdf.utils.logging import log


def export_onnx(
    weights: Path,
    imgsz: int = 224,
    half: bool = False,
    output_dir: Path | None = None,
) -> Path:
    from ultralytics import YOLO

    # Task (cls/detect/segment) comes from the checkpoint itself — don't force a wrapper.
    model = YOLO(str(weights))

    log.info(f"Exporting {model.task} model to ONNX: imgsz={imgsz}, half={half}")
    export_path = model.export(format="onnx", imgsz=imgsz, half=half)

    onnx_path = Path(export_path)
    if output_dir is not None:
        import shutil

        dest = output_dir / onnx_path.name
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(onnx_path, dest)
        onnx_path = dest

    log.info(f"ONNX exported: {onnx_path}")
    return onnx_path


def topological_sort(graph) -> bool:  # type: ignore[no-untyped-def]
    """Reorder `graph.node` in place so producers precede consumers. True if it changed.

    `half=True` exports run through onnxconverter-common's float16 pass, which appends its
    `graph_input_cast*` nodes at the end of the list even though the very first Conv
    consumes them. onnxruntime sorts internally and does not care, but the ONNX spec
    requires topological order and TensorRT's ONNX parser walks the nodes as listed.
    """
    produced = {i.name for i in graph.input} | {i.name for i in graph.initializer}
    original = list(graph.node)
    pending = original
    ordered = []

    # Protobuf messages compare by value, so track membership by identity.
    while pending:
        ready = [n for n in pending if all(not x or x in produced for x in n.input)]
        if not ready:
            return False  # cycle, or an input from an enclosing scope — leave it alone
        ready_ids = {id(n) for n in ready}
        for node in ready:
            produced.update(o for o in node.output if o)
        ordered.extend(ready)
        pending = [n for n in pending if id(n) not in ready_ids]

    if [id(n) for n in ordered] == [id(n) for n in original]:
        return False

    del graph.node[:]
    graph.node.extend(ordered)
    return True


def verify_onnx(onnx_path: Path) -> bool:
    import onnx

    model = None
    try:
        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)
        log.info(f"ONNX model valid: {onnx_path}")
        return True
    except Exception as e:
        repairable = model is not None and "topologically sorted" in str(e)
        if not repairable or not topological_sort(model.graph):
            log.error(f"ONNX verification failed: {e}")
            return False

    try:
        onnx.checker.check_model(model)
    except Exception as e:
        log.error(f"ONNX verification failed after reordering nodes: {e}")
        return False

    onnx.save(model, str(onnx_path))
    log.warning(f"ONNX nodes were out of topological order; reordered and re-saved: {onnx_path}")
    return True
