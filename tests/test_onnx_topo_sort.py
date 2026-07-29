"""topological_sort repairs the node ordering that Ultralytics' half=True export produces."""

import onnx
from onnx import TensorProto, helper

from gdf.export.onnx import topological_sort


def _graph_with_trailing_cast():
    """Mimic the FP16 export: the Cast feeding node 0 is appended last."""
    conv_in = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 4, 4])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT16, [1, 3, 4, 4])
    relu = helper.make_node("Relu", ["cast_out"], ["out"], name="relu")
    cast = helper.make_node("Cast", ["images"], ["cast_out"], to=TensorProto.FLOAT16, name="cast")
    return helper.make_graph([relu, cast], "g", [conv_in], [out])


def test_reorders_producer_before_consumer():
    graph = _graph_with_trailing_cast()
    assert [n.name for n in graph.node] == ["relu", "cast"]

    assert topological_sort(graph) is True
    assert [n.name for n in graph.node] == ["cast", "relu"]
    onnx.checker.check_graph(graph)


def test_already_sorted_graph_is_left_alone():
    graph = _graph_with_trailing_cast()
    topological_sort(graph)
    assert topological_sort(graph) is False


def test_unresolvable_graph_is_not_mangled():
    # 'ghost' is produced by nothing — sorting must bail, leaving the graph untouched.
    node = helper.make_node("Relu", ["ghost"], ["out"], name="relu")
    graph = helper.make_graph(
        [node], "g", [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1])],
    )
    assert topological_sort(graph) is False
    assert [n.name for n in graph.node] == ["relu"]
