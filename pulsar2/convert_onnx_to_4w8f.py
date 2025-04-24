import argparse

import numpy as np

import onnx

from onnx_graphsurgeon.exporters.onnx_exporter import OnnxExporter
from onnx_graphsurgeon.exporters.onnx_exporter import check_duplicate_node_names
from onnx_graphsurgeon.exporters.onnx_exporter import dtype_to_onnx
from onnx_graphsurgeon.exporters.onnx_exporter import update_import_domains
from onnx_graphsurgeon.exporters.onnx_exporter import _NUMPY_ARRAY_CONVERTERS
from onnx_graphsurgeon.logger import G_LOGGER
from onnx_graphsurgeon.importers.onnx_importer import import_onnx
from onnx_graphsurgeon.ir.graph import Graph
from onnx_graphsurgeon.ir.node import Node
from onnx_graphsurgeon.ir.tensor import Constant
from onnx_graphsurgeon.ir.tensor import LazyValues
from onnx_graphsurgeon.ir.tensor import SparseValues
from onnx_graphsurgeon.ir.tensor import Variable


INT4_DTYPE = np.dtype((np.int8, {"int4": (np.int8, 0)}))


def constant_to_onnx_tensor(tensor: Constant) -> onnx.TensorProto:
    source_dtype = dtype_to_onnx(tensor.dtype)
    target_dtype = dtype_to_onnx(tensor.export_dtype)

    if source_dtype != target_dtype:
        source_dtype_str = onnx.helper.tensor_dtype_to_string(source_dtype)
        target_dtype_str = onnx.helper.tensor_dtype_to_string(target_dtype)
        assert source_dtype == onnx.TensorProto.FLOAT, (
            f"Cannot convert onnx dtype {source_dtype_str} to {target_dtype_str}. "
            "Source dtype must be float32 to convert to numpy unsupported dtypes."
        )
        assert target_dtype in _NUMPY_ARRAY_CONVERTERS.keys(), (
            f"Cannot convert onnx dtype {source_dtype_str} to {target_dtype_str}. "
            f"Only float32 to {_NUMPY_ARRAY_CONVERTERS.keys()} is supported."
        )
        arr = _NUMPY_ARRAY_CONVERTERS[target_dtype](tensor.values)
        tensor_raw_bytes = arr.tobytes()
    elif source_dtype == onnx.TensorProto.INT4:
        from onnxruntime.quantization.quant_utils import pack_bytes_to_4bit

        tensor_raw_bytes = bytes(pack_bytes_to_4bit(tensor.values.tobytes()))
    else:
        tensor_raw_bytes = tensor.values.tobytes()

    return onnx.helper.make_tensor(
        name=tensor.name,
        data_type=target_dtype,
        dims=tensor.shape,
        vals=tensor_raw_bytes,
        raw=True,
    )


class AxOnnxExporter(OnnxExporter):
    @staticmethod
    def export_tensor_proto(tensor: Constant) -> onnx.TensorProto:
        # Do *not* load LazyValues into an intermediate numpy array - instead, use
        # the original onnx.TensorProto directly.
        if isinstance(tensor._values, LazyValues):
            onnx_tensor = tensor._values.tensor
            onnx_tensor.name = tensor.name
        else:
            onnx_tensor = constant_to_onnx_tensor(tensor)

            if tensor.data_location is not None:
                onnx_tensor.data_location = tensor.data_location
        return onnx_tensor

    @staticmethod
    def export_graph(graph: Graph, do_type_check=True) -> onnx.GraphProto:
        """
        Export an onnx-graphsurgeon Graph to an ONNX GraphProto.

        Args:
            graph (Graph): The graph to export.

            do_type_check (bool): Whether to check that input and output tensors have data types defined, and fail if not.
                                  Defaults to True.
        """
        check_duplicate_node_names(graph.nodes, level=G_LOGGER.WARNING)
        nodes = [AxOnnxExporter.export_node(node) for node in graph.nodes]
        inputs = [
            AxOnnxExporter.export_value_info_proto(inp, do_type_check)
            for inp in graph.inputs
        ]
        outputs = [
            AxOnnxExporter.export_value_info_proto(out, do_type_check)
            for out in graph.outputs
        ]
        tensor_map = graph.tensors()
        initializer = [
            AxOnnxExporter.export_tensor_proto(tensor)
            for tensor in tensor_map.values()
            if isinstance(tensor, Constant)
            and not isinstance(tensor._values, SparseValues)
        ]
        sparse_initializer = [
            AxOnnxExporter.export_sparse_tensor_proto(tensor)
            for tensor in tensor_map.values()
            if isinstance(tensor, Constant) and isinstance(tensor._values, SparseValues)
        ]

        # Remove inputs and outputs to export ValueInfoProtos
        for tensor in graph.inputs + graph.outputs:
            if tensor.name in tensor_map:
                del tensor_map[tensor.name]

        # Omit tensors from value_info if we don't know their shape/dtype
        def has_value_info(tensor):
            return isinstance(tensor, Variable) and (
                tensor.dtype is not None or tensor.shape is not None
            )

        value_info = [
            AxOnnxExporter.export_value_info_proto(tensor, do_type_check)
            for tensor in tensor_map.values()
            if has_value_info(tensor)
        ]

        return onnx.helper.make_graph(
            nodes=nodes,
            name=graph.name,
            inputs=inputs,
            outputs=outputs,
            initializer=initializer,
            sparse_initializer=sparse_initializer,
            doc_string=graph.doc_string,
            value_info=value_info,
        )


def export_onnx(graph: Graph, do_type_check=True, **kwargs) -> "onnx.ModelProto":
    """
    Exports an onnx-graphsurgeon Graph to an ONNX model.

    Args:
        graph (Graph): The graph to export

        do_type_check (bool): Whether to check that input and output tensors have data types defined, and fail if not.
                              Defaults to True.
        kwargs: Additional arguments to onnx.helper.make_model

    Returns:
        onnx.ModelProto: A corresponding ONNX model.
    """
    onnx_graph = AxOnnxExporter.export_graph(graph, do_type_check=do_type_check)
    onnx_functions = [AxOnnxExporter.export_function(func) for func in graph.functions]
    kwargs["functions"] = onnx_functions

    if "opset_imports" not in kwargs:
        kwargs["opset_imports"] = update_import_domains(graph)

    model = onnx.helper.make_model(onnx_graph, **kwargs)
    model.producer_name = graph.producer_name
    model.producer_version = graph.producer_version
    return model


def is_weight_dequant(node: Node):
    return (
        node.op == "DequantizeLinear"
        and len(node.outputs[0].outputs) == 1
        and node.o().op == "Conv"
        and len(node.inputs) == 3
        and isinstance(node.inputs[0], Constant)
        and isinstance(node.inputs[1], Constant)
        and isinstance(node.inputs[2], Constant)
        and len(node.inputs[0].shape) > 1
    )


def main():
    from onnxslim.utils import print_model_info_as_table, summarize_model

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()

    print(f"Loading {args.input} ...")
    onnx_model = onnx.load(args.input)
    model_info = summarize_model(onnx_model)
    onnx_graph = import_onnx(onnx_model)

    for node in onnx_graph.nodes:
        if is_weight_dequant(node):
            x: Constant = node.inputs[0]
            x.values = x.values.astype(INT4_DTYPE)
            x_zero_point: Constant = node.inputs[2]
            x_zero_point.values = x_zero_point.values.astype(INT4_DTYPE)

    onnx_model = export_onnx(onnx_graph)
    onnx.save(onnx_model, args.output)
    print(f"Save {args.output}")
    model_4w8f_info = summarize_model(args.output)
    print_model_info_as_table([model_info, model_4w8f_info])


if __name__ == "__main__":
    main()
