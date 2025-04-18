import argparse

import numpy as np

import onnx
import onnx_graphsurgeon as gs

INT4_DTYPE = np.dtype((np.int8, {"int4": (np.int8, 0)}))


def is_weight_dequant(node: gs.Node):
    return (
        node.op == "DequantizeLinear"
        and len(node.outputs[0].outputs) == 1
        and node.o().op == "Conv"
        and len(node.inputs) == 3
        and isinstance(node.inputs[0], gs.Constant)
        and isinstance(node.inputs[1], gs.Constant)
        and isinstance(node.inputs[2], gs.Constant)
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
    onnx_graph = gs.import_onnx(onnx_model)

    for node in onnx_graph.nodes:
        if is_weight_dequant(node):
            x: gs.Constant = node.inputs[0]
            x.values = x.values.astype(INT4_DTYPE)
            x_zero_point: gs.Constant = node.inputs[2]
            x_zero_point.values = x_zero_point.values.astype(INT4_DTYPE)

    onnx_model = gs.export_onnx(onnx_graph)
    onnx.save(onnx_model, args.output)
    print(f"Save {args.output}")
    model_4w8f_info = summarize_model(args.output)
    print_model_info_as_table([model_info, model_4w8f_info])


if __name__ == "__main__":
    main()
