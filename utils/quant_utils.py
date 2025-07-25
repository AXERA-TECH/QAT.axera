import re
import onnx
import torch
import onnxslim.third_party.onnx_graphsurgeon as gs
from onnxruntime.quantization.quant_utils import pack_bytes_to_4bit


def fix_4bit_dtype(qat_path: str, fix_path: str):
    # load
    onnx_model = onnx.load(qat_path)

    # 4bit info
    tensors_4bit = {}
    for node in onnx_model.graph.node:
        if node.op_type not in ["QuantizeLinear", "DequantizeLinear"]:
            continue

        metadata_props = {}
        for metadata_prop in node.metadata_props:
            metadata_props.update({metadata_prop.key: metadata_prop.value})
        namespace = metadata_props.get("namespace", None)
        fx_node = metadata_props.get("pkg.torch.onnx.fx_node", None)
        if not namespace or not fx_node:
            continue

        target = namespace.split(": ")[1]
        node_args = re.findall(r"args\s*=\s*\(([^)]+)\)", fx_node)[0].split(", ")

        if target == "quantized_decomposed.quantize_per_tensor.default":
            assert len(node_args) == 6
            input, scale, zp, quant_min, quant_max, dtype = node_args
            if int(quant_min) == 0 and int(quant_max) == 15 and dtype == "torch.uint8":
                tensors_4bit.update({node.output[0]: onnx.TensorProto.UINT4})
                tensors_4bit.update({node.input[2]: onnx.TensorProto.UINT4})
        elif target == "quantized_decomposed.dequantize_per_tensor.default":
            assert len(node_args) == 6
            input, scale, zp, quant_min, quant_max, dtype = node_args
            if int(quant_min) == 0 and int(quant_max) == 15 and dtype == "torch.uint8":
                tensors_4bit.update({node.input[0]: onnx.TensorProto.UINT4})
                tensors_4bit.update({node.input[2]: onnx.TensorProto.UINT4})
        elif target == "quantized_decomposed.dequantize_per_channel.default":
            assert len(node_args) == 7
            input, scale, zp, axis, quant_min, quant_max, dtype = node_args
            if int(quant_min) == -7 and int(quant_max) == 7 and dtype == "torch.int8":
                tensors_4bit.update({node.input[0]: onnx.TensorProto.INT4})
                tensors_4bit.update({node.input[2]: onnx.TensorProto.INT4})
        else:
            assert False, f"node target: [{target}] is illegal"

    # constant falding
    graph = gs.import_onnx(onnx_model).toposort()
    graph.fold_constants().cleanup().toposort()
    sim_model = gs.export_onnx(graph)

    # fix 4bit node
    vis = {}
    for vi in sim_model.graph.value_info:
        vis.update({vi.name: vi})
    params = {}
    for param in sim_model.graph.initializer:
        params.update({param.name: param})

    for name, dtype in tensors_4bit.items():
        if name in vis:
            vis[name].type.tensor_type.elem_type = dtype
        if name in params:
            data = onnx.numpy_helper.to_array(params[name])
            data = bytes(pack_bytes_to_4bit(data.tobytes()))
            params[name].data_type = dtype
            params[name].raw_data = data

    # save
    onnx.save(sim_model, fix_path)
    print(f"save onnx model to [{fix_path}] Successfully!")