"""全配置 2.6 vs 2.10 sim 结构对比总表。"""
import onnx
from collections import Counter

D = "resnet50_2_10"
PAIRS = [
    ("config(全局U8/S8)", f"{D}/resnet50_qat_2_6_sim.onnx", f"{D}/resnet50_qat_2_10_sim.onnx"),
    ("4w4f_all(全局4bit)", f"{D}/resnet50_qat_2_6_4w4f_all_sim.onnx", f"{D}/resnet50_qat_2_10_4w4f_all_sim.onnx"),
    ("4w4f(U8+U4区域)", f"{D}/resnet50_qat_2_6_4w4f_sim.onnx", f"{D}/resnet50_qat_2_10_4w4f_sim.onnx"),
    ("16f(U8+U16区域)", f"{D}/resnet50_qat_2_6_16f_sim.onnx", f"{D}/resnet50_qat_2_10_16f_sim.onnx"),
    ("fp32(U16+FP32区域)", f"{D}/resnet50_qat_2_6_fp32_sim.onnx", f"{D}/resnet50_qat_2_10_fp32_sim.onnx"),
]

DT = {onnx.TensorProto.INT8: "S8", onnx.TensorProto.UINT8: "U8",
      onnx.TensorProto.INT16: "S16", onnx.TensorProto.UINT16: "U16",
      onnx.TensorProto.INT4: "S4", onnx.TensorProto.UINT4: "U4",
      onnx.TensorProto.INT32: "S32"}

def profile(path):
    m = onnx.load(path)
    g = m.graph
    inits = {t.name: t for t in g.initializer}
    def zp_dtype(n):
        if len(n.input) < 3:
            return "S8?"
        t = inits.get(n.input[2])
        return DT.get(t.data_type, "?") if t is not None else "?"
    wq, aq = Counter(), Counter()
    for n in g.node:
        if n.op_type == "DequantizeLinear" and n.input[0] in inits:
            wq[zp_dtype(n)] += 1
        elif n.op_type == "QuantizeLinear" and n.input[0] not in inits:
            aq[zp_dtype(n)] += 1
    return {
        "seq": [n.op_type for n in g.node],
        "hist": Counter(n.op_type for n in g.node),
        "wq": dict(sorted(wq.items())),
        "aq": dict(sorted(aq.items())),
        "ir": m.ir_version,
        "opset": {o.domain or "ai.onnx": o.version for o in m.opset_import}["ai.onnx"],
    }

print(f"{'配置':<20} {'节点':>9} {'拓扑序':<8} {'算子直方图':<10} {'权重量化 2.6 | 2.10':<28} {'激活量化 2.6 | 2.10'}")
for name, pa, pb in PAIRS:
    try:
        a, b = profile(pa), profile(pb)
    except Exception as e:
        print(f"{name:<20} 文件缺失: {e}")
        continue
    seq_eq = "逐位一致" if a["seq"] == b["seq"] else f"不同"
    hist_eq = "一致" if a["hist"] == b["hist"] else f"差:{ {k: (a['hist'][k], b['hist'][k]) for k in a['hist'].keys() | b['hist'].keys() if a['hist'][k] != b['hist'][k]} }"
    wq_s = f"{a['wq']} | {b['wq']}" + ("  ✓" if a["wq"] == b["wq"] else "  ✗")
    aq_s = f"{a['aq']} | {b['aq']}" + ("  ✓" if a["aq"] == b["aq"] else "  ✗")
    n_s = f"{len(a['seq'])}/{len(b['seq'])}"
    print(f"{name:<20} {n_s:>9} {seq_eq:<8} {hist_eq:<10} {wq_s:<28} {aq_s}")
    assert (a["ir"], a["opset"]) == (b["ir"], b["opset"]) == (10, 21), f"{name} ir/opset 异常"
print("(ir_version 均=10,opset 均=21,已断言)")
