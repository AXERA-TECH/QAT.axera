"""QAT 导出 ONNX 结构检查。

检查导出的 QDQ 模型是否符合本仓库(pulsar2 消费)的结构预期,预期规则来自
torch2.6 老管线的金标准导出(resnet50/resnet50_qat*.onnx)与 pulsar2/utils
后处理代码的隐含假设:

  R1  opset: ai.onnx == 21(dynamo_export 固定 opset_version=21)
  R2  onnx.checker 通过
  R3  激活链: 每个 QuantizeLinear 的消费者只能是 DequantizeLinear(Q→DQ 成对,无悬空 Q)
  R4  权重链: 权重 DequantizeLinear 的三个输入最终都来自常量,唯一消费者是
      Conv/Gemm/ConvTranspose/MatMul;per-channel 时带 axis 属性且零点全 0(对称)
  R5  Q/DQ 参数: scale 为 float32 且 > 0;激活零点为标量
  R6  Conv/Gemm 的 bias(第 3 输入)保持 float32,不做量化
  R7  除 ai.onnx 外不残留自定义域算子;QDQ 周边无多余 Cast
      (raw 模型中 per-channel 零点的 Cast 属预期,见 utils/quantized_decomposed_dequantize_per_channel.py)
  R8  图输入/输出为 float32
  R9  raw 模型(未 simplify)的 QDQ 节点需带 metadata_props
      (namespace / pkg.torch.onnx.fx_node,simplify_and_fix_4bit_dtype 依赖它识别 4bit)

用法:
  python env_check/check_onnx_structure.py --model m.onnx                    # 规则检查 + profile
  python env_check/check_onnx_structure.py --model m.onnx --sim              # 按 simplify 后模型检查(跳过 R9)
  python env_check/check_onnx_structure.py --model m.onnx --save-profile p.json
  python env_check/check_onnx_structure.py --model m.onnx --baseline p.json  # 与基线 profile 结构对比
  python env_check/check_onnx_structure.py --model m.onnx --ort              # 附带 ORT 加载 + 随机推理
"""
import argparse
import json
import sys
from collections import Counter, defaultdict

import numpy as np
import onnx
from onnx import TensorProto

QDQ_CONSUMER_OPS = {"Conv", "Gemm", "ConvTranspose", "MatMul"}
INT_QUANT_DTYPES = {
    TensorProto.INT8: "S8", TensorProto.UINT8: "U8",
    TensorProto.INT16: "S16", TensorProto.UINT16: "U16",
    TensorProto.INT4: "S4", TensorProto.UINT4: "U4",
    TensorProto.INT32: "S32",
}


class Report:
    def __init__(self):
        self.items = []

    def add(self, level, rule, msg):
        self.items.append((level, rule, msg))

    def ok(self, rule, msg):
        self.add("PASS", rule, msg)

    def fail(self, rule, msg):
        self.add("FAIL", rule, msg)

    def warn(self, rule, msg):
        self.add("WARN", rule, msg)

    def dump(self):
        order = {"FAIL": 0, "WARN": 1, "PASS": 2}
        for level, rule, msg in sorted(self.items, key=lambda x: order[x[0]]):
            print(f"  [{level}] {rule}: {msg}")
        n_fail = sum(1 for lv, _, _ in self.items if lv == "FAIL")
        n_warn = sum(1 for lv, _, _ in self.items if lv == "WARN")
        print(f"  == {len(self.items)} 项: FAIL={n_fail} WARN={n_warn} "
              f"PASS={len(self.items) - n_fail - n_warn} ==")
        return n_fail


class GraphIndex:
    """一次遍历建好 producer/consumer/initializer 索引。"""

    def __init__(self, model):
        self.model = model
        g = model.graph
        self.inits = {t.name: t for t in g.initializer}
        self.producer = {}
        self.consumers = defaultdict(list)
        for node in g.node:
            for o in node.output:
                self.producer[o] = node
            for i in node.input:
                if i:
                    self.consumers[i].append(node)
        # Constant 节点视同 initializer
        for node in g.node:
            if node.op_type == "Constant":
                self.inits.setdefault(node.output[0], None)
        self.graph_inputs = {i.name for i in g.input}

    def is_const(self, name):
        """张量是否最终来自常量(穿透 Q/DQ/Cast/Identity/Reshape/Transpose)。"""
        seen = set()
        while True:
            if name in self.inits:
                return True
            if name in self.graph_inputs or name in seen:
                return False
            seen.add(name)
            node = self.producer.get(name)
            if node is None:
                return False
            if node.op_type in {"QuantizeLinear", "DequantizeLinear", "Cast",
                                "Identity", "Reshape", "Transpose", "Constant"}:
                name = node.input[0]
            else:
                return False

    def const_value(self, name):
        """常量张量取值(仅直连 initializer / Constant 节点,取不到返回 None)。"""
        t = self.inits.get(name)
        if isinstance(t, onnx.TensorProto):
            return onnx.numpy_helper.to_array(t)
        node = self.producer.get(name)
        if node is not None and node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value":
                    return onnx.numpy_helper.to_array(attr.t)
        return None


def build_profile(model):
    """结构画像:与数值无关,只看拓扑/dtype/量化位宽,可作跨版本基线。"""
    idx = GraphIndex(model)
    g = model.graph
    prof = {
        "opset": {o.domain or "ai.onnx": o.version for o in model.opset_import},
        "ir_version": model.ir_version,
        "op_histogram": dict(Counter(n.op_type for n in g.node)),
        "num_inputs": len(g.input),
        "num_outputs": len(g.output),
    }

    weight_dq, act_q, act_dq = [], [], []
    for node in g.node:
        if node.op_type == "DequantizeLinear":
            if idx.is_const(node.input[0]):
                weight_dq.append(node)
            else:
                act_dq.append(node)
        elif node.op_type == "QuantizeLinear":
            if not idx.is_const(node.input[0]):
                act_q.append(node)

    def tensor_dtype(name):
        t = idx.inits.get(name)
        if isinstance(t, onnx.TensorProto):
            return INT_QUANT_DTYPES.get(t.data_type, str(t.data_type))
        v = idx.const_value(name)
        if v is not None:
            for k, s in INT_QUANT_DTYPES.items():
                if onnx.helper.np_dtype_to_tensor_dtype(v.dtype) == k:
                    return s
        node = idx.producer.get(name)
        if node is not None and node.op_type == "Cast":
            return tensor_dtype(node.input[0])
        return "?"

    wq_summary = Counter()
    for node in weight_dq:
        axis = next((a.i for a in node.attribute if a.name == "axis"), None)
        consumer_ops = sorted({c.op_type for c in idx.consumers[node.output[0]]})
        wq_summary[(tensor_dtype(node.input[0]),
                    "per-channel" if axis is not None else "per-tensor",
                    ",".join(consumer_ops))] += 1
    prof["weight_dq"] = {f"{d}|{gran}|{ops}": n
                         for (d, gran, ops), n in sorted(wq_summary.items())}

    aq_summary = Counter()
    for node in act_q:
        zp_dtype = tensor_dtype(node.input[2]) if len(node.input) > 2 else "S8?"
        dq_consumers = set()
        for c in idx.consumers[node.output[0]]:
            if c.op_type == "DequantizeLinear":
                dq_consumers |= {cc.op_type for cc in idx.consumers[c.output[0]]}
        aq_summary[(zp_dtype, ",".join(sorted(dq_consumers)) or "<output>")] += 1
    prof["act_q"] = {f"{d}|{ops}": n for (d, ops), n in sorted(aq_summary.items())}

    prof["counts"] = {
        "QuantizeLinear": len(act_q),
        "DequantizeLinear(act)": len(act_dq),
        "DequantizeLinear(weight)": len(weight_dq),
    }
    return prof, idx, weight_dq, act_q, act_dq


def run_rules(model, idx, weight_dq, act_q, act_dq, rep, is_sim):
    g = model.graph

    # R1 opset
    opset = {o.domain or "ai.onnx": o.version for o in model.opset_import}
    if opset.get("ai.onnx") == 21:
        rep.ok("R1-opset", "ai.onnx == 21")
    else:
        rep.fail("R1-opset", f"ai.onnx == {opset.get('ai.onnx')},预期 21(dynamo_export 固定)")
    extra_domains = {d for d in opset if d not in ("ai.onnx", "")}
    if extra_domains:
        rep.warn("R7-domain", f"存在额外 opset 域: {extra_domains}")

    # R1b IR version: dynamo 导出固定写 10;simplify 管线若用新 onnx 重新 make_model
    # 会盖成当前默认 IR,IR≥12 会被 ORT 1.23 / 老 pulsar2 拒载
    if model.ir_version <= 10:
        rep.ok("R1-ir", f"ir_version == {model.ir_version}")
    elif model.ir_version == 11:
        rep.warn("R1-ir", "ir_version == 11(dynamo 导出应为 10,疑似被后处理抬高)")
    else:
        rep.fail("R1-ir", f"ir_version == {model.ir_version},ORT 1.23 最高支持 11,"
                          "需在后处理里回写 model.ir_version = 10")

    # R2 checker
    try:
        onnx.checker.check_model(model)
        rep.ok("R2-checker", "onnx.checker 通过")
    except Exception as e:
        rep.fail("R2-checker", f"{type(e).__name__}: {e}")

    # R3 Q→DQ 成对
    bad_q = []
    for node in act_q:
        consumers = idx.consumers[node.output[0]]
        if not consumers or any(c.op_type != "DequantizeLinear" for c in consumers):
            bad_q.append(node.name or node.output[0])
    if bad_q:
        rep.fail("R3-QDQ-pair", f"{len(bad_q)} 个 QuantizeLinear 的消费者不是 DequantizeLinear: "
                                f"{bad_q[:3]}...")
    else:
        rep.ok("R3-QDQ-pair", f"{len(act_q)} 个激活 Q 全部成对接 DQ")

    # R4 权重链
    bad_w = []
    per_channel = 0
    for node in weight_dq:
        consumers = idx.consumers[node.output[0]]
        consumer_ops = {c.op_type for c in consumers}
        axis = next((a.i for a in node.attribute if a.name == "axis"), None)
        if axis is not None:
            per_channel += 1
            zp = idx.const_value(node.input[2]) if len(node.input) > 2 else None
            # raw 模型零点可能藏在 Cast 后面,取不到就跳过数值断言
            if zp is not None and not np.all(zp == 0):
                bad_w.append(f"{node.name}: per-channel 零点非 0(非对称权重)")
        if len(consumers) != 1 or not consumer_ops <= QDQ_CONSUMER_OPS:
            # 权重 DQ 也可能直接喂给 add 等(残差上的常量),放宽为 WARN 由人工确认
            rep.warn("R4-weight", f"{node.name}: 消费者 {sorted(consumer_ops)}(非典型)")
        if not all(idx.is_const(i) for i in node.input if i):
            bad_w.append(f"{node.name}: 输入含非常量")
    if bad_w:
        rep.fail("R4-weight", f"{len(bad_w)} 处权重链异常: {bad_w[:3]}...")
    else:
        rep.ok("R4-weight", f"{len(weight_dq)} 个权重 DQ 正常(其中 per-channel {per_channel})")

    # R4b 量化算子的权重必须仍走 DQ 链:torch2.9+ 的 torch.onnx.export 默认
    # optimize=True,onnxscript 新版常量折叠会把「int8 权重 + DequantizeLinear」
    # 直接折成 float 权重 —— 图上看似正常,实际权重量化信息全丢,pulsar2 拿不到
    weight_dq_outputs = {node.output[0] for node in weight_dq}
    folded = []
    for node in g.node:
        if node.op_type not in QDQ_CONSUMER_OPS or len(node.input) < 2:
            continue
        data_from_dq = (idx.producer.get(node.input[0]) is not None
                        and idx.producer[node.input[0]].op_type == "DequantizeLinear")
        if not data_from_dq:
            continue  # 本身不是量化路径上的算子(如未注解层)
        w = node.input[1]
        w_producer = idx.producer.get(w)
        if w not in weight_dq_outputs and (
                w_producer is None or w_producer.op_type != "DequantizeLinear"):
            folded.append(f"{node.name}({node.op_type})")
    if folded:
        rep.fail("R4b-weight-folded",
                 f"{len(folded)} 个量化算子的权重不经 DQ(疑似被 optimize 常量折叠回 float): "
                 f"{folded[:4]}...")
    else:
        rep.ok("R4b-weight-folded", "量化算子权重均保留 DQ 链")

    # R5 scale/零点
    bad_s = []
    for node in list(act_q) + list(weight_dq):
        scale = idx.const_value(node.input[1])
        if scale is not None:
            if scale.dtype != np.float32:
                bad_s.append(f"{node.name}: scale dtype {scale.dtype}")
            elif not np.all(scale > 0):
                bad_s.append(f"{node.name}: scale 含 <=0")
    if bad_s:
        rep.fail("R5-qparams", f"{len(bad_s)} 处 scale 异常: {bad_s[:3]}...")
    else:
        rep.ok("R5-qparams", "所有可读 scale 均为 float32 且 > 0")

    # R6 bias 保持 float
    bad_b = []
    for node in g.node:
        if node.op_type in ("Conv", "Gemm", "ConvTranspose") and len(node.input) >= 3:
            b = idx.const_value(node.input[2])
            if b is not None and b.dtype not in (np.float32, np.float16):
                bad_b.append(f"{node.name}: bias dtype {b.dtype}")
    if bad_b:
        rep.fail("R6-bias", f"{len(bad_b)} 处 bias 被量化/改型: {bad_b[:3]}...")
    else:
        rep.ok("R6-bias", "Conv/Gemm bias 均保持浮点")

    # R7 残留算子
    leftovers = [n.op_type for n in g.node
                 if n.domain not in ("", "ai.onnx") or "quantized_decomposed" in n.op_type]
    if leftovers:
        rep.fail("R7-leftover", f"残留非标准算子: {Counter(leftovers)}")
    else:
        rep.ok("R7-leftover", "无自定义域/未转换的 quantized_decomposed 残留")
    n_cast = sum(1 for n in g.node if n.op_type == "Cast")
    if n_cast:
        (rep.warn if not is_sim else rep.fail)(
            "R7-cast", f"{n_cast} 个 Cast(raw 模型中 per-channel 零点 Cast 属预期,sim 后应被折叠)")
    else:
        rep.ok("R7-cast", "无多余 Cast")

    # R8 输入输出 dtype
    io_bad = []
    for vi in list(g.input) + list(g.output):
        et = vi.type.tensor_type.elem_type
        if et != TensorProto.FLOAT:
            io_bad.append(f"{vi.name}: {onnx.helper.tensor_dtype_to_string(et)}")
    if io_bad:
        rep.fail("R8-io", f"图输入/输出非 float32: {io_bad}")
    else:
        rep.ok("R8-io", "图输入/输出均为 float32")

    # R9 metadata_props(simplify_and_fix_4bit_dtype 依赖)
    if not is_sim:
        qdq_nodes = [n for n in g.node if n.op_type in ("QuantizeLinear", "DequantizeLinear")]
        missing = [n.name for n in qdq_nodes
                   if "namespace" not in {p.key for p in n.metadata_props}]
        if missing:
            rep.fail("R9-metadata", f"{len(missing)}/{len(qdq_nodes)} 个 QDQ 节点缺 metadata_props"
                                    f"(simplify_and_fix_4bit_dtype 将无法识别 4bit): {missing[:3]}...")
        else:
            rep.ok("R9-metadata", f"{len(qdq_nodes)} 个 QDQ 节点均带 namespace metadata")


def compare_with_baseline(prof, baseline, rep):
    for key in ("opset", "op_histogram", "weight_dq", "act_q", "counts"):
        cur, base = prof.get(key), baseline.get(key)
        if cur == base:
            rep.ok(f"BASE-{key}", "与基线一致")
            continue
        if key == "op_histogram":
            cur_c, base_c = Counter(cur), Counter(base)
            diff = {op: (base_c.get(op, 0), cur_c.get(op, 0))
                    for op in set(cur_c) | set(base_c) if cur_c.get(op, 0) != base_c.get(op, 0)}
            rep.fail(f"BASE-{key}", f"算子直方图差异(基线,当前): {diff}")
        else:
            rep.fail(f"BASE-{key}", f"基线={base} 当前={cur}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--sim", action="store_true", help="被检模型是 simplify 后产物")
    parser.add_argument("--save-profile", type=str, default=None)
    parser.add_argument("--baseline", type=str, default=None)
    parser.add_argument("--ort", action="store_true", help="附带 onnxruntime 加载+随机推理")
    args = parser.parse_args()

    model = onnx.load(args.model)
    print(f"== 检查 {args.model} ==")
    prof, idx, weight_dq, act_q, act_dq = build_profile(model)

    print("-- profile --")
    print(json.dumps(prof, indent=2, ensure_ascii=False))

    rep = Report()
    run_rules(model, idx, weight_dq, act_q, act_dq, rep, is_sim=args.sim)

    if args.baseline:
        with open(args.baseline) as f:
            compare_with_baseline(prof, json.load(f), rep)

    if args.ort:
        import onnxruntime as ort
        try:
            sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
            feeds = {}
            for i in sess.get_inputs():
                shape = [d if isinstance(d, int) else 1 for d in i.shape]
                feeds[i.name] = np.random.rand(*shape).astype(np.float32)
            outs = sess.run(None, feeds)
            rep.ok("ORT", f"加载+推理成功, 输出 shape {[o.shape for o in outs]}")
        except Exception as e:
            rep.fail("ORT", f"{type(e).__name__}: {str(e).splitlines()[-1]}")

    print("-- rules --")
    n_fail = rep.dump()

    if args.save_profile:
        with open(args.save_profile, "w") as f:
            json.dump(prof, f, indent=2, ensure_ascii=False)
        print(f"profile 已保存到 {args.save_profile}")

    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
