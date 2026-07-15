# conv_clip_demo.py 测试流程

## 模型定义

```
ConvClipNet:
  input(1,3,64,64) → Conv2d(3,16,k3) → BatchNorm2d(16) → clamp(min=-2, max=2) → output
```

## 测试流程伪代码

```python
# 1. 准备
torch.manual_seed(42)
input = rand(1, 3, 64, 64)

float_model = ConvClipNet().eval()
float_out = float_model(input)                        # 基准输出

# 2. QAT 准备
quantizer = AXQuantizer("config.json")                 # U8/S8 全局配置
exported_model = export_for_training(float_model)       # 保留 BN 训练行为，生成 FX 图
prepared_model = prepare_qat_pt2e(exported_model, quantizer)  # 插入 FakeQuantize 节点
move_to_eval(prepared_model)                           # 冻结 BN 统计量

# 3. 获取 "等价浮点" 输出（关闭所有量化效果）
disable_fake_quant(prepared_model)                     # FakeQuantize 直通
disable_observer(prepared_model)                       # observer 停止统计
prepared_float_like = prepared_model(input)            # 输出应与 float_model 一致

# 4. 校准 observer（用数据统计 min/max）
enable_fake_quant(prepared_model)                      # 恢复 FakeQuantize
enable_observer(prepared_model)                        # 恢复 observer 统计
prepared_model(input)                                  # 前向，observer 更新 running_min/max

# 5. 获得量化输出（冻结 observer，仅启用 fake_quant）
disable_observer(prepared_model)                       # 停止统计，使用已计算的 scale/zp
qat_out = prepared_model(input)                        # 真实的量化输出

# 6. 导出 ONNX
quantized_model = convert_pt2e(prepared_model)         # observer 融合为 Q/DQ 节点
dynamo_export(quantized_model, "conv_clip_qat.onnx")   # 导出 QDQ 格式 ONNX
onnx_simplify("conv_clip_qat.onnx", "conv_clip_qat_sim.onnx")  # 常量折叠

# 7. 对比验证
diff_graph     = abs(float_out - prepared_float_like)  # 图结构差异（应为 0）
diff_quant     = abs(float_out - qat_out)              # 量化误差
diff_quant_eff = abs(prepared_float_like - qat_out)    # 纯量化效应
cos_sim = dot(float_out, qat_out) / (|float| * |qat|)  # 余弦相似度
allclose(float_out, prepared_float_like)               # 图结构校验（应为 True）
```

## 关键步骤说明

| 步骤 | 操作 | 目的 |
|------|------|------|
| `disable_fake_quant` + `disable_observer` | 关闭所有量化 | 验证图结构无损 |
| `enable_fake_quant` + `enable_observer` + forward | 校准 observer | 统计实际 min/max 用于计算 scale/zp |
| `disable_observer` + forward | 冻结量化参数 | 用校准好的 scale/zp 模拟推理 |
| `convert_pt2e` | observer 融合 | 将 observer 转换为 Q/DQ 节点 |

## 输出指标

| 指标 | 含义 |
|------|------|
| `diff float vs prepared OFF` | 图结构差异（理想为 0） |
| `diff float vs prepared ON` | 量化总误差 |
| `diff prepared OFF vs ON` | 纯量化引入的误差 |
| `cos_sim` | 余弦相似度（越接近 1 越好） |
| `allclose(prepared OFF)` | 图结构正确性（应为 True） |
