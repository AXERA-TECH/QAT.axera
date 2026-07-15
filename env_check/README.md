# env_check — torch2.10 环境与导出结构检查

> 注:本文档大部分为迁移过程的历史记录,内文的 utils_2_10/resnet50_2_10 等
> 路径反映当时布局;2026-07-14 起实现层已统一为 utils/、目录对齐上游
> (见 README_2_10.md 与 plan_unified_api.md)。

## 文件说明

| 文件 | 作用 |
|------|------|
| `qax_smoke.py` / `qax_probe*.py` / `qax_smoke2.py` | 环境冒烟与 PT2E API 对照实验(见 ../env.md) |
| `check_onnx_structure.py` | **QAT 导出 ONNX 结构检查器**(规则 R1~R9 + profile + 基线对比) |
| `gen_candidate_torch210.py` | 验收脚本(统一 utils API):AXQuantizer 全链路(注解计数断言→QAT→导出→simplify),双环境可跑 |
| `baselines/resnet50_qat.profile.json` | 金标准结构基线(torch2.6 管线 raw 导出,S8 权重/U8 激活) |
| `baselines/resnet50_qat_sim.profile.json` | 金标准结构基线(simplify 后,4w4f 流:S4 权重/U4 激活) |
| `baselines/minimum_qat.profile.json` | minimum demo 金标准(torch2.6 env 实跑原版 demo 生成,raw) |
| `baselines/minimum_qat_sim.profile.json` | minimum demo 金标准(同上,simplify 后) |
| `out/` | 候选模型产物(不入库) |

## 2026-07-10:onnxscript 升 0.6.2(QDQ 折叠修复)后的验证与决策

背景:optimize 折叠权重 DQ 的问题在 onnxscript 0.6.2 上游修复(本项目负责人
提报)。升级(0.5.4→0.6.2;onnx-ir 按用户指定压在 0.1.15 即 `<0.1.16`,
pip check 无冲突)后实测:

- ✅ **折叠已修**:optimize=True 导出 4w4f resnet50,54/54 权重 DQ 保留,
  checker 13/13 全 PASS(`resnet50_qat_2_10_4w4f_opt.onnx`);
- ✅ 现行 optimize=False 管线回归全绿(gen_candidate + checker);
- ❌ **但 optimize 不能直接开回来**:其 initializer 去重仍会把相等激活 zp
  合并共享,`simplify_and_fix_4bit_dtype` 按名标 4bit 即被污染——
  optimize=True 路线跑混合 4w4f,sim 激活 U4=55(本意 8),与 2.6 老 bug
  同数;权重侧 S4=5 ✓(per-channel zp 向量未被合并)。

**当前决策:`dynamo_export` 维持 optimize=False**(对所有配置安全,sim 终点
产物不受影响)。若要开回 optimize(收益:raw 干净、与 2.6 管线形态对齐),
需先给 simplify 的 4bit 标记加共享防御(翻转 dtype 前按引用计数拆分共享
zp 常量),届时全配置矩阵重新回归。


### Backlog:simplify 共享 zp 防御(设计已定稿,2026-07-10;按需执行)

**目的**:让 `simplify_and_fix_4bit_dtype` 的 4bit 标记对"共享 zp 常量"免疫,
从而解锁 `dynamo_export` 开回 optimize=True(onnxscript 0.6.2 已修折叠,
只剩 initializer 去重这一关)。

**机制回顾**(归因实验数据见上文"全配置矩阵复测"):标记的**判定**按节点
metadata(准确),**落笔**按张量名翻转 dtype;若 zp 常量被去重共享
(如 ReLU 后激活 zp 全为 0),翻转一个名字连带全部共享者
(2.6 raw:判定 21 节点 → 波及 133、污染 112;2.10 optimize=False raw:
23 → 23、污染 0)。

**实现设计**(~20 行,改 utils/quant_utils.py——统一实现层,原 utils_2_10——的 tensors_4bit 构建段):
1. 第一遍只收集「4bit 节点集合」(现有 metadata 判定逻辑不动);
2. 第二遍对每个待标记的 initializer 名:统计引用它的全部 Q/DQ 节点,
   若存在非 4bit 引用者 → 复制私有常量副本(如 name+"_4bit",同值),
   把 4bit 节点的对应输入改接副本,标记副本名;否则按原名标记;
3. Q 输出等 SSA 张量名不会跨节点共享,照旧直接标记。

**完成判据**:①optimize=True 路线跑混合 config_4w4f,sim 激活 U4=8、
权重 S4=5(当前 optimize 路线为 55/5);②cmp_sim_matrix.py 全配置回归;
③gen_candidate TinyNet 链路回归;④若同时开回 optimize,
dynamo_export 里的 onnx.inliner 补丁可保留(functions 为空时是无操作)。

**触发条件**(满足其一再做):需要开回 optimize=True(如 raw 需被直接消费/
导出耗时敏感);或 2.6 老管线继续产出混合 4bit 交付件——届时同样的防御
抄到 utils/quant_utils.py(2.6 版)即可修复老 bug,历史混合 4bit 产物需重导。

### 待确认:Bias 量化覆盖面(2026-07-15 记录,待与内部后端团队核对)

**现状**(utils/ax_quantizer_utils.py::annotate_bias,双版本一致):

| 算子 | bias 量化? | 说明 |
|------|-----------|------|
| conv1d / conv2d | ✅ 默认量化 | DerivedQuantizationSpec:int32、per-channel(ch_axis=0)、对称;scale 派生 = act_scale × weight_scale,zp=0;导出为 int32 initializer + per-channel DQ |
| Linear / Gemm | ❌ 不量化 | annotate_bias 里 linear 行被注释(上游原样保留) |
| ConvTranspose | ❌ 不量化 | 不在 annotate_bias 匹配列表内 |
| reuse_conv 系列 demo | ❌ 显式关闭 | AXQuantizer(..., annotate_bias=False),原 demo 既有选择 |

另:config.json 无 bias 配置通道(get_quantization_config 恒置 None),
唯一生效路径是 annotate_bias;resnet50 场景实际无量化 bias
(conv 全 bias=False、fc 为 Linear)——金标准 R6"bias 保持浮点"由此而来。

**待后端团队确认**:

1. bias 是否应**全部**量化(补上 Linear/Gemm 与 ConvTranspose)?
2. 派生方案(int32、scale=Sa×Sw、per-channel、直接加在累加器上)是否与
   pulsar2 的累加器语义一致?
3. reuse_conv 关闭 bias 量化是否合理,还是应统一开启?

**若定案全量化,改动点**:annotate_bias 解除 linear 注释并补
conv_transpose 匹配(注意 ConvTranspose 权重 per-channel 轴是 1,派生
spec 的 ch_axis 需对应核对);验证 = checker(R6 对 DQ bias 自动跳过,
需新增"bias 必须走 DQ"的正向断言)+ 全配置矩阵 + 双环境等价性回归。

## 2026-07-10:test_clamp 移植补记(P3 完结)

四个 clip/relu6 demo 脚本化移植,2.6 vs 2.10 数值输出逐字符一致(4/4)。
新发现并修复:**函数型 torchlib 算子的内联缺口**——optimize=False 下
hardtanh(relu6)以未内联函数节点 `aten_hardtanh`(pkg.onnxscript.torch_lib 域)
存活到 sim(onnxslim 不做函数内联),pulsar2 不认识 ONNX functions;
`utils_2_10/train_utils.dynamo_export` 已加 `onnx.inliner.inline_local_functions`
(独立内联,不触发 QDQ 常量折叠),sim 恢复标准 `Clip`,与 2.6 形态一致;
无函数模型不受影响(TinyNet 链路回归全绿)。

## FAQ:2.10 的 raw onnx 为什么比 2.6 的"杂乱"?

正常现象,是 optimize=False 的代价(不关会丢权重量化,见下文问题 1)。
以 resnet50 4w4f 为例:2.10 raw 636 节点(Constant×240 未收编 + 权重 zp
Cast×54 未折叠)vs 2.6 raw 330 节点(导出时 optimize() 已清理)。清理只是
推迟到 sim 阶段(gs.fold_constants + onnxslim,不碰 QDQ):**两侧 sim 均为
324 节点、零杂项、结构一致**。看图/交付/对比一律用 `*_sim.onnx`,raw 仅是
管线中间产物。

## 检查器用法

```bash
P=/home/heqi/miniforge3/envs/torch2.10/bin/python
# 规则检查(raw 导出)
$P env_check/check_onnx_structure.py --model xxx.onnx --ort
# simplify 后的模型(跳过 metadata 规则,Cast 从 WARN 升 FAIL)
$P env_check/check_onnx_structure.py --model xxx_sim.onnx --sim
# 与金标准基线做结构对比(同网络才有意义)
$P env_check/check_onnx_structure.py --model xxx.onnx --baseline env_check/baselines/resnet50_qat.profile.json
```

退出码非 0 = 存在 FAIL,可直接接 CI。

## 2026-07-09 检查结论

金标准(torch2.6 管线)特征:opset 21 / IR 10;权重 int8 initializer + per-channel
DequantizeLinear(零点全 0 对称,唯一消费者 Conv/Gemm);激活 U8 per-tensor Q→DQ
成对;bias 保持 float;QDQ 节点带 namespace/fx_node metadata。

torch2.10 复现同样结构时发现 **3 个真问题**:

1. **权重 DQ 链被折叠(R4b,阻塞级)**
   torch 2.9+ `torch.onnx.export` 默认 `optimize=True`,新版 onnxscript(0.5.4)的
   常量折叠会把「int8 权重 + DequantizeLinear」折成 float 权重 —— 图检查一切正常
   但权重量化信息全丢。`utils/train_utils.dynamo_export` 里的 `onnx_program.optimize()`
   同样触发。**迁移时导出必须 `optimize=False`**,折叠交给后面 gs.fold_constants +
   onnxslim(它们不折 QDQ)。
2. **simplify_and_fix_4bit_dtype 失配(R9 关联,阻塞级)**
   QDQ 节点 metadata 里 namespace 的 target 格式从
   `quantized_decomposed.quantize_per_tensor.default` 变成
   `torch.export._trace...Wrapper/quantize_per_tensor`,
   `utils/quant_utils.py` 的 `namespace.split(": ")[1]` 解析直接 assert。
   需按新格式(取 `/` 后缀)适配。
3. **simplify 管线抬高 IR version(R1-ir)**
   现存 `resnet50/resnet50_qat_sim.onnx` 的 ir_version=12,ORT 1.23 拒载
   (max 11),老 pulsar2 同样有风险。后处理 make_model 后应回写
   `model.ir_version = 10`。

好消息:`optimize=False` 的 torch2.10 导出 12/13 全过(1 个预期 WARN:per-channel
零点 Cast,sim 阶段会折掉);自定义 `quantized_decomposed::dequantize_per_channel`
的 torchlib 注册在 onnxscript 0.5.4 上仍然生效;metadata_props 仍在(只是格式变了)。

## 2026-07-09 P0 完成:三个问题已在 `utils_2_10/` 修复

`utils_2_10/` 为 torch2.10/torchao 平行实现(`utils/` 原样保留 = torch2.6 版,
双环境共存;包名用下划线因 Python 包名不能含点):

1. 权重 DQ 折叠 → `utils_2_10/train_utils.dynamo_export` 显式 `optimize=False`;
2. metadata 失配 → `utils_2_10/quant_utils.py` 改从 fx_node 提取 target;
3. IR 抬高 → simplify 保存前回写 `ir_version=10`。

另:注解器已按 2.10 适配(avgpool2d/layernorm/groupnorm/concat 改 aten 直匹配,
gru/mha/remove_reused_bn_param_hack 加显式 NotImplementedError 守卫)。

P0 验收结果(TinyNet + AXQuantizer(minimum/config.json)):raw 13 项 0 FAIL
(1 预期 WARN),sim 12 项全 PASS,ORT 推理正常;带 bias 的 conv 正确出现
int32 per-channel bias DQ(annotate_bias 语义保留)。

## 2026-07-09 P1 完成:minimum demo 双轨打通,与 2.6 金标准逐项一致

- 新建 `minimum/minimum_demo_2_10.py`(2.6 原版不动),utils_2_10 + torchao +
  `torch.export.export` 全链路跑通,产物 `minimum_*_2_10.onnx`;
- checker:raw 13 项 0 FAIL(1 预期 WARN),sim 12 项全 PASS(ir_version=10);
- **头对头**:torch2.6 env 重跑原版 demo 生成金标准(baselines/minimum_qat*),
  `--baseline` 对比 **sim vs sim 16/16 全项一致**;raw vs raw 仅差
  1 Cast + 5 Constant(2.10 关 optimize 的预期残留,sim 阶段被 fold_constants
  清掉,不影响最终产物);
- 新增 2.6→2.10 差异(已入 plan 改动 3):**训练态 BN 模型不能直接 dynamo 导出**
  (buffer 突变报 `b_bn_running_mean ... getitem_3`),float 参考模型须用
  eval 态深拷贝导出;WIP resnet50/train.py 对此包 try/except pass 属吞异常,
  P2 迁移时改正;
- 备注:torch2.6 env 缺 onnx-graphsurgeon,已经清华源补装(0.6.1)。

## 2026-07-09 P2(fake_data 部分)完成:resnet50 结构回归通过

- 新建 `resnet50_2_10/train.py|test.py`:**双环境单脚本**(按 torch 版本自动切
  utils/utils_2_10),`--data fake|cifar10`;产物带 `_2_6/_2_10` 后缀;
- fake_data + `config_4w4f_all.json` 短训导出:raw 13 项 0 FAIL,
  **sim 对金标准基线(resnet50_qat_sim.profile.json)5 项全一致**;
- 新发现(详见 plan P2 节):2.10 QAT 捕获需
  `Dim("batch", min=1, max=N)` 动态 batch + 捕获样例 batch>=2(0/1 特化);
  ORT 拒载 4w4f sim(`MaxPool` 不接受 uint4)经对照实验证实为金标准同款行为,
  非迁移回归——**4bit sim 的 `--ort` 校验跳过,数值对齐在 raw 上做**;
- ✅ **等价性对照通过**(CIFAR-10,同 seed/超参,2.6 与 2.10 并行):
  注解计数同为 128;逐 batch loss 锁步(逐点差 ≤0.044 无发散);
  eval top1 差 2.8pt@n=640(统计噪声内);两侧 sim 结构 16/16 全项一致。
  **P2 结论:resnet50 上 2.10 迁移与 2.6 结构、训练行为等价。**
  日志:远程 /tmp/qax_eq_26.log、/tmp/qax_eq_210.log(重启即丢,结论已录此处)。
- ✅ **完整 1 epoch 最终精度对照**(2026-07-10,全量训练集 + 全量 1 万张 test):
  末尾 running loss 0.362(2.6) vs 0.361(2.10);
  **test top1 94.72 vs 94.97(Δ0.25pt,噪声级,2.10 略高),top5 双侧 99.96**
  → 50 步短训时的 2.8pt 差异确证为"欠训练 + 小评测集"噪声,精度等价闭环。
  (备注:等价性 50 步 checkpoint 已备份为 checkpoint/*.eq50bak.pth,
  本轮全 epoch 产物覆盖了同名 checkpoint/onnx。)

### 两份 sim.onnx 逐节点深对比(2026-07-10)

对象:`resnet50_2_10/resnet50_qat_{2_6,2_10}_sim.onnx`(等价性运行产物,
同 config.json/seed/数据,各自独立训 50 步)。
环境:2.6 侧 = envs/torch2.6(torch 2.6.0+cu124,onnx 1.17.0,onnxscript 0.4.0,
onnxslim 0.1.48);2.10 侧 = envs/torch2.10(见 env.md)。

**结构完全一致**:

- 拓扑序算子序列 **324 节点逐位置相同**(非仅直方图相等);
- ir=10 / opset ai.onnx:21 / 输入输出签名(`x:FLOAT[1,3,224,224]`→`output:FLOAT[1,10]`)相同;
- 202 个 Q/DQ 类型逐位置一致;int 常量被节点引用总次数两侧相等(256)。

**三处差异,均无害**:

1. initializer 数 265 vs 272:常量去重程度不同(2.6 共享 100 个/349 次引用,
   2.10 共享 96 个/338 次;全零 int 常量 8 vs 17)。引用总数相等 → 纯存储形式差异;
2. 量化参数数值漂移(非结构):scale 相对差中位数 5.8%/最大 130%,zp 有差 36/202
   ——两侧独立训练 50 步的 observer 统计漂移所致(已被下面交叉实验证死);
3. 节点命名风格:2.6 `node_DequantizeLinear_1` vs 2.10 `node_dequantize_per_channel`
   (保留 fx 目标名)。pulsar2 按图结构消费不受影响;**按 onnx 节点名定位的
   下游脚本需留意**。

### 交叉实验:同一 checkpoint 两体系导出,qparam 逐 bit 一致(2026-07-10)

方法(`resnet50_2_10/cross_export.py`):把 **2.6 训练的
`last_checkpoint_2_6.pth`** 分别在 torch2.6(torch.ao)与 torch2.10(torchao)
环境里加载 → convert_pt2e → 导出 sim,对比量化参数。结论:

1. **checkpoint 跨体系完全兼容**:1216 个键 strict 模式直接载入
   torchao prepared 模型,零改动;
2. **量化参数逐 bit 一致**:202 个 Q/DQ 的 (scale, zero_point) 按值多重集合
   完全一一对应(md5 级);等价性运行里 5.8% 的 scale 漂移由此确证为
   独立训练的数值漂移,与量化计算无关;
3. 唯一现象:约 8 个权重 DQ 在拓扑序中的相对位置互换(resnet 并行分支的
   平序不同),按位置对齐比较时会出现假差异——**跨版本比较应按值/按消费者
   匹配,不能只按拓扑位置**。

**至此 P2 证据链完整:结构逐节点一致 + 训练行为锁步 + qparam 计算逐 bit
等价 + checkpoint 互通。**

### 全配置矩阵复测(2026-07-10)

resnet50 下 5 份量化配置全部在 2.10 复测(fake_data,seed 42,双环境对照,
产物 `resnet50_2_10/resnet50_qat_{2_6,2_10}_<cfg>[_sim].onnx`):

| 配置 | 语义 | 注解数(两侧同) | sim 对比结论 |
|------|------|--------------|-------------|
| config.json | 全局 U8/S8 | 128 | 16/16(等价性运行+交叉实验) |
| config_4w4f_all.json | 全局 4bit | 128 | 基线 5 项一致(vs 6/23 金标准) |
| config_16f.json | U8 + U16 region | 128 | **16/16 全项一致**,U16 精确落在 8 条边 |
| config_4w4f.json | U8 + U4 region | 128 | 结构一致;act_q 差异为 **2.6 侧 bug** |
| config_fp32.json | U16 + FP32 region | 124 | 基线 5 项一致(经 2 处修复) |

全配置 sim 逐项总表(`/tmp/qax_matrix_cmp.py` 产出,2026-07-10):

| 配置 | 节点数 | 拓扑序 | 直方图 | 权重量化 2.6 \| 2.10 | 激活量化 2.6 \| 2.10 |
|------|-------|--------|--------|---------------------|---------------------|
| config | 324/324 | 逐位一致 | 一致 | S8:54 \| S8:54 ✓ | U8:74 \| U8:74 ✓ |
| 4w4f_all | 324/324 | 平序不同 | 一致 | S4:54 \| S4:54 ✓ | U4:74 \| U4:74 ✓ |
| 4w4f | 324/324 | 平序不同 | 一致 | **S4:23,S8:31 \| S4:5,S8:49 ✗** | **U4:55,U8:19 \| U4:8,U8:66 ✗** |
| 16f | 324/324 | 平序不同 | 一致 | S8:54 \| S8:54 ✓ | U16:8,U8:66 \| U16:8,U8:66 ✓ |
| fp32 | 319/319 | 平序不同 | 一致 | S8:53 \| S8:53 ✓ | U16:4,U8:67 \| U16:4,U8:67 ✓ |

(拓扑"平序不同"= 并行分支输出顺序不同,节点数/直方图/连接关系一致,属无害,
比较须按值匹配;config 对是 cifar 等价性产物,恰好逐位一致。)

关键发现:

1. **2.6 管线混合 4bit 的 zp 污染 bug**(实锤,含机制,激活/权重两侧都有):
   2.6 export 时 optimize() 去重合并相等 zp 常量(74 个激活 Q 只剩 22 个独立
   zp,2 个共享 zp 覆盖 54 节点),simplify 按名字标 4bit → 激活侧 54 个 U8
   被误标 U4(55 vs 配置本意 8);权重侧同理,等形全零 zp 向量被合并 →
   **23 个权重被标 S4(配置本意只有 conv2d_1..5 共 5 个)**,2.10 恰为 5 ✓。
   2.10 管线 optimize=False,标记先于 slim 去重且去重不跨 dtype → 两侧均
   精确命中配置意图。**跨版本对比时 2.6 的混合 4bit 产物不能作为金标准**。
   污染源精确定位(2_6_4w4f_sim.onnx):共享常量 **`val_5`(zp=0 标量,
   52 个 Q 引用——ReLU 后激活 zp 均为 0 被 optimize 合并,其中仅 5 个属
   配置本意,47 个被污染)**;另一共享常量 `val_21`(zp=7,2 个 Q)是两个
   本意 U4 的 add 输入间无害共享。
   归因实验(raw 层面对账,2026-07-10):simplify_and_fix_4bit_dtype 的
   metadata **判定准确**(2.6 raw 判定 21 节点=配置本意),但其按张量名翻转
   dtype 的落笔方式遇共享常量即扩散——2.6 raw 里 21 个本意节点只涉及
   18 个名字(已共享),标记波及 133 节点、污染 112;同一函数喂 2.10 raw
   (optimize=False,zp 独立,共享常量 0 个)则判定 23、波及 23、污染 0。
   → 责任链 = 2.6 export 的 optimize() 去重(引入共享) × simplify 按名
   标记(无共享防御),两环相乘;2.10 管线已从第一环根治。若需给 2.6 老
   管线打补丁:翻转前查 zp 引用计数,共享时先复制拆分私有常量。配置本意的 4bit 边(2.10 侧点名):
   U4 激活 8 个(maxpool 前级 ×1、conv2d_1&4 共享 ×1、conv2d_2/3 ×2、
   conv2d_5&add_1 ×1、残差 add ×3),S4 权重 5 个
   (dequantize_per_channel_2..6 → conv2d_1..5);
2. FP32 区域揭出两个 2.10 适配点(已修):convert 后需
   move_exported_model_to_eval(未量化 conv+BN 不折叠,训练态 BN 导不出,
   train.py 两环境同改);零 bias 的 Expand(CastLike) 残链无法常量折叠
   (utils_2_10/quant_utils 新增 `_castlike_to_cast` pass,与 2.6 optimize
   产物对齐);
3. checker R4b 在 FP32 区域 conv 上两侧**对称误报**(conv 读共享 DQ 输出、
   权重故意保持 float),属已知例外;
4. regional module_names(conv2d_1..5、add__5、add__9)在 2.6/2.10 图中
   命名完全一致(已探针验证),混合精度配置无需改名。
