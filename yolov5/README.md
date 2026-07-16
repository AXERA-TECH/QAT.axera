# yolov5 QAT 参考补丁

本目录不含可独立运行的工程：`train.py` 是给
https://github.com/ultralytics/yolov5/tree/master 检出目录用的**参考补丁**
（在其原版 train.py 基础上加入 QAT 流程），`config.json` 是配套的量化配置。

## 使用步骤

1. 检出 ultralytics/yolov5，把本目录的 `train.py`、`config.json` 拷入其根目录；
2. 把本仓库的 `utils/` 整个包拷为 yolov5 根目录下的 **`qat_utils/`**
   （⚠️ 不能叫 `utils`——会与 yolov5 自带的 utils 包撞名；包内全部为相对
   import，改名拷贝不影响功能）：
   ```bash
   cp -r /path/to/QAT.axera/utils /path/to/yolov5/qat_utils
   ```
3. 注释掉 yolov5 的这一行（AMP 检查干扰 QAT）：
   https://github.com/ultralytics/yolov5/blob/master/models/common.py#L337
4. 环境：torch 2.10 按 QAT.axera 的 `requirements_2_10.txt` 配置，
   torch 2.6 按 `requirements.txt`——补丁经统一 API（`from qat_utils import ...`）
   双版本同一份代码；
5. 执行：
   ```bash
   python train.py --data coco.yaml --epochs 1 --weights yolov5s.pt --cfg yolov5s.yaml --batch-size 1
   ```

## 说明

- QAT 相关改动集中在 train.py 的量化器构造、`capture()` 图捕获
  （batch 维已按 torch.export 原生 `dynamic_shapes` 声明动态，改 batch-size
  无需动代码）与 `convert_pt2e` 导出段，可全文搜索 `qat_utils` 定位；
- 导出产物的结构体检可拷回 QAT.axera 用 `utils/check_onnx_structure.py`
  （用法与已知例外见 `README_2_10.md` 与 qat-check skill）。
