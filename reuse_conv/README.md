# 分段定义模型，并将其中部分子模型循环调用

运行 <br>
python3 -m reuse_conv.train_resnet <br>
python3 -m reuse_conv.test_resnet <br>

分别定义 3 个子模型，将需要循环使用的模块单独定义为其中一个子模型； <br>
先将 3 个子模型配置好量化配置后，再在外部模型中将子模型链接起来，在外部模型上训练。 <br>
无法保存完整模型的 pth 和 onnx，只能分段保存
