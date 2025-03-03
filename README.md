# QAT.axera

axera QAT demo
包含一个最小导出 demo 和一个 resnet50 训练 demo

---
## minimum export demo
```
python3 -m minimum.minimum_demo
```

---
## resnet50 train
```
# download imagenet dataset
cd QAT.axera
mkdir -p dataset/imagenet && cd dataset/imagenet
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar --no-check-certificate
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar --no-check-certificate
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz --no-check-certificate

# download resnet50 pretrained model
cd QAT.axera
wget -O resnet50/resnet50_pretrained_float.pth https://download.pytorch.org/models/resnet50-0676ba61.pth

# train
cd QAT.axera
mkdir -p resnet50/checkpoint
python3 -m resnet50.train

# test 
cd QAT.axera
python3 -m resnet50.test