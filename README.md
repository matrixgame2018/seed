# seed分割模块

还有很多参数可调节，我并没有精调参，可以自行调参，譬如将优化器换成sgd，然后T_0=2, T_mult=2,max_epoch=70。(效果应该会好于adamW)

## 当前实验结果
- backnone: efficientb6
- model: unet++
- img_size: 768
- dice-loss+softCrossBSE联合Loss
- optimize: adamW,SGD均可
- warmUpConsineScheduler

**特殊涨点tricks**：
- 模型融合
- TTA
- 注意力机制("scse"),仅Unet系列支持


**后续更多操作**：
- 换更大backbone
- 换模型(pspnet,deeplabplus,Unet等)
- 数据增强
- 图像裁剪，patch处理（可以参开nnUnet的做法进行优化）
- 多尺度训练/测试
- SWA，这是一种可以在模型训练完之后的复盘优化，我已经将具体的实现写在了deeplearning中，
可以通过调节swa_start控制swa在什么时间开启，swa一旦开启知道跑完所有的epoch才会停止。
预计通过swa技术可以实现近一个点的涨点。
- 另外除了”scse“结构外本框架还实现了中大最新提出的无参化注意力机制”simam“的在Unet系列的运用，
只需要修改train中seg_qyl的decoder_attention_type即可


## 数据处理
原始标注:

{
    0：背景
    1：前景
 }

```
├── satellite_data
│        ├── ann_dir
│        └── img_dir
├── satellite_jpg
│        ├── ann_dir
│        └── img_dir
```
## 代码运行说明
### 环境:
- torch>1.6,(因为使用了自动混合精度训练。如果<1.6，自行将混合精度训练那部分代码注释掉即可)
- segmentation_models_pytorch
- pytorch_toolbelt
### 运行
数据处理(可自行灵活处理)
```
python tif_jpg.py
python make_datasets.py
```
训练
```shell
python train.py
```
预测
```shell
python infer.py
```
infer里面use_demo=True,可以可视化一张预测图片，若为False,则为生成提交结果
