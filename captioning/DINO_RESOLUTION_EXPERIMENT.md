# DINOv2 + Transformer：336输入分辨率实验

2026-11-20。当前保留候选为224输入、DINO最后2个block微调的模型，独立验证相似度0.675967336。上一轮扩至4个block得到0.674344242，未替换原候选。本轮只研究增加输入分辨率的配方。

## 预定方案

- 336×336输入是patch14的整数倍，DINO空间token由16×16=256增加到24×24=576。
- 验证图像短边resize384后center crop336，保持原resize256/crop224的8:7比例。插值仍为bilinear；训练RandomResizedCrop的scale、ratio和颜色增强不变，输出从224变336。像素取整造成的细微裁剪差异不可完全避免。
- 更高分辨率增加裁剪区域内部细节，不会恢复被裁掉的图像边缘。它不是全图letterbox实验。早先ResNet+letterbox失败不能预判本轮结果。
- DINO官方位置embedding按输入尺寸插值，视觉关系模块二维位置编码按新网格计算；参数量保持25,788,937，完整权重仍约103.29MB。
- 其余沿用control：12,000 Training、2,500 Validation、train-only词表1417、seed2205、batch24、最多16轮、patience4、前2轮冻结视觉模块，第3轮起微调最后2个block和输出LayerNorm、视觉LR1e-5、decoder LR3e-4、AdamW、warmup1epoch+cosine、AMP、clip1.0、label smoothing0.05、beam3/alpha0.6/24步。
- 从相同视觉预训练权重和随机初始化文本模块重新训练，不从经验证集挑选的旧best checkpoint继续拟合。MiniLM仅作验证评分，Test不参与实验。

模型选择依据为完整验证集相似度；低于原候选则保留原权重。先完成这一项预先指定实验与独立核验，再决定后续方向，不连续扫描多个输入尺寸。即使超过0.7，也只代表当前反复使用的验证集，不能据此保证隐藏测试成绩。

## 实现与预检查

`get_transforms`增加`scaled_centercrop`模式，支持224/336；在224时输出与legacy模式逐像素相同。训练CLI增加`--image-size`，checkpoint保存实际输入尺寸与预处理，推理/独立评分共用同一变换。旧checkpoint和resume默认保持224。

24张真实训练图通过：预训练权重原样核对、576个空间token、未来词mask、padding、AMP梯度、冻结与解冻参数更新、逐步前缀等价、beam batch独立性、严格保存加载、旧模型兼容性。三步loss为7.3928、6.9123、6.5783；preflight PyTorch分配显存峰值1,010,879,488 bytes（不等同于驱动全部显存占用），checkpoint103,290,761 bytes。

## 复现

项目根目录PowerShell：

```powershell
$env:PYTHONPATH=(Resolve-Path 'captioning/.env/site-packages').Path
python captioning/train_transformer.py --backbone dinov2_small --encoder-lr 1e-5 --image-size 336 --output-dir captioning/runs/dinov2_transformer_336_v1
```

恢复使用相同参数并加`--resume`。新试验使用新目录，已完成目录不覆盖。

## 结果

已完成16轮，最佳为第15轮。独立重载验证相似度 **0.686261475**，比224输入的0.675967336提高 **0.010294139**。完整权重103,290,761 bytes。2500条重载描述与训练时最佳CSV的SHA256均为`72c5dc3768a116cda12f88bfb8e4a6b3382ef6bcc694eb7b68fa23f2753e5351`。

2026-11-22核验时发现：原独立推理脚本未开启训练使用的`torch.backends.cudnn.benchmark=True`，在FP16/beam下导致9条描述不同，分数0.685865462。仅将该后端设置与训练对齐后，2500条描述及分数全部恢复一致。默认后端的首次核验保存在`verification_default_backend.json`和`validation_predictions_default_backend.csv`。现已让验证/推理脚本读取该设置，新训练写入checkpoint；旧Transformer checkpoint按其训练行为使用True。这是运行配置修正，未改动权重或标签。该精度/内核敏感性意味着不能保证不同硬件上逐字一致。

336两层版本成为当前最佳候选，仍未达到0.7。模型选择反复使用了同一验证集，本结果不能保证隐藏测试分数。下一轮固定336输入研究文本解码器深度，见`DINO_DECODER_DEPTH_EXPERIMENT.md`。
