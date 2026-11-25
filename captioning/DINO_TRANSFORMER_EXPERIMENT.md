# DINOv2-small + Transformer对照实验

日期：2026-11-20。目标：验证纯视觉自监督预训练编码器能否改进当前ResNet50+Transformer候选（独立验证0.595902145）。

## 数据和来源

使用课程12,000条训练图文对训练，2,500条验证对评分和选checkpoint。测试集不用于训练或选择；词表仅由训练描述建立。MiniLM只负责验证评分。

采用Meta纯视觉 `facebook/dinov2-small`，不是完整caption/VLM模型，也没有使用dino.txt或预训练文本decoder。下载revision为 `ed25f3a31f01632728cabb09d1542f84ab7b0056`；safetensors为88,249,960 bytes，SHA256为 `ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1`。完整来源记录在 `.weights/dinov2-small/provenance.json`。

## 模型与对照

| 设置 | 前一候选 | 本轮候选 |
|---|---|---|
| 视觉骨干 | ImageNet ResNet50 | 自监督DINOv2 ViT-S/14 |
| 224输入的空间特征 | 49×2048 | 256×384，排除CLS |
| 投影后维度 | 256 | 256 |
| 视觉关系模块 | 1层4头Transformer | 相同 |
| 文本decoder | 2层4头、FFN1024 | 相同，从零训练 |
| 视觉微调 | 第3轮起layer4，LR2e-5 | 第3轮起最后2个block和输出LayerNorm，LR1e-5 |
| 模型参数 | 27,666,377 | 25,788,937 |
| 完整推理checkpoint | 111,032,331 bytes | preflight实测103,290,761 bytes |

两者都使用相同的224裁剪/归一化、1417词表、seed2205、batch24、decoder LR3e-4、AdamW、weight decay1e-4、dropout0.2、label smoothing0.05、AMP、梯度裁剪1.0、一轮warmup后cosine衰减。解码固定beam3/alpha0.6/最多24步，最多16轮，patience4。

DINO内部包含预训练位置编码，本模型仍保留原视觉关系模块前的二维位置编码。DINO官方processor使用同样的均值、方差、resize256/center224，但默认插值为bicubic；本轮沿用已有对照的torchvision默认bilinear裁剪流程，以保持输入处理一致。未声称逐像素完全等于官方processor。

这是更换视觉骨干及其适配微调设置的整体候选比较，不是完全相同架构的单变量实验。DINO来自更大规模视觉预训练，不能把结果仅归因于ViT替代CNN。所有文本模块仍只从课程训练描述学习。

## 实现与检查

- `transformer_model.py`支持两个视觉骨干，decoder与beam接口共用；checkpoint包含完整DINO配置和全部权重，重载不需要下载任何视觉权重。
- `model_io.py`支持新DINO、已有ResNet Transformer、旧LSTM checkpoint。旧ResNet Transformer缺少backbone_name时自动使用原默认值。
- `train_transformer.py`支持显式backbone和encoder LR，训练状态与推理权重分开保存，保留epoch边界恢复。
- 加载时检查missing/unexpected/mismatched keys；随后逐项验证223个DINO预训练张量原样保留。Transformers5更改了attention投影的命名，校验时做明确名称映射，张量数值不转换。
- 24张真实训练图完成3步AMP训练，loss为7.4225、6.9208、6.5997；解冻视觉层和decoder更新，冻结层不变，梯度有限。
- 因果mask、padding、256个patch token形状、逐步与teacher-forcing前缀一致性、beam batch独立性、保存重载完全一致均通过。旧AoA生成兼容性和旧ResNet Transformer加载也通过。

预检查中的“BN frozen”通用项目在DINO本身没有BatchNorm时为空检查，DINO冻结参数是否不变另行验证。preflight显存值不是完整训练显存的保证。

## 复现

在项目根目录PowerShell中：

```powershell
$env:PYTHONPATH=(Resolve-Path 'captioning/.env/site-packages').Path
$env:PYTHONIOENCODING='utf-8'
python captioning/train_transformer.py --backbone dinov2_small --encoder-lr 1e-5 --output-dir captioning/runs/dinov2_transformer_v1
```

已存在训练目录不覆盖；恢复时使用相同参数并加 `--resume`。新实验改用新目录。独立验证/推理：

```powershell
python captioning/evaluate_saved.py --checkpoint captioning/runs/dinov2_transformer_v1/best_model.pt
python captioning/infer.py --checkpoint captioning/runs/dinov2_transformer_v1/best_model.pt --image-dir Validation --output captioning/runs/dinov2_transformer_v1/inference_verified.csv
```

## 结果

第15轮早停，选中第11轮。全部2500条验证对独立重载结果：

| 候选 | 验证相似度 | 完整推理权重 |
|---|---:|---:|
| ResNet50 + Transformer | 0.595902145 | 111,032,331 bytes |
| **DINOv2-small + Transformer** | **0.675967336** | **103,290,761 bytes** |

绝对提升 **0.080065191**，文件减少7,741,570 bytes。距离0.7仍差 **0.024032664**，尚未达到目标。

训练时最佳CSV、独立重载评分CSV、独立CLI推理CSV全部2500条描述逐条一致；ID唯一、无遗漏、描述非空。权重SHA256为 `33b1c1df58772402da743d51455aa29148c3ce11220975633e3bea4c049c5c2a`。推理只从完整checkpoint构造，不依赖下载目录里的预训练权重。最佳文件为 `runs/dinov2_transformer_v1/best_model.pt`。

逐图比较（相同验证参考和MiniLM处理）：

- 1521张相似度提高，735张下降，244张相等。
- 相似度低于0.4的图片从 **480减少到262**。
- 相似度至少0.7的图片从 **870增加到1236**。
- 对逐图差值做2000次paired bootstrap，条件95%区间约 `[0.07243, 0.08730]`。它只刻画当前这组已选择checkpoint的验证样本波动，不衡量训练随机种子差异，也不消除重复使用验证集的选择偏差。因此不能等价地宣称隐藏测试也会提高同样幅度。

逐图统计的CPU均值与评分脚本GPU均值存在约6e-8的求和数值差异，不影响上述结论。逐图详情为 `paired_validation.csv`，汇总为 `paired_comparison.json`；独立评分和推理核验为 `verification.json`、`inference_verification.json`。

训练日志记录第9、13轮各一次AMP overflow自动跳过更新并降scale，无中断。15轮累计记录约18.7分钟，不含下载、preflight和独立核验。训练loss从第11轮1.2185降到第15轮1.0983，但验证未再超过第11轮，简单延长epoch缺少支持。

本轮没有替换根目录旧提交文件，也没有运行Test选择模型。已有所有实验权重保留。

## 后续优化建议

先对当前262个低分样本进行固定抽样的原图/实际裁剪/预测对照，区分裁剪丢主体、动作/属性错误和参考表达差异。当前统计并未完成逐图人工视觉核实，不能把这些错误全部归因于视觉分辨率。

随后优先单独测试更深视觉适配：从最后2个block扩到最后4个block，对新增层采用更低LR，保持其他设置一致；另一个独立方向是更高分辨率的patch特征，必须同时控制输入裁剪策略。不要把加层、换分辨率、改变解码同时混合，也不要将0.024的剩余差距视为必然能通过这些手段补齐。
