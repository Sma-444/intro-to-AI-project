# 成熟视觉语言技术与课程内的Transformer试验

调研与试验日期：2026-11-20。

工业界已经有能直接生成图像描述的预训练模型。建立一个可用原型通常可以复用这些模型；但在特定数据分布上稳定生成准确描述、控制幻觉、延迟和成本，仍然需要评估。不能从“行业已有方案”推出“只用12,000条训练样本，从零训练文本模块就能轻易达到某个相似度”。

## 本次核对的官方资料

| 项目 | 已核对的功能/设计 | 对课程的适用性 |
|---|---|---|
| [Salesforce BLIP-2](https://github.com/salesforce/LAVIS/blob/05cb670f47296522e24e7727e366b4a997d28582/projects/blip2/README.md) | 组合预训练视觉模型与OPT/FlanT5等语言模型，提供zero-shot及COCO-style captioning路径 | 是成熟方案，但完整预训练图文模型和文本模型不允许作为本次生成系统使用 |
| [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL/blob/96588727e44c78b25ba03ea03b8e12f7e64fd0da/README.md) | 图文理解与生成；DeepStack融合多层视觉特征；提供2B等规模的模型 | 可参考空间信息与多层特征建模思想，不能直接使用完整权重或其生成描述训练我们的模型 |
| [SmolVLM](https://github.com/huggingface/smollm/blob/438be18dae658a638f3a6976fdf5190bb6d68619/vision/README.md) | 文档所述版本用SmolLM2-1.7B-Instruct作为语言骨干，支持captioning与端侧应用 | “小型”仍包含预训练语言模型，不能因为更轻量就忽略课程限制 |

来源记录见 `research/modern_sources.json`。这些是核实过的代表性项目，不是宣称已穷尽2026年所有最新模型，也没有把官方跨数据集的成绩换算成本课程MiniLM分数。尝试的Microsoft Florence-2 GitHub仓库路径返回404，未据该路径作技术结论。

本次只读取公开说明，没有下载完整VLM权重、调用外部图像描述服务或用外部描述训练。LoRA和量化也不会改变预训练文本模型不被允许这一事实。

## 更适合本课程的现代视觉预训练方向

进一步核对了Meta的两个官方仓库，来源记录为 `research/modern_visual_sources.json`：

- [DINOv2](https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/README.md)：官方说明基础视觉模型在142M张图上做无标签/标注预训练；ViT-S/14 distilled约21M参数，并提供backbone-only权重。只采用纯视觉backbone，不使用仓库另外提供的dino.txt语言对齐模型。这是一条与课程允许“预训练图像编码器”相符、体积也有希望控制在200MB内的路线。
- [DINOv3](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/README.md)：较新的视觉基础模型，官方列出约21M参数的ViT-S/16与高分辨率dense特征，并已支持Transformers/timm加载。官方权重入口与许可证不同于DINOv2，后续采用时需按官方提供的方式获取并记录权重来源。

这一方向比“直接使用完整BLIP/Qwen权重”更适合课程：保留从零训练的文本模块，只替换允许预训练的视觉模块。以ViT-S/14在224输入下的16×16 patch token为例，空间token多于ResNet50的7×7网格，但注意力计算和显存也会上升；更多token不保证语义质量更好。

如果本轮ResNet50+Transformer达到稳定平台，下一项优先比较 **DINOv2 ViT-S/14纯视觉编码器 + 同一小型Transformer解码器**，保持词表、解码与训练预算一致，并为视觉微调设独立小学习率。DINOv3-S/16可作为更进一步的候选。这里尚未下载、运行或测量这两者在本数据上的分数，21M是官方近似参数量，不是我们已序列化的完整caption模型大小。

## 实际实现的候选

`transformer_model.py`：

- ResNet50，ImageNet1K V2视觉预训练权重。
- 224像素输入产生7×7空间特征，投影到256维。
- 二维正弦位置编码，1层4头视觉self-attention模块。
- 2层4头Transformer文本decoder，FFN1024，pre-LayerNorm，dropout0.2。
- 文本词表仅由训练描述建立，共1417个token；全部文本嵌入与decoder从零训练。
- 显式causal mask与padding mask，teacher forcing下masked CE，label smoothing0.05。
- 参数量27,666,377。相较先前预算多出的归一化层等小差异已计入实际模型。

`train_transformer.py`：

- 全部12,000训练对，全部2,500验证对，seed2205，batch24。
- AdamW：decoder基础LR3e-4，视觉最后stage基础LR2e-5，weight decay1e-4。
- 冻结CNN前两轮，第3轮开始微调layer4；所有视觉BatchNorm统计量保持冻结。
- warmup一轮后cosine衰减，AMP，梯度裁剪1.0。
- 最多16轮，验证分数4轮不提高即早停。每轮统一用beam3、alpha0.6、最多24步选择checkpoint。
- MiniLM只计算验证评分，不用于梯度、奖励、蒸馏、候选句重排或生成特征。
- `best_model.pt`包含全部推理权重并检查严格小于200,000,000 bytes。
- `last_training.pt`另存optimizer、scheduler、scaler、随机状态、DataLoader generator状态和早停计数。只支持epoch边界恢复，worker每轮重建；CUDA计算仍可能有数值不确定性。

这是一组“Transformer＋相应训练配方”的可用性试验，同时改变了模型结构、BN策略、调度器和checkpoint选择方式。若提升，首先说明该整体候选有效；不能把所有提升单独归因于Transformer结构。旧AoA最佳模型为0.586902320的beam3结果，是已有候选比较点，不是这次所有变量匹配的消融control。

## 运行前验证

`check_transformer.py` 在24张真实训练图上通过：

- 改变未来token不影响过去位置的logits；追加padding不改变有效前缀输出。
- 三步AMP训练loss为7.3175、7.0124、6.7138，梯度有限，视觉关系层梯度非零，CNN已解冻层与decoder参数实际更新。
- 视觉BN的running statistics未变化。
- beam batch独立性、完整teacher-forcing与逐步前缀预测等价。
- state_dict重载后的float32 logits完全一致，完整smoke checkpoint为111,032,331 bytes。
- 旧AoA checkpoint在共享模型加载器中严格加载，真实图片生成结果与原构造方式一致。

这些检查证明实现的基础行为正确，不证明生成质量或0.7目标已经达到。正式成绩以训练完成后独立重载验证为准。

训练到第8轮时发生一次FP16梯度溢出。最初的严格梯度裁剪检查在GradScaler处理前抛出异常，该批次没有执行权重更新。已修正为让GradScaler跳过溢出更新并减小scale，scheduler只在有效更新后前进。随后成功加载第7轮保存的模型、optimizer、scheduler、scaler及随机状态恢复训练。原始日志为 `console.log`，恢复日志为 `console_resume.log`，原训练源码和修正后的恢复源码均保留在run目录。这是数值稳定性修复，不是重新挑选数据或改变模型结构。

## 使用方式

在项目根目录的PowerShell中设置环境：

```powershell
$env:PYTHONPATH=(Resolve-Path 'captioning/.env/site-packages').Path
$env:TORCH_HOME=(Resolve-Path 'captioning/.weights').Path
```

训练新目录：

```powershell
python captioning/train_transformer.py --output-dir captioning/runs/resnet50_transformer_new
```

恢复该目录：

```powershell
python captioning/train_transformer.py --output-dir captioning/runs/resnet50_transformer_new --resume
```

恢复时必须保持原参数（包括output-dir字符串形式）一致，不能通过resume悄悄延长cosine训练总步数。

独立验证：

```powershell
python captioning/evaluate_saved.py --checkpoint captioning/runs/resnet50_transformer_v1/best_model.pt
python captioning/infer.py --checkpoint captioning/runs/resnet50_transformer_v1/best_model.pt --image-dir Validation --output captioning/runs/resnet50_transformer_v1/inference_verified.csv
```

新Transformer checkpoint携带beam3配置，推理默认读取该配置；旧checkpoint未携带该字段时仍默认greedy。命令行可显式覆盖。本文中的验证命令只针对Validation，不使用Test进行选择。

## 结果

训练已完成，第15轮触发早停，最佳checkpoint来自第11轮。

| 比较项 | 验证平均余弦 | 完整推理权重 |
|---|---:|---:|
| 旧最佳ResNet50 + AoA-LSTM，beam3/alpha0.6 | 0.586902320 | 124,741,979 bytes |
| 新ResNet50 + Transformer，beam3/alpha0.6，独立重载 | **0.595902145** | **111,032,331 bytes** |

新候选绝对提升 **0.008999825**，模型文件减少 **13,709,648 bytes**。距离0.7仍差 **0.104097855**。这说明本次小型Transformer配方可行且在当前验证集上优于已有候选，但没有达到0.7，也没有多种子或完全匹配配方的消融证据，不能将其表述为Transformer结构单独带来的确定收益。

训练时最佳分数为0.596045971；独立重载为0.595902145，相差约0.000144。2500条描述中19条在这两种执行情境下不同，可能与训练启用cuDNN benchmark、AMP及计算路径差异有关，尚未逐因素归因。正文统一使用独立重载值，不声称训练与重载文本完全一致。

随后通过独立 `infer.py` 生成全部2500条验证描述，与独立重载评分CSV **逐条完全一致**；图片ID唯一、无遗漏、描述非空，检查见 `runs/resnet50_transformer_v1/inference_verification.json`。模型加载不需要重新下载预训练权重。基础模型检查、真实训练和恢复、独立评分及命令行推理均完成。

完整结果与记录：`result.json`（训练时）、`verification.json`（独立重载）、`training_history.csv`、`reload_caption_differences.json`。最佳权重为 `runs/resnet50_transformer_v1/best_model.pt`。根目录的旧提交权重和测试预测未替换，本轮不使用Test进行选择。

下一步优先考虑纯视觉DINOv2 ViT-S/14配合同一Transformer的对照试验；它属于尚未实施的后续方向。避免仅把训练轮数增加或继续扩大文本decoder当作通往0.7的可靠方法：本轮第11轮后训练loss继续下降，但验证分数没有再超过最佳值。
