# 课件、GitHub 实现与下一步优化建议

调研日期：2026-11-20。结论：先统一实验与评估流程，用小规模消融验证视觉微调，再比较视觉关系模块和紧凑 Transformer。当前证据支持这些方向值得试验，不能据此保证验证相似度超过 0.7。

## 阅读范围与证据边界

- 提取了 `课件/`、`L02/` 共 18 份 PPTX 的逐页文本，重点阅读 RNN、NLP/Attention、Transformer、CNN、过拟合、Optimization in Practice 中相关内容。提取结果和原始路径见 `research/course_text/index.json`。
- 本轮依据幻灯片文本阅读，没有逐页渲染图示；嵌入图片中的公式不在文本提取范围内。没有把目录内 PDF、所有 notebook 或其他课件都算作已精读。
- 实际调用 GitHub 仓库搜索，查询 `image captioning flickr8k transformer language:Python`，并下载、检查 5 个相关仓库的关键源码。搜索结果中部分 ViT+GPT-2、BLIP 项目依赖本课程禁止的预训练文本/完整图文模型，未作为可直接采用的方案。
- 阅读了 4 篇论文的 arXiv 摘要页及相关官方实现；没有声称完成论文全文复现。下载的源码只作阅读，没有执行。
- `research/github_sources/manifest.json` 保存查询结果、访问时间、commit、源码 URL、SHA256 和 GitHub 识别出的许可证；`research/paper_sources/manifest.json` 保存论文与额外源码来源。一个猜测的 `configs/transformer.yml` 路径返回 404，没有用它支持任何结论。

## 当前基线：需要解释的差距

| 已完成实验 | 验证平均余弦 | 解码方式 |
|---|---:|---|
| ResNet18 + additive attention | 0.528876 | greedy |
| ResNet50 + additive attention | 0.577977 | greedy |
| ResNet50 + AoA-inspired gate | 0.582443 | greedy |
| ConvNeXt-Tiny + additive attention | 0.570801 | greedy |
| ResNet50 + AoA gate + letterbox224 | 0.576513 | greedy |
| 当前最佳候选：ResNet50 + AoA gate | **0.586902** | beam3，alpha=0.6 |

来源：`ARCHITECTURE_COMPARISON.md`、`FULL_IMAGE_EXPERIMENT.md`、`DECODING_EXPERIMENT.md`。当前最佳权重是 `runs/resnet50_aoa_v1/best_model.pt`，124,741,979 bytes。根目录提交用 `model/final_model.pt` 仍是旧 ResNet18，不能把它写成当前最佳候选。

从 0.586902 到 0.7 还差约 **0.113098**。同一权重修改解码仅提高约 0.004459，不能据此预测再调 beam 就能补齐差距。

旧 greedy 诊断中，540/2500 条低于 0.4，97 条有重复 trigram；这些统计属于 greedy，不能标成 beam 的诊断。文字对比提示主体/动作差异，但尚缺逐图人工核实，不能直接断言错误都由视觉编码器引起。训练损失下降、验证表现停滞符合过拟合或训练目标与生成评价不一致的可能性，不能单凭曲线确定原因。

## 课件如何对应到模型设计

页码为 PPTX 内 slide 序号。

| 课件与页码 | 阅读到的要点 | 对本项目的含义 |
|---|---|---|
| `Lecture 10 RNN` 第18、56、73页 | 图像描述是 one-to-many；梯度裁剪；LSTM缓解但不保证消除梯度问题 | 现有 CNN+LSTM 是合理基线；保留裁剪；不因模型“较旧”就推定必须替换 |
| `Lecture 11-12 NLP with RNN and Attention` 第34、47、49、55、86–87页 | Teacher forcing；图像到序列；单向量瓶颈；注意力提供对源输入的访问 | 保留空间特征；导出注意力用于诊断；单独评估 teacher forcing 下 loss 与自由生成效果 |
| 同课件第110–111页 | ImageNet CNN 迁移，小数据任务利用预训练特征 | 合法利用视觉预训练；逐层微调比从零训练视觉骨干更适合作为起点 |
| `Lecture 11-12 Transformer - Attention is all you need` 第5–6、22–23、26–30、33页 | 残差、位置编码、多头注意力、归一化、因果mask、cross-attention、随机初始化的可训练词嵌入 | 图像网格可作为cross-attention的K/V；文本作为Q；从零训练小型解码器符合要求 |
| `CNN` 第94–97、103页 | BatchNorm依赖batch统计；ResNet残差块 | 检查微调时BN状态。冻结BN统计量是待验证策略，不是课件宣称的必然最优策略 |
| `Lecture 5 - Overfitting and Model Metrics` 第7–8页 | 数据少、模型大、训练久可能过拟合；正则化、dropout、早停 | 先控制比较，再加层；当前单次种子的小增益需要复核 |
| `Optimization in Practice` 第2、12、27页 | 学习率影响收敛；Adam有适用性也有超参数敏感性 | 先测试学习率安排，不把换优化器本身当作解决方案 |

课件中的一般 NLP 预训练介绍不等于本项目允许预训练文本解码器；适用限制以项目 README 为准。

## GitHub 案例：具体借鉴什么

### 1. Keras 官方 Image Captioning：最贴近小数据的架构案例

[源码，固定commit](https://github.com/keras-team/keras-io/blob/f5891b3cd4df69f6bd5a3c26ea92310f3a973626/examples/vision/image_captioning.py)

源码使用 Flickr8K，每张图片 5 条描述，ImageNet EfficientNetB0 冻结提取空间特征，随后接一个 Transformer encoder block 和一个 decoder block。实现包括词/位置嵌入、文本因果mask、padding处理、masked CE、warmup 和 early stopping。关键代码：第257行 CNN，第271行 encoder，第560行附近早停与学习率安排。

可借鉴：**小型、随机初始化的 Transformer 可以构成同类教学任务的实现起点**。无需迁移到 TensorFlow，在现有 PyTorch 中实现相同思想即可。

不能直接照搬的细节：

- 它一图五描述，监督量与我们的数据不同，教程没有我们的 MiniLM 指标，无法推算我们能到 0.7。
- 第194行 `vectorization.adapt(text_data)` 使用切分前汇总文本。本项目必须只从训练描述建立词表，不能沿用该数据流程。
- 翻转/旋转/强色彩增强可能改变方向、动作或颜色语义，需要检查训练描述再选用。
- Keras `MultiHeadAttention(key_dim=...)` 的 `key_dim` 是每个头的维度，不能机械当成 PyTorch 的总 `d_model`。

### 2. sgrvinod 教程：用于改进现有 attention-LSTM 的实验实现

[models.py](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/blob/cc9c7e2f4017938d414178d3781fed8dbe442852/models.py#L40) · [train.py](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/blob/cc9c7e2f4017938d414178d3781fed8dbe442852/train.py#L183)

`fine_tune()` 支持 ResNet 多个后部stage；训练代码有encoder/decoder独立学习率、停滞后衰减、断点恢复，以及第186行 doubly stochastic attention 正则。

可借鉴：分阶段微调、恢复训练状态、暴露attention权重。该教程默认 `fine_tune_encoder=False`，不能把“支持多层解冻”说成默认训练就使用它。也不能照搬 ResNet101 和全部训练超参数到当前模型。

Coverage 正则放在后续小实验：短描述不要求覆盖所有背景，直接要求每个网格累计注意力接近1可能引入错误偏好；应先检查attention并正确排除padding时刻。

### 3. AoANet 官方实现：当前模型仍有未覆盖的结构

[AoAModel.py](https://github.com/husthuaan/AoANet/blob/94ffe176c3777bc54e53e9dd7b305b4046d91a70/models/AoAModel.py#L114) · [官方训练参数](https://github.com/husthuaan/AoANet/blob/94ffe176c3777bc54e53e9dd7b305b4046d91a70/train.sh)

官方代码有多头点积注意力、6层视觉refiner、AoA信息门控。decoder先更新attention LSTM，再以更新后的状态查询视觉特征。官方脚本采用1024维、8头、bottom-up特征、每图5描述，并分CE与SCST阶段。

当前 `train.py:145` 的模型先用上一时刻hidden做单头additive attention，再门控context并更新LSTM；没有视觉refiner。因此已有0.582443结果只验证了 **AoA-inspired gate**，没有验证完整AoANet。

推荐先借鉴 **1层视觉关系建模**，控制容量；完整6层、1024维结构和官方预训练caption权重不适合直接采用。官方数据说明也支持ResNet网格路线，不必为使用AoA就引入大型目标检测链路。

### 4. ImageCaptioning.pytorch：主要借鉴工程组织

[TransformerModel.py](https://github.com/ruotianluo/ImageCaptioning.pytorch/blob/4c48a3304932d58c5349434e7b0085f48dcb4be4/captioning/models/TransformerModel.py#L239) · [训练代码](https://github.com/ruotianluo/ImageCaptioning.pytorch/blob/4c48a3304932d58c5349434e7b0085f48dcb4be4/tools/train.py#L91)

该项目把caption模型和训练器分开，提供Transformer、Noam/warmup或plateau调度、scheduled sampling，以及模型和optimizer的保存/恢复。值得移植的是配置与接口组织。当前Python3.14/PyTorch2.11环境与旧项目不同，不建议直接安装并整体运行旧仓库。

Scheduled sampling改变训练目标与历史输入分布，有收益也有风险；应在稳定CE基线后单独测试。SCST若后续尝试，只能使用训练描述构造的允许奖励，不能用课程MiniLM作reward。

### 5. M² Transformer：后续结构研究，不作为第一轮完整复现

[官方说明](https://github.com/aimagelab/meshed-memory-transformer/blob/239100befa61d1e77ddf3b538810c83703d1e3a7/README.md) · [encoder](https://github.com/aimagelab/meshed-memory-transformer/blob/239100befa61d1e77ddf3b538810c83703d1e3a7/models/transformer/encoders.py#L51) · [decoder](https://github.com/aimagelab/meshed-memory-transformer/blob/239100befa61d1e77ddf3b538810c83703d1e3a7/models/transformer/decoders.py#L11)

源码对视觉特征做投影与LayerNorm，decoder通过门控融合多个encoder层的输出。论文摘要提出学习图像区域关系的多层表示。官方复现需COCO检测特征，README列出的预提取文件约53.5GB；该文件不是可直接用于我们的图片的特征，也不能作为额外训练数据引入。

可以从它借鉴视觉token之间交互的设计。暂不引入完整memory+meshed结构，以免在数据较少时同时增加多个难以归因的模块。

## 建议的重构顺序

### P0：统一模型、评估和checkpoint协议

建议拆出 `models/`、`data.py`、`decoding.py`、`metrics.py`、`checkpoint.py`，训练和独立推理共用同一配置构建模型，不通过重复的硬编码构建模型。

需要解决的实际问题：

1. 当前训练按greedy分数选checkpoint，最佳候选按beam3汇报。以后每个epoch用固定beam3/alpha0.6选checkpoint，并记录greedy作辅助诊断。所有候选必须采用同一规则；旧模型仍只按已保存权重汇报，不能推断未保存epoch的beam成绩。
2. 增加配置版本，完整保存输入变换、词表、维度、最大长度、解码策略、种子、来源权重与代码版本。保留旧checkpoint加载路径。
3. 分开 `best_inference.pt` 与 `last_training.pt`：前者包括整个生成模型并检查小于200,000,000 bytes；后者另存optimizer、scheduler、AMP scaler、epoch、early-stop计数、Python/NumPy/Torch随机状态和数据采样状态，以支持恢复。训练态文件不作为课程要求的推理权重提交。
4. 当前每次验证都重新加载MiniLM、编码参考文本。可在评估模块中缓存冻结参考embedding，并用标签/模型/tokenizer配置hash失效；只用于评分，不进入caption模型训练。
5. 保存逐图分数、完整预测CSV、耗时、显存峰值、完整配置，生成唯一候选清单，避免最佳实验与根目录旧提交文件混淆。

纯重构验收：固定已存checkpoint与相同精度/解码配置，对全部2500张验证图输出与已有CSV逐条一致；模型state_dict严格加载。浮点分数允许记录已有AMP级微差，不能用重构导致的新预测冒充旧基线。

### P1：以当前AoA模型检查视觉微调，避免混杂

代码证据：`train.py:381`只解冻最后stage，`train.py:388`把该stage切回train，因此其中BN统计量继续更新；`train.py:438`每个微调epoch给所有参数组乘0.8。

按下面顺序进行，固定seed2205、数据、词表、224预处理和解码：

| 实验 | 相对父实验只改变 | 需要回答的问题 |
|---|---|---|
| A0 | 用统一beam选择规则重训原配方 | 后续训练的可比control是什么？ |
| A1 vs A0 | 冻结所有视觉BN的running statistics，保留原affine参数可训练性 | batch24下统计量更新是否损害迁移？ |
| A2 vs A1 | 在layer4已开始微调后再解冻layer3，layer3起始LR可用5e-6，layer4仍2e-5 | 更深视觉适配是否改善主体/动作表达？ |
| A3 vs 当前较优A实验 | 仅调整scheduler，试warmup后cosine，保持相同总预算 | 固定逐epoch衰减是否过早限制学习？ |

A2需在optimizer中正确加入/保留layer3参数并记录解冻时刻。冻结BN统计量不同于冻结全部CNN，也不同于梯度累积；累积梯度不会让BN看到更大的单次batch。以上学习率是试验起点，未被证明最优。

每项都可能负收益，失败时保留父实验。不要把冻结BN、多层解冻、换scheduler、改分辨率一次性混合，然后声称确定某一项有效。

### P2：真正的结构候选

**候选B：当前512维AoA-LSTM前增加1层视觉关系模块。** 保持CNN与LSTM结构，空间token经位置编码、归一化、多头self-attention和残差后再交给原decoder。这是借鉴AoA/M²思想的小型改造，不称作完整AoANet复现。训练配方采用P1确定的control，注意力图只作诊断，不能当作因果解释。

**候选C：ResNet50 + 小型Transformer。** 建议起始配置：224输入、49个网格token、2048→256投影、二维空间位置编码、1层视觉关系encoder、2层文本decoder、4头、FFN1024。词嵌入和全部文本模块从零训练，cross-attention读取图像token，self-attention正确屏蔽未来和padding。初始dropout可设0.2，CE label smoothing沿用0.05。

`research/estimate_budget.py` 使用CPU随机初始化组件计数，未下载权重、未训练：

- 总参数 **27,665,353**。
- 包含ResNet50全部生成所需视觉权重及BN buffers的state_dict张量约 **110,874,316 bytes（110.87 MB）**。
- 这只是结构张量预算，最终序列化文件仍须实测；8GB显存的实际训练峰值和速度尚未验证。

可先用AMP、microbatch16进行短训练检查，需要时用梯度累积；检查mask、有限梯度、参数更新和保存加载后生成一致性，再投入完整训练。B与C同时改变的结构不止decoder，因此二者比较只能归因于整体候选；若要证明Transformer decoder本身有效，再做相同视觉前端和宽度的对应消融。

**若只选择一个大方向：选择“ResNet50空间特征 + 紧凑Transformer”的可替换模型接口，并保留现有AoA-LSTM作为control。** 它有课程结构依据，也有Flickr8K的可读实现，体积预算充足；是否比LSTM好必须由本项目实验决定。

### P3：有诊断支持再做的方向

- 对低/中/高分各固定抽样，查看原图、模型实际输入、参考描述与预测；分主体、动作、属性、关系、可接受替代表述标记。当前只有文字差异统计，尚未完成这一步。
- 若明显漏小目标，再单独测试320输入或layer3/layer4多尺度特征。224 letterbox失败不能说明320无效；插值把7×7变14×14也不会创造新的视觉细节。
- 若图像信息足够但词语表达仍不稳定，可尝试训练描述构造的对象/动作辅助目标。标注未提及不等于不存在，不能把所有未提及词都当可靠负类。
- Coverage、scheduled sampling、训练数据CIDEr的SCST均放在稳定CE模型之后；收益与本项目MiniLM指标不必一致。课程MiniLM始终只做验证评分。

## 实验预算与选择规则

先完成P0、A0/A1；只有结果支持才推进A2/A3，再比较B/C。新候选与父实验使用相同最大epoch/早停和解码预算，首轮沿用原16-epoch上限、patience4，延长训练时同步建立匹配control。

预先把验证提升0.003设为是否值得追加实验的工程门槛，而非统计显著性界限；0.000x量级先当不确定。候选胜出后至少对候选和匹配control补跑同一第二种子，比较均值与每次差值。逐图paired bootstrap可量化验证样本波动，但不能代替多种子，也不能消除反复使用验证集的选择偏差。

不把现有验证集重新划出的子集称为“从未见过”的独立确认集。测试图仍仅在方案选定后做最终推理。公开项目BLEU/CIDEr、teacher-forced token accuracy与我们的平均余弦不是同一指标，没有合法换算公式。正式成绩还依赖隐藏测试P和教师Q；0.7是目标，不是当前可承诺结果。

## 论文入口

- [Show, Attend and Tell，2015](https://arxiv.org/abs/1502.03044)：视觉注意力图像描述，摘要包含Flickr8K/Flickr30K/COCO实验范围；对照sgrvinod实现理解attention和coverage。
- [Attention on Attention，ICCV 2019](https://arxiv.org/abs/1908.06954)：查询与注意力结果的相关性门控；本次进一步核对了官方多头/refiner/decoder结构。
- [Meshed-Memory Transformer，CVPR 2020](https://arxiv.org/abs/1912.08226)：区域关系的多层表示与跨层解码融合；完整方法不作为第一轮。
- [In Defense of Grid Features for Visual Question Answering，CVPR 2020](https://arxiv.org/abs/2001.03615)：主任务是VQA，摘要提及迁移到captioning；支持继续研究grid方案，但其预训练条件和任务指标不构成我们必然达到某分数的证据。

本轮产出为调研材料、来源快照和结构预算；没有启动新训练，也没有将候选方案记为已取得新分数。
