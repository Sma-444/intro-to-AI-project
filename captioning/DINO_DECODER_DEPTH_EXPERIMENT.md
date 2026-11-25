# DINOv2 + Transformer：文本解码器深度对照

2026-11-22。目的：固定336输入后，检验将文本Transformer解码器从2层增加到4层是否改善验证相似度。增加层数使每个词经过更多轮因果自注意力、图像交叉注意力及前馈变换；它增加容量，但不保证改善泛化。

## 预先确定的实验

control：`runs/dinov2_transformer_336_v1`，2层decoder。candidate：`runs/dinov2_transformer_336_decoder4_v1`，4层decoder。

保持d_model256、4个attention heads、FFN1024、dropout0.2、1层视觉关系模块、336输入、DINO最后2个block微调、encoder LR1e-5、decoder LR3e-4、batch24、seed2205、最多16轮、patience4、warmup1epoch+cosine、AMP、label smoothing0.05、beam3/alpha0.6不变。使用同一预训练视觉权重，从头随机初始化文本模块。结构改变会改变随机初始化的抽样消耗；相同seed并不代表共享参数具有完全相同的初始张量。本轮是单种子配方对照，不能据此宣称跨种子稳定。

训练只使用12,000张Training及其描述；2,500张Validation用于选checkpoint和评分。课程MiniLM只作冻结验证评分；Test不参与。完整推理checkpoint必须小于200,000,000 bytes。

先检查四层的梯度、因果mask、padding、推理前缀一致性、beam批次独立性及checkpoint重载，再训练。训练后独立重载完整验证集，并逐条比较最佳预测。新方案没有提升则保留2层候选；本轮不连续扫描更多层数。

## 复现

项目根目录PowerShell：

```powershell
$env:PYTHONPATH=(Resolve-Path 'captioning/.env/site-packages').Path
C:/Python314/python.exe captioning/check_transformer.py --backbone dinov2_small --image-size 336 --decoder-layers 4
C:/Python314/python.exe captioning/train_transformer.py --backbone dinov2_small --encoder-lr 1e-5 --image-size 336 --decoder-layers 4 --output-dir captioning/runs/dinov2_transformer_336_decoder4_v1
```

恢复时使用相同参数并加`--resume`。旧实验默认decoder_layers=2；推理按checkpoint配置构造实际层数。

## 结果

预检查已通过：四层self-attention和cross-attention均有非零梯度，因果mask、padding、AMP更新、冻结参数、逐步推理前缀、beam批次独立性、严格重载和旧模型兼容性通过。24张训练图片的三步loss为7.39866、6.81392、6.50550。参数27,895,817，完整checkpoint111,731,393 bytes，预检查PyTorch分配显存峰值1,096,305,664 bytes。

已完成16轮，最佳为第13轮。独立重载与命令行推理均完成全部2500张Validation，结果如下：

| 模型 | 最佳epoch | 独立验证相似度 | 完整checkpoint |
|---|---:|---:|---:|
| 224输入、2层decoder | 11 | 0.675967336 | 103,290,761 bytes |
| 336输入、2层decoder | 15 | 0.686261475 | 103,290,761 bytes |
| 336输入、4层decoder | 13 | **0.689380348** | 111,731,457 bytes |

4层相对同分辨率2层提高 **0.003118873**，相对早期224候选提高0.013413012。仍未达到0.7。训练第9和13轮各有一次AMP自动跳过溢出更新，缩小scale后正常继续；第16轮loss1.0044，验证0.687427402，没有取代第13轮最佳权重。

训练最佳CSV、独立重载CSV及CLI推理CSV三者SHA256完全相同：`cf9d4859570740397ba6f0676679c840fc5310295e42fff508452eaaed9df172`。非空描述、2500个唯一ID、全覆盖和200 MB限制均通过。新checkpoint保存`cudnn_benchmark=True`，验证与推理读取该设置；当前硬件上结果一致不保证其他硬件/精度/批大小逐字相同。

逐图对照：930张得分提高、867张降低、703张不变；相似度>=0.7的图片从1257增加到1296，但<0.4的图片也从239增加到246。因此不能声称全面改善或已经修复动作/关系错误。配对bootstrap的平均差值95%区间约 **[-0.00251, 0.00886]**，包含0。该区间仅描述当前重复使用的验证集与选定checkpoint，不涵盖训练种子波动或模型选择偏差。

4层模型保留为当前最高测得分数候选，同时保留体积更小的2层control。本实验支持“加深后本次分数小幅提高”，尚不支持“4层稳定优于2层”。下一步应优先用预先固定的第二种子同时重跑两种深度，检查提升是否稳定，再考虑增加宽度或更复杂的视觉特征融合；不凭本次小差异继续盲目堆层。本轮没有执行第二种子实验。

结果与逐图诊断位于`runs/dinov2_transformer_336_decoder4_v1/`中的`verification.json`、`paired_comparison.json`、`paired_validation.csv`、`training_history.csv`。Test未参与训练/选择，根目录提交模型和测试CSV尚未更新。
