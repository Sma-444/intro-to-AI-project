# AIAA2205 图像描述生成项目交接

当前最佳：DINOv2-small + 4层Transformer，336输入，seed2206；完整Validation相似度 **0.690574467**，最佳epoch12，模型111.73 MB。尚未达到0.7。训练代码、历史实验记录和论文研究资料在captioning/。

## 接手步骤

1. 下载本仓库代码（git clone或Code → Download ZIP）。
2. 使用Python 3.11+运行 `python download_assets.py`，下载并校验继续训练所需的全部图片、MiniLM评分模型、DINO预训练权重和最佳实验的权重/恢复状态。Python3.14为原训练环境。
3. 如需所有历史模型、课件和旧提交产物，运行 `python download_assets.py --all`。
4. 环境安装、评分定义、完整实验表及续训命令见 [交接说明](handoff/完整本地交接说明.md)。该文档描述原本地包布局；**本仓库根目录就是文档中的01_可运行项目**，课程和旧提交资料在下载后handoff/下。
5. 原课程要求已另存 [COURSE_README.md](COURSE_README.md)。原机器依赖版本见handoff/原机器环境.json。

## 环境和验证

先安装适合GPU驱动的torch/torchvision（原版本2.11.0+cu128/0.26.0+cu128），再 `pip install -r requirements-portable.txt`。原版本不可用时选兼容组合并记录差异，先验证再训练。

```bash
python captioning/evaluate_saved.py --checkpoint captioning/runs/dinov2_transformer_336_decoder4_seed2206/best_model.pt
```

## 从最佳权重继续训练新阶段

```bash
python captioning/train_transformer_continue.py --backbone dinov2_small --image-size 336 --decoder-layers 4 --seed 2206 --init-checkpoint captioning/runs/dinov2_transformer_336_decoder4_seed2206/best_model.pt --encoder-lr 2e-6 --decoder-lr 3e-5 --freeze-epochs 0 --epochs 8 --patience 3 --batch-size 24 --output-dir captioning/runs/finetune_stage2_v1
```

这是交接新增的微调起点，已通过小批量更新/保存加载检查，但没有跑完该8轮配方，不能声称已有提升。原16轮实验已完成，直接resume不会增加训练预算。

大文件在[Release](https://github.com/Sma-444/intro-to-AI-project/releases/tag/handoff-2026-11-25)，支持单独下载；默认不下载历史模型及课件。文件清单与SHA256见assets-manifest.json。不上传Windows虚拟环境、下载缓存、重复的完整7.62 GB ZIP。

## 课程边界

Training用于拟合，Validation只用于选择/评分，Test只做最终推理。禁止外部图文训练数据、预训练文本decoder/完整VLM、以MiniLM输出训练模型。完整提交checkpoint小于200 MB。验证分数不是正确率或正式课程分数。旧提交文件不是当前最佳模型的预测，不能直接提交。正式四页报告尚未完成。
