# 消融实验配置文件

## 三个创新模块

1. **GT-IGU (Geometry-Texture Interaction Gating Unit)**
   - 位置: `pointnext.py` 的 `PointNextEncoder`
   - 控制: `encoder_args.use_gt_igu: True/False`
   - 作用: 在 Encoder 输出层解耦几何和纹理特征，通过门控机制自适应融合

2. **TSI-Loss (Texture Smoothness Invariance Loss)**
   - 位置: `custom_innovations.py` 的 `TextureSmoothnessInvarianceLoss`
   - 控制: `tsi_loss_args` 配置块
   - 作用: 约束同类物体内部特征对纹理变化的不变性

3. **STS + L_cons (Structural Texture Swapping + Consistency Loss)**
   - 位置: `custom_innovations.py` 的 `apply_semantic_texture_swapping` + `TextureConsistencyLoss`
   - 控制: `use_consistency_loss: True/False`
   - 作用: 数据增强(打断纹理-语义伪相关) + 特征一致性约束

## 消融实验矩阵

| 配置文件 | GT-IGU | TSI-Loss | STS+L_cons | 说明 |
|---------|--------|----------|------------|------|
| `exp0-baseline.yaml` | ❌ | ❌ | ❌ | Baseline |
| `exp1-gtigu-only.yaml` | ✅ | ❌ | ❌ | 仅 GT-IGU |
| `exp2-tsi-only.yaml` | ❌ | ✅ | ❌ | 仅 TSI-Loss |
| `exp3-sts-only.yaml` | ❌ | ❌ | ✅ | 仅 STS+L_cons |
| `exp4-gtigu-tsi.yaml` | ✅ | ✅ | ❌ | GT-IGU + TSI |
| `exp5-gtigu-sts.yaml` | ✅ | ❌ | ✅ | GT-IGU + STS |
| `exp6-tsi-sts.yaml` | ❌ | ✅ | ✅ | TSI + STS |
| `exp7-full.yaml` | ✅ | ✅ | ✅ | Full (Ours) |

## 运行命令

```bash
# 在 PointNeXt 根目录下运行
cd /home/LIANGYudong_2023/PointNeXt

# 运行消融实验 (以 S3DIS 为例)
python examples/segmentation/main.py --cfg cfgs/ablation/exp0-baseline.yaml
python examples/segmentation/main.py --cfg cfgs/ablation/exp1-gtigu-only.yaml
python examples/segmentation/main.py --cfg cfgs/ablation/exp2-tsi-only.yaml
# ... 依次运行其他配置
```

## 评估指标

1. **标准分割指标**: mIoU, mAcc, OA
2. **鲁棒性指标**: 
   - Texture Robustness (TR): 语义冲突测试中正确识别几何类别的比例
   - Texture Deception Rate (TDR): 被纹理欺骗的比例
3. **效率指标**: 参数量、推理时间
