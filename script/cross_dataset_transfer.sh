#!/usr/bin/env bash
# ============================================================
# 跨数据集迁移学习脚本
# 用法: 
#   1. 只加载encoder微调: bash script/cross_dataset_transfer.sh finetune_encoder
#   2. 只加载encoder测试: bash script/cross_dataset_transfer.sh test_encoder
#   3. 完整模型微调:      bash script/cross_dataset_transfer.sh finetune_full
#   4. 零样本测试:        bash script/cross_dataset_transfer.sh zeroshot
# ============================================================

MODE=${1:-finetune_encoder}
PRETRAINED_PATH=${2:-"log/scannet/pointnext-xl/pointnext-xl_ckpt_best.pth"}
CONFIG=${3:-"cfgs/s3dis/pointnext-xl-scannet-pretrain.yaml"}

echo "============================================================"
echo "跨数据集迁移学习"
echo "模式: $MODE"
echo "预训练模型路径: $PRETRAINED_PATH"
echo "目标数据集配置: $CONFIG"
echo "============================================================"

case $MODE in
    finetune_encoder)
        # 只加载encoder进行微调 (推荐: 解决类别数不同问题)
        echo ">> 正在加载encoder权重进行微调..."
        python examples/segmentation/main.py \
            --cfg $CONFIG \
            mode=encoder \
            pretrained_path=$PRETRAINED_PATH
        ;;
    
    test_encoder)
        # 只加载encoder直接测试 (需要已有训练好的S3DIS分类头)
        echo ">> 正在使用encoder进行测试..."
        python examples/segmentation/main.py \
            --cfg $CONFIG \
            mode=test \
            pretrained_path=$PRETRAINED_PATH
        ;;
    
    finetune_full)
        # 完整模型微调 (需要类别数一致，或者自动跳过不匹配层)
        echo ">> 正在加载完整模型进行微调..."
        python examples/segmentation/main.py \
            --cfg $CONFIG \
            pretrained_path=$PRETRAINED_PATH
        ;;
    
    zeroshot)
        # 零样本测试 - ScanNet模型直接在S3DIS上测试（类别自动映射）
        echo ">> 零样本跨数据集测试..."
        echo ">> 使用类别映射: ScanNet(20类) → S3DIS(13类)"
        python examples/segmentation/main.py \
            --cfg cfgs/s3dis/pointnext-xl-zeroshot.yaml \
            mode=zeroshot \
            pretrained_path=$PRETRAINED_PATH
        ;;
    
    *)
        echo "未知模式: $MODE"
        echo "可用模式: finetune_encoder, test_encoder, finetune_full, zeroshot"
        exit 1
        ;;
esac
