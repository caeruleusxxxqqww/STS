#!/bin/bash
# =========================================================================
# 消融实验 Fine-tuning 批量运行脚本
# =========================================================================
# 方法: 加载 Baseline checkpoint，微调不同模块组合
# 优势: 节省 70% 训练时间，且起点一致更公平
# =========================================================================
# 用法:
#   全部运行 (S3DIS + ScanNet):
#     bash scripts/run_ablation_finetune.sh [GPU_ID] [S3DIS_CKPT] [SCANNET_CKPT]
#
#   只跑 S3DIS:
#     bash scripts/run_ablation_finetune.sh [GPU_ID] [S3DIS_CKPT]
#
# 示例:
#   bash scripts/run_ablation_finetune.sh 0 \
#       /path/to/s3dis_baseline/ckpt_best.pth \
#       /path/to/scannet_baseline/ckpt_best.pth
# =========================================================================

set -e

# 解析参数
GPU_ID=${1:-0}
S3DIS_CKPT=${2:-""}
SCANNET_CKPT=${3:-""}

if [ -z "$S3DIS_CKPT" ]; then
    echo "❌ 错误: 请至少提供 S3DIS Baseline checkpoint 路径"
    echo "用法: bash scripts/run_ablation_finetune.sh [GPU_ID] [S3DIS_CKPT] [SCANNET_CKPT]"
    echo "示例: bash scripts/run_ablation_finetune.sh 0 /path/to/s3dis/ckpt_best.pth /path/to/scannet/ckpt_best.pth"
    exit 1
fi

if [ ! -f "$S3DIS_CKPT" ]; then
    echo "❌ 错误: S3DIS Baseline checkpoint 不存在: $S3DIS_CKPT"
    exit 1
fi

if [ -n "$SCANNET_CKPT" ] && [ ! -f "$SCANNET_CKPT" ]; then
    echo "❌ 错误: ScanNet Baseline checkpoint 不存在: $SCANNET_CKPT"
    exit 1
fi

# 切换到项目根目录
cd /home/LIANGYudong_2023/PointNeXt

# =========================================================================
# 定义实验列表
# =========================================================================
# S3DIS 实验
S3DIS_EXPERIMENTS=(
    "exp3-sts-only-finetune"
    "exp5-gtigu-sts-finetune"
)
S3DIS_DESCRIPTIONS=(
    "[S3DIS] Exp-3: Baseline + STS (完全加载 + Fine-tune)"
    "[S3DIS] Exp-5: Baseline + STS + GT-IGU (部分加载 + Fine-tune)"
)

# ScanNet 实验
SCANNET_EXPERIMENTS=(
    "exp3-sts-only-finetune-scannet"
    "exp5-gtigu-sts-finetune-scannet"
)
SCANNET_DESCRIPTIONS=(
    "[ScanNet] Exp-3: Baseline + STS (完全加载 + Fine-tune)"
    "[ScanNet] Exp-5: Baseline + STS + GT-IGU (部分加载 + Fine-tune)"
)

# 创建日志目录
LOG_DIR="logs/ablation_finetune_$(date +%Y%m%d_%H%M%S)"
mkdir -p $LOG_DIR

# 汇总用的数组
ALL_EXP_NAMES=()
ALL_LOG_FILES=()

echo "==========================================================================="
echo "🧪 消融实验 Fine-tuning 批量运行"
echo "==========================================================================="
echo "GPU: $GPU_ID"
echo "S3DIS  checkpoint: $S3DIS_CKPT"
if [ -n "$SCANNET_CKPT" ]; then
    echo "ScanNet checkpoint: $SCANNET_CKPT"
else
    echo "ScanNet checkpoint: (未提供，跳过 ScanNet 实验)"
fi
echo "日志目录: $LOG_DIR"
echo "==========================================================================="

# ----- 运行函数 -----
run_experiment() {
    local EXP_NAME=$1
    local EXP_DESC=$2
    local CKPT_PATH=$3
    local EXP_IDX=$4
    local TOTAL=$5

    local CFG_PATH="cfgs/ablation/finetune/${EXP_NAME}.yaml"
    local LOG_FILE="$LOG_DIR/${EXP_NAME}.log"

    # 记录到汇总列表
    ALL_EXP_NAMES+=("$EXP_NAME")
    ALL_LOG_FILES+=("$LOG_FILE")

    echo ""
    echo "==========================================================================="
    echo "📌 实验 ${EXP_IDX}/${TOTAL}: $EXP_NAME"
    echo "📝 描述: $EXP_DESC"
    echo "📄 配置: $CFG_PATH"
    echo "🏋️ Checkpoint: $CKPT_PATH"
    echo "📊 日志: $LOG_FILE"
    echo "==========================================================================="

    # 检查配置文件
    if [ ! -f "$CFG_PATH" ]; then
        echo "❌ 配置文件不存在: $CFG_PATH，跳过"
        return
    fi

    # 运行 Fine-tuning
    echo "🚀 开始 Fine-tuning..."
    CUDA_VISIBLE_DEVICES=$GPU_ID python examples/segmentation/main.py \
        --cfg $CFG_PATH \
        pretrained_path=$CKPT_PATH \
        2>&1 | tee $LOG_FILE

    echo "✅ 实验 $EXP_NAME 完成"
}

# =========================================================================
# 计算总实验数
# =========================================================================
TOTAL_EXPS=${#S3DIS_EXPERIMENTS[@]}
if [ -n "$SCANNET_CKPT" ]; then
    TOTAL_EXPS=$((TOTAL_EXPS + ${#SCANNET_EXPERIMENTS[@]}))
fi
echo "实验总数: $TOTAL_EXPS 个"

EXP_COUNTER=0

# =========================================================================
# 运行 S3DIS 实验
# =========================================================================
echo ""
echo "==================== 🗂️  S3DIS 实验 ===================="
for i in "${!S3DIS_EXPERIMENTS[@]}"; do
    EXP_COUNTER=$((EXP_COUNTER + 1))
    run_experiment "${S3DIS_EXPERIMENTS[$i]}" "${S3DIS_DESCRIPTIONS[$i]}" "$S3DIS_CKPT" "$EXP_COUNTER" "$TOTAL_EXPS"
done

# =========================================================================
# 运行 ScanNet 实验
# =========================================================================
if [ -n "$SCANNET_CKPT" ]; then
    echo ""
    echo "==================== 🗂️  ScanNet 实验 ===================="
    for i in "${!SCANNET_EXPERIMENTS[@]}"; do
        EXP_COUNTER=$((EXP_COUNTER + 1))
        run_experiment "${SCANNET_EXPERIMENTS[$i]}" "${SCANNET_DESCRIPTIONS[$i]}" "$SCANNET_CKPT" "$EXP_COUNTER" "$TOTAL_EXPS"
    done
fi

# =========================================================================
# 汇总所有结果
# =========================================================================
echo ""
echo "==========================================================================="
echo "🎉 所有 Fine-tuning 实验完成！"
echo "日志保存在: $LOG_DIR"
echo "==========================================================================="
echo ""
echo "📊 结果汇总:"
echo "-----------------------------------------------------------"
printf "  %-45s  %s\n" "实验名称" "Best mIoU"
echo "-----------------------------------------------------------"

for i in "${!ALL_EXP_NAMES[@]}"; do
    EXP_NAME=${ALL_EXP_NAMES[$i]}
    LOG_FILE=${ALL_LOG_FILES[$i]}

    if [ -f "$LOG_FILE" ]; then
        BEST_MIOU=$(grep -oP "best val miou \K[\d.]+" "$LOG_FILE" 2>/dev/null | tail -1)
        if [ -z "$BEST_MIOU" ]; then
            BEST_MIOU=$(grep -oP "val_miou \K[\d.]+" "$LOG_FILE" 2>/dev/null | tail -1)
        fi
        printf "  %-45s  %s\n" "$EXP_NAME" "${BEST_MIOU:-N/A}"
    else
        printf "  %-45s  %s\n" "$EXP_NAME" "(日志不存在)"
    fi
done

echo "-----------------------------------------------------------"
