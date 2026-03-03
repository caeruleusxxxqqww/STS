#!/bin/bash
# =========================================================================
# 消融实验批量运行脚本
# =========================================================================
# 用法: bash scripts/run_ablation_experiments.sh [GPU_ID] [START_EXP] [END_EXP]
# 示例: bash scripts/run_ablation_experiments.sh 0 0 7      # GPU 0, 运行 exp0-exp7
# 示例: bash scripts/run_ablation_experiments.sh 1 4 7      # GPU 1, 只运行 exp4-exp7
# =========================================================================

set -e

# 解析参数
GPU_ID=${1:-0}
START_EXP=${2:-0}
END_EXP=${3:-7}

# 切换到项目根目录
cd /home/LIANGYudong_2023/PointNeXt

# 定义实验配置
EXPERIMENTS=(
    "exp0-baseline"
    "exp1-gtigu-only"
    "exp2-tsi-only"
    "exp3-sts-only"
    "exp4-gtigu-tsi"
    "exp5-gtigu-sts"
    "exp6-tsi-sts"
    "exp7-full"
)

DESCRIPTIONS=(
    "Baseline (无任何创新模块)"
    "仅 GT-IGU"
    "仅 TSI-Loss"
    "仅 STS + L_cons"
    "GT-IGU + TSI-Loss"
    "GT-IGU + STS + L_cons"
    "TSI-Loss + STS + L_cons"
    "Full Model (所有模块)"
)

# 创建日志目录
LOG_DIR="logs/ablation_$(date +%Y%m%d_%H%M%S)"
mkdir -p $LOG_DIR

echo "==========================================================================="
echo "🧪 消融实验批量运行"
echo "==========================================================================="
echo "GPU: $GPU_ID"
echo "实验范围: exp$START_EXP ~ exp$END_EXP"
echo "日志目录: $LOG_DIR"
echo "==========================================================================="

# 运行实验
for i in $(seq $START_EXP $END_EXP); do
    EXP_NAME=${EXPERIMENTS[$i]}
    EXP_DESC=${DESCRIPTIONS[$i]}
    CFG_PATH="cfgs/ablation/${EXP_NAME}.yaml"
    LOG_FILE="$LOG_DIR/${EXP_NAME}.log"
    
    echo ""
    echo "==========================================================================="
    echo "📌 实验 $i: $EXP_NAME"
    echo "📝 描述: $EXP_DESC"
    echo "📄 配置: $CFG_PATH"
    echo "📊 日志: $LOG_FILE"
    echo "==========================================================================="
    
    # 检查配置文件是否存在
    if [ ! -f "$CFG_PATH" ]; then
        echo "❌ 配置文件不存在: $CFG_PATH"
        continue
    fi
    
    # 运行训练
    echo "🚀 开始训练..."
    CUDA_VISIBLE_DEVICES=$GPU_ID python examples/segmentation/main.py \
        --cfg $CFG_PATH \
        2>&1 | tee $LOG_FILE
    
    echo "✅ 实验 $i 完成"
done

echo ""
echo "==========================================================================="
echo "🎉 所有消融实验完成！"
echo "日志保存在: $LOG_DIR"
echo "==========================================================================="

# 汇总结果 (可选)
echo ""
echo "📊 结果汇总:"
for i in $(seq $START_EXP $END_EXP); do
    EXP_NAME=${EXPERIMENTS[$i]}
    LOG_FILE="$LOG_DIR/${EXP_NAME}.log"
    
    if [ -f "$LOG_FILE" ]; then
        # 提取最佳 mIoU
        BEST_MIOU=$(grep -E "Best.*mIoU|best.*miou" $LOG_FILE | tail -1 || echo "N/A")
        echo "  $EXP_NAME: $BEST_MIOU"
    fi
done
