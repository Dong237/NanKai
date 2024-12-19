#!/bin/bash

# === 用于运行收敛测试脚本的高级 Bash 脚本 ===

# 参数默认值
THRESHOLD=0.001
LOSS_FOLDER="losses"
PYTHON_SCRIPT="cauchy_test.py"

# 打印使用说明
usage() {
    echo "使用方法: $0 [-f <loss_folder>] [-t <threshold>] [-s <python_script>]"
    echo ""
    echo "选项:"
    echo "  -f, --folder     包含 .npy 损失文件的文件夹路径 (默认值: 'losses')"
    echo "  -t, --threshold  柯西准则测试的收敛阈值 (默认值: 0.001)"
    echo "  -s, --script     要运行的 Python 脚本路径 (默认值: 'cauchy_test.py')"
    echo "  -h, --help       显示此帮助信息"
}

# 解析命令行参数
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -f|--folder)
            LOSS_FOLDER="$2"
            shift
            ;;
        -t|--threshold)
            THRESHOLD="$2"
            shift
            ;;
        -s|--script)
            PYTHON_SCRIPT="$2"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            usage
            exit 1
            ;;
    esac
    shift
done

# 验证输入
if [[ ! -d "$LOSS_FOLDER" ]]; then
    echo "错误: 指定的损失文件夹 '$LOSS_FOLDER' 不存在。"
    exit 1
fi

if ! [[ "$THRESHOLD" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    echo "错误: 阈值必须是正数。"
    exit 1
fi

if [[ ! -f "$PYTHON_SCRIPT" ]]; then
    echo "错误: 找不到 Python 脚本 '$PYTHON_SCRIPT'。"
    exit 1
fi

# 日志: 开始
START_TIME=$(date +"%Y-%m-%d %H:%M:%S")
echo "==========================================="
echo "开始运行收敛测试脚本"
echo "启动时间: $START_TIME"
echo "损失文件夹: $LOSS_FOLDER"
echo "阈值: $THRESHOLD"
echo "脚本: $PYTHON_SCRIPT"
echo "==========================================="

# 执行 Python 脚本
python "$PYTHON_SCRIPT" --loss_folder "$LOSS_FOLDER" --threshold "$THRESHOLD"

# 日志: 结束
END_TIME=$(date +"%Y-%m-%d %H:%M:%S")
echo "==========================================="
echo "收敛测试脚本运行结束"
echo "结束时间: $END_TIME"
echo "==========================================="
