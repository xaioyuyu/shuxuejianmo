#!/bin/bash
# 运行所有数学建模问题的程序
# 使用方法：./run_all.sh

echo "===================================================================="
echo "               数学建模竞赛 - 全部程序运行脚本"
echo "===================================================================="
echo ""

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "支撑材料目录: $SCRIPT_DIR"

# 切换到项目根目录（支撑材料的上上级目录）
cd "$SCRIPT_DIR/../.." || exit 1
echo "工作目录: $(pwd)"
echo ""

# 检查Python是否可用
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到 python3"
    echo "请先安装 Python 3.8+"
    exit 1
fi

# 检查必需的库
echo "检查Python依赖库..."
python3 -c "import numpy, pandas, matplotlib, seaborn, scipy, sklearn, statsmodels" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "警告: 缺少必需的Python库"
    echo "请运行: pip install numpy pandas matplotlib seaborn scipy scikit-learn statsmodels"
    echo ""
    read -p "是否继续？(y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "===================================================================="
echo "开始运行所有问题..."
echo "===================================================================="
echo ""

# 运行5个问题
problems=(
    "第一个问题/第一个问题_实现代码_20251105.py|问题一：USDT与USDC对比分析"
    "第二个问题/第二个问题_实现代码_20251105.py|问题二：储备资产配置优化"
    "第三个问题/第三个问题_实现代码_20251105.py|问题三：需求预测与市场份额"
    "第四个问题/第四个问题_实现代码_20251105.py|问题四：货币主权影响评估"
    "第五个问题/第五个问题_实现代码_20251105.py|问题五：政策简报生成"
)

success_count=0
total_count=${#problems[@]}

for i in "${!problems[@]}"; do
    IFS='|' read -r code_file title <<< "${problems[$i]}"
    echo "【$((i+1))/$total_count】正在运行 $title..."
    echo "  代码文件: $code_file"
    
    if python3 "$code_file"; then
        echo "  ✓ 运行成功"
        ((success_count++))
    else
        echo "  ✗ 运行失败"
    fi
    echo ""
done

# 将数据文件复制到支撑材料/数据文件目录
echo "正在整理数据文件和图片..."
DATA_DIR="$SCRIPT_DIR/数据文件"
mkdir -p "$DATA_DIR"

# 复制CSV数据文件
cp -f 第一个问题/*.csv "$DATA_DIR/" 2>/dev/null
cp -f 第二个问题/*.csv "$DATA_DIR/" 2>/dev/null
cp -f 第三个问题/*.csv "$DATA_DIR/" 2>/dev/null
cp -f 第四个问题/*.csv "$DATA_DIR/" 2>/dev/null

# 复制TXT文件
cp -f 第五个问题/*.txt "$DATA_DIR/" 2>/dev/null

# 复制PNG图片文件
cp -f 第一个问题/*.png "$DATA_DIR/" 2>/dev/null
cp -f 第二个问题/*.png "$DATA_DIR/" 2>/dev/null
cp -f 第三个问题/*.png "$DATA_DIR/" 2>/dev/null
cp -f 第四个问题/*.png "$DATA_DIR/" 2>/dev/null
cp -f 第五个问题/*.png "$DATA_DIR/" 2>/dev/null

echo ""
echo "===================================================================="
echo "                         运行完成！"
echo "===================================================================="
echo "成功运行: $success_count/$total_count 个问题"
echo "数据文件保存位置: $DATA_DIR"
echo ""
echo "生成的文件："
ls -lh "$DATA_DIR" | grep -v "^d" | awk '{print "  " $9 " (" $5 ")"}'
echo ""
echo "===================================================================="
