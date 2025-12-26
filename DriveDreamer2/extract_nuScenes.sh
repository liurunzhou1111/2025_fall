#!/bin/bash

OUT_DIR="/mnt/home/yf2578/runzhouliu/DriveDreamer2/nuScenes"
cd "$OUT_DIR" || exit 1

echo "开始解压 nuScenes 文件夹中的所有文件..."

# 解压 .tar.bz2 文件
for file in *.tar.bz2; do
    if [ -f "$file" ]; then
        echo "解压 $file..."
        tar -xjf "$file"
        echo "完成: $file"
    fi
done

# 解压 .tgz 文件
for file in *.tgz; do
    if [ -f "$file" ]; then
        echo "解压 $file..."
        tar -xzf "$file"
        echo "完成: $file"
    fi
done

echo "所有文件解压完成！"

