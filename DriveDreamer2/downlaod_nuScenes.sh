#!/bin/bash

OUT_DIR="/mnt/home/yf2578/runzhouliu/DriveDreamer2/nuScenes"
mkdir -p "$OUT_DIR"
cd "$OUT_DIR" || exit 1

BASE_URL="https://motional-nuscenes.s3.amazonaws.com/public/v1.0"

echo "下载 trainval v1.0 01–10 blobs..."

PREFIX="v1.0-trainval"
SUFFIX="_blobs.tgz"

# 循环下载 trainval01–trainval10
for i in $(seq 1 10); do
    num=$(printf "%02d" "$i")
    FILE="${PREFIX}${num}${SUFFIX}"
    URL="${BASE_URL}/${FILE}"
    echo "Downloading $FILE ..."
    wget -c -O "$FILE" "$URL"
done

echo "trainval 下载完成！"

echo "下载 test blob..."

TEST_FILE="v1.0-test_blobs.tgz"
wget -c -O "$TEST_FILE" "${BASE_URL}/${TEST_FILE}"

echo "test blob 下载完成！"
echo "所有文件已存放到: $OUT_DIR"
