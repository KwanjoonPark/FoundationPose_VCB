#!/bin/bash
# LINEMOD 데이터셋 준비 스크립트
# mask -> mask_refined 심볼릭 링크 생성

REF_VIEW_DIR="${1:-/home/ebduser/FoundationPose/linemod/ref_views}"

if [ ! -d "$REF_VIEW_DIR" ]; then
    echo "Error: Directory not found: $REF_VIEW_DIR"
    echo "Usage: $0 [ref_view_dir]"
    exit 1
fi

echo "Creating mask_refined symlinks in $REF_VIEW_DIR..."

for dir in "$REF_VIEW_DIR"/ob_*/; do
    if [ -d "${dir}mask" ] && [ ! -e "${dir}mask_refined" ]; then
        ln -s "${dir}mask" "${dir}mask_refined"
        echo "Created: ${dir}mask_refined -> ${dir}mask"
    elif [ -e "${dir}mask_refined" ]; then
        echo "Already exists: ${dir}mask_refined"
    fi
done

echo "Done!"
