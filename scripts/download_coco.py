#!/usr/bin/env python
"""COCO 2017 数据集标注文件下载脚本。

从 COCO 官方网站下载标注文件 (annotations_trainval2017.zip, ~250MB)
和可选的验证集图像 (val2017.zip, ~1GB)。

用法:
    python scripts/download_coco.py                    # 仅下载标注
    python scripts/download_coco.py --with-val-images  # 下载标注 + 验证集图像
    python scripts/download_coco.py --output ./my_data  # 指定输出目录
"""

import argparse
import os
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

# 添加项目根目录到 path，以便导入 config 模块
sys.path.insert(0, str(Path(__file__).parent.parent))

COCO_BASE_URL = "http://images.cocodataset.org"

FILES = {
    "annotations": {
        "url": f"{COCO_BASE_URL}/annotations/annotations_trainval2017.zip",
        "filename": "annotations_trainval2017.zip",
        "description": "COCO 2017 标注文件 (train+val)",
    },
    "val_images": {
        "url": f"{COCO_BASE_URL}/zips/val2017.zip",
        "filename": "val2017.zip",
        "description": "COCO 2017 验证集图像 (val2017)",
    },
    "train_images": {
        "url": f"{COCO_BASE_URL}/zips/train2017.zip",
        "filename": "train2017.zip",
        "description": "COCO 2017 训练集图像 (train2017, ~18GB)",
    },
}


def download_file(url: str, dest_path: Path, description: str) -> None:
    """下载文件并显示进度条。

    Args:
        url: 下载链接
        dest_path: 目标文件路径
        description: 文件描述（用于进度条）
    """
    print(f"\n正在下载: {description}")
    print(f"  地址: {url}")
    print(f"  保存到: {dest_path}")

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))

    with open(dest_path, "wb") as f:
        with tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=dest_path.name,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

    print(f"  ✓ 下载完成: {dest_path.name}")


def extract_zip(zip_path: Path, extract_to: Path) -> None:
    """解压 ZIP 文件。

    Args:
        zip_path: ZIP 文件路径
        extract_to: 解压目标目录
    """
    print(f"\n正在解压: {zip_path.name} -> {extract_to}")
    extract_to.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        with tqdm(total=len(members), desc="解压中") as pbar:
            for member in members:
                zf.extract(member, extract_to)
                pbar.update(1)

    print(f"  ✓ 解压完成")


def main():
    parser = argparse.ArgumentParser(
        description="下载 COCO 2017 数据集文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python download_coco.py                        # 仅下载标注文件 (~250MB)
  python download_coco.py --with-val-images       # 标注 + 验证集图像 (~1.2GB)
  python download_coco.py --with-train-images     # 标注 + 训练集图像 (~18.5GB)
  python download_coco.py --output ./my_data      # 指定输出目录
        """,
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="data/coco",
        help="输出根目录 (默认: data/coco)",
    )
    parser.add_argument(
        "--with-val-images",
        action="store_true",
        help="同时下载验证集图像 (val2017.zip, ~1GB)",
    )
    parser.add_argument(
        "--with-train-images",
        action="store_true",
        help="同时下载训练集图像 (train2017.zip, ~18GB)",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="下载后不解压",
    )
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="解压后保留 ZIP 文件",
    )

    args = parser.parse_args()

    # 创建输出目录
    output_root = Path(args.output)
    annotations_dir = output_root / "annotations"
    images_dir = output_root / "images"
    annotations_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  COCO 2017 数据集下载工具")
    print("=" * 60)
    print(f"  输出目录: {output_root.resolve()}")
    print(f"  标注目录: {annotations_dir}")
    print(f"  图像目录: {images_dir}")
    print()

    # 确定要下载的文件
    to_download = ["annotations"]
    if args.with_val_images:
        to_download.append("val_images")
    if args.with_train_images:
        to_download.append("train_images")

    downloaded_zips = []

    for key in to_download:
        info = FILES[key]
        zip_path = output_root / info["filename"]

        # 检查是否已下载
        if zip_path.exists():
            print(f"  ⚠ {info['filename']} 已存在，跳过下载")
        else:
            try:
                download_file(info["url"], zip_path, info["description"])
            except requests.RequestException as e:
                print(f"  ✗ 下载失败: {e}")
                sys.exit(1)

        downloaded_zips.append(zip_path)

    # 解压
    if not args.no_extract:
        for zip_path in downloaded_zips:
            if not zip_path.exists():
                continue

            # 判断解压目标
            if "annotations" in zip_path.name:
                extract_to = output_root
            else:
                extract_to = output_root

            try:
                extract_zip(zip_path, extract_to)
            except Exception as e:
                print(f"  ✗ 解压失败: {e}")
                sys.exit(1)

            # 清理 ZIP
            if not args.keep_zip:
                zip_path.unlink()
                print(f"  已删除: {zip_path.name}")

    print("\n" + "=" * 60)
    print("  ✓ 所有文件下载完成!")
    print("=" * 60)

    # 列出下载的文件
    print("\n标注文件:")
    for f in sorted(annotations_dir.rglob("*.json")):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  - {f.relative_to(output_root)} ({size_mb:.1f} MB)")

    if images_dir.exists():
        image_files = list(images_dir.rglob("*.jpg"))
        if image_files:
            print(f"\n图像文件: {len(image_files)} 张")
            print(f"  (位于 {images_dir})")

    print("\n下一步:")
    print(f"  python scripts/run_analysis.py --annotations {annotations_dir}/instances_train2017.json")


if __name__ == "__main__":
    main()
