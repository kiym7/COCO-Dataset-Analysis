"""全局配置管理模块。

从 config.yaml 读取配置，提供类型安全的配置访问接口。
"""

import os
from pathlib import Path
from typing import Any, Dict

import yaml


class Config:
    """全局配置单例，从 config.yaml 加载所有可配置参数。

    Attributes:
        paths: 路径相关配置（data_dir, output_dir 等）
        coco: COCO 数据集配置（年份、标注文件名）
        analysis: 分析参数配置（阈值、权重等）
        visualization: 可视化样式配置
        download: 下载相关配置
    """

    _instance: "Config | None" = None

    def __init__(self, config_path: str = "config.yaml"):
        """初始化配置。

        Args:
            config_path: YAML 配置文件路径，默认为项目根目录的 config.yaml
        """
        # 尝试多个路径查找配置文件
        search_paths = [
            Path(config_path),
            Path(__file__).parent.parent.parent / config_path,
            Path.cwd() / config_path,
        ]
        config_file = None
        for sp in search_paths:
            if sp.exists():
                config_file = sp
                break

        if config_file is None:
            raise FileNotFoundError(
                f"配置文件未找到。尝试过的路径: {[str(p) for p in search_paths]}"
            )

        with open(config_file, "r", encoding="utf-8") as f:
            self._data: Dict[str, Any] = yaml.safe_load(f)

        self._resolve_paths()

    def _resolve_paths(self) -> None:
        """将配置中的相对路径解析为绝对路径（基于项目根目录）。"""
        project_root = Path(__file__).parent.parent.parent

        paths = self._data.get("paths", {})
        for key, value in paths.items():
            if isinstance(value, str) and not os.path.isabs(value):
                paths[key] = str(project_root / value)

    # --- 便捷属性 ---

    @property
    def paths(self) -> Dict[str, str]:
        """路径配置字典。"""
        return self._data.get("paths", {})

    @property
    def coco(self) -> Dict[str, Any]:
        """COCO 数据集配置。"""
        return self._data.get("coco", {})

    @property
    def analysis(self) -> Dict[str, Any]:
        """分析参数配置。"""
        return self._data.get("analysis", {})

    @property
    def visualization(self) -> Dict[str, Any]:
        """可视化样式配置。"""
        return self._data.get("visualization", {})

    @property
    def download(self) -> Dict[str, Any]:
        """下载相关配置。"""
        return self._data.get("download", {})

    def get(self, key: str, default: Any = None) -> Any:
        """获取任意配置键的值。

        Args:
            key: 配置键（支持点号分隔的嵌套键，如 "analysis.bbox.small_area_max"）
            default: 默认值

        Returns:
            配置值或默认值
        """
        keys = key.split(".")
        value = self._data
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            if value is None:
                return default
        return value


# 全局配置实例（懒加载）
_config: Config | None = None


def get_config(config_path: str = "config.yaml") -> Config:
    """获取全局配置实例。

    Args:
        config_path: 配置文件路径

    Returns:
        Config 实例
    """
    global _config
    if _config is None:
        _config = Config(config_path)
    return _config


def reset_config() -> None:
    """重置全局配置实例（主要用于测试）。"""
    global _config
    _config = None
