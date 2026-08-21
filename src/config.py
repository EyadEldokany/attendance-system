"""
Load and manage system settings from config.yaml
"""
import yaml
import os
from pathlib import Path
from threading import Lock

from src.path_utils import get_app_dir, get_bundle_dir, resolve_path, ensure_config_exists

_CONFIG_LOCK = Lock()


class Config:
    """
    A simple singleton to load settings once and share them across all system modules.
    Supports safe rewriting (for example, after adjusting the door line from the settings interface).
    """
    _instance = None

    def __new__(cls, config_path="config.yaml"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load(config_path)
        return cls._instance

    def _load(self, config_path):
        if config_path == "config.yaml" or str(config_path) == "config.yaml":
            self._path = ensure_config_exists("config.yaml")
        else:
            self._path = resolve_path(config_path)

        if not self._path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path} (resolved to {self._path})")
        with open(self._path, "r", encoding="utf-8") as f:
            self._data = yaml.safe_load(f)

    def get(self, dotted_key, default=None):
        """
        Access settings using dotted notation, e.g.: config.get("camera.source")
        """
        node = self._data
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set(self, dotted_key, value, persist=False):
        """Update a setting value during runtime, with an option to save it to the file."""
        with _CONFIG_LOCK:
            parts = dotted_key.split(".")
            node = self._data
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = value
            if persist:
                self._save()

    def _save(self):
        with open(self._path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self._data, f, allow_unicode=True, sort_keys=False)

    @property
    def raw(self):
        return self._data


def load_config(config_path="config.yaml") -> Config:
    return Config(config_path)

