"""
Global Configuration Manager for Hallucination-Resistant Medical AI.

Loads YAML configuration files, merges environment overrides,
and dynamically resolves paths for Kaggle and Local development setups.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml

from src.environment import get_environment


class ProjectConfig:
    """Manages loaded configurations and unified path resolutions."""

    def __init__(self, config_dir: Optional[str] = None):
        self.env = get_environment()
        self.base_dir = self._find_project_root()

        if config_dir is None:
            self.config_dir = self.base_dir / "configs"
        else:
            self.config_dir = Path(config_dir)

        # Load configurations
        self.dataset_cfg = self._load_yaml(self.config_dir / "dataset.yaml")
        self.model_cfg = self._load_yaml(self.config_dir / "model.yaml")
        self.experiments_cfg = self._load_yaml(self.config_dir / "experiments.yaml")

        # Dynamic output directory resolution
        self.output_dir = self._resolve_output_dir()
        self._ensure_output_dirs()

    def _find_project_root(self) -> Path:
        """Determines the project root directory."""
        # Check standard locations
        candidates = [
            Path("/kaggle/working/hallucination_medical_ai"),
            Path("/kaggle/working"),
            Path(__file__).resolve().parent.parent,
            Path.cwd(),
        ]
        for candidate in candidates:
            if (candidate / "configs" / "dataset.yaml").exists():
                return candidate
        # Fallback to parent of current file
        return Path(__file__).resolve().parent.parent

    def _resolve_output_dir(self) -> Path:
        """Kaggle persistent path /kaggle/working/medical_ai_outputs/ or local outputs/."""
        env_override = os.environ.get("MEDICAL_AI_OUTPUT_DIR")
        if env_override:
            return Path(env_override)

        cfg_override = self.model_cfg.get("model", {}).get("hardware", {}).get("output_dir_override")
        if self.env.is_kaggle:
            if cfg_override:
                return Path(cfg_override)
            return Path("/kaggle/working/medical_ai_outputs")
        return self.base_dir / "outputs"


    def _ensure_output_dirs(self) -> None:
        """Creates output subdirectories for persistent artifacts."""
        subdirs = ["reports", "metrics", "checkpoints", "plots", "faiss_index"]
        for s in subdirs:
            (self.output_dir / s).mkdir(parents=True, exist_ok=True)

    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        """Safely loads a YAML configuration file."""
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    @property
    def dataset(self) -> Dict[str, Any]:
        return self.dataset_cfg.get("dataset", {})

    @property
    def model(self) -> Dict[str, Any]:
        return self.model_cfg.get("model", {})

    @property
    def experiments(self) -> Dict[str, Any]:
        return self.experiments_cfg.get("experiments", {})

    def resolve_dataset_path(self, relative_or_absolute: str) -> Path:
        """Resolves dataset paths checking Kaggle input roots and local data."""
        p = Path(relative_or_absolute)
        if p.is_absolute() and p.exists():
            return p

        # Check search roots
        search_roots = self.dataset.get("search_roots", {})
        active_roots = search_roots.get("kaggle" if self.env.is_kaggle else "local", [])

        for root in active_roots:
            candidate = Path(root) / relative_or_absolute
            if candidate.exists():
                return candidate

        # Return fallback path under project
        return self.base_dir / relative_or_absolute


# Module-level singleton instance
_config_instance = None


def get_config(reload: bool = False) -> ProjectConfig:
    """Returns the global ProjectConfig instance."""
    global _config_instance
    if _config_instance is None or reload:
        _config_instance = ProjectConfig()
    return _config_instance


if __name__ == "__main__":
    cfg = get_config()
    print("Project Root:", cfg.base_dir)
    print("Output Dir:  ", cfg.output_dir)
    print("Abnormality classes (count):", len(cfg.dataset.get("abnormality_classes", [])))
    print("Vision Backbone:", cfg.model.get("vision_encoder", {}).get("backbone"))

