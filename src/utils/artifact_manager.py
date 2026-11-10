"""
Artifact Management and Persistent Export System.
Packages metrics, model checkpoints, FAISS indexes, generated reports,
and plots into persistent downloadable archives in Kaggle and local setups.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import shutil
import json
import zipfile
from datetime import datetime
from src.config import ProjectConfig, get_config
from src.utils.logger import get_logger

logger = get_logger("ArtifactManager")


class ArtifactManager:
    """Handles persistence, export bundling, and checkpoint management."""

    def __init__(self, config: Optional[ProjectConfig] = None):
        self.cfg = config or get_config()
        self.output_dir = self.cfg.output_dir

    def save_experiment_outputs(self, export_archive_name: Optional[str] = None) -> Path:
        """Packages all reports, metrics, plots, and metadata into a persistent zip archive."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if export_archive_name is None:
            archive_filename = f"medical_ai_research_artifacts_{timestamp}.zip"
        else:
            archive_filename = f"{export_archive_name}.zip"

        archive_path = self.output_dir / archive_filename

        # Write manifest metadata
        manifest = {
            "created_at": datetime.now().isoformat(),
            "kaggle_environment": self.cfg.env.is_kaggle,
            "device": str(self.cfg.env.device),
            "pytorch_version": self.cfg.env.system_info.get("python_version"),
            "dataset_config": self.cfg.dataset,
            "model_config": self.cfg.model,
        }
        with open(self.output_dir / "experiment_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        # Create zip archive of outputs/
        logger.info(f"Creating export archive at {archive_path}...")
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for file_path in self.output_dir.rglob("*"):
                if file_path.is_file() and file_path != archive_path and not file_path.suffix == ".zip":
                    rel_path = file_path.relative_to(self.output_dir)
                    zipf.write(file_path, arcname=str(rel_path))

        logger.info(f"Artifact bundle successfully exported to: {archive_path}")
        return archive_path

    def save_checkpoint(
        self,
        model_state_dict: Dict[str, Any],
        epoch: int,
        metrics: Dict[str, float],
        filename: str = "checkpoint_best.pt",
    ) -> Path:
        """Saves a model training checkpoint with resumption metadata."""
        ckpt_dir = self.output_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / filename

        import torch
        payload = {
            "epoch": epoch,
            "metrics": metrics,
            "state_dict": model_state_dict,
            "timestamp": datetime.now().isoformat(),
        }
        torch.save(payload, ckpt_path)
        logger.info(f"Checkpoint saved to {ckpt_path}")
        return ckpt_path

    def load_checkpoint(self, filename: str = "checkpoint_best.pt") -> Optional[Dict[str, Any]]:
        """Loads a checkpoint if it exists for resumption."""
        ckpt_path = self.output_dir / "checkpoints" / filename
        if not ckpt_path.exists():
            return None
        import torch
        logger.info(f"Resuming from checkpoint {ckpt_path}...")
        return torch.load(ckpt_path, map_location=self.cfg.env.device)


def save_experiment_outputs(output_dir: Optional[Path] = None) -> Path:
    """Convenience function satisfying Part 17 specification."""
    mgr = ArtifactManager()
    return mgr.save_experiment_outputs()

