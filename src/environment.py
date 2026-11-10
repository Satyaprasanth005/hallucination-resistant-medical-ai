"""
Environment and Hardware Detection Utility for Kaggle and Local Development.

Detects CUDA GPUs (NVIDIA T4, P100, V100, A100, etc.), available VRAM,
system memory, CPU cores, platform details, and configures dynamic batch sizing,
mixed-precision (FP16/BF16), and execution paths.
"""

import os
import platform
import psutil
from typing import Dict, Any, Tuple
import torch


class EnvironmentConfig:
    """Manages platform detection, hardware resources, and execution profiles."""

    def __init__(self):
        self.is_kaggle = self._detect_kaggle()
        self.system_info = self._get_system_info()
        self.gpu_info = self._get_gpu_info()
        self.device, self.dtype = self._configure_device()
        self.recommended_settings = self._get_recommended_settings()

    def _detect_kaggle(self) -> bool:
        """Determines if the code is executing inside a Kaggle notebook."""
        return os.path.exists("/kaggle") or "KAGGLE_KERNEL_RUN_TYPE" in os.environ

    def _get_system_info(self) -> Dict[str, Any]:
        """Inspects OS, Python, CPU cores, and RAM."""
        ram_bytes = psutil.virtual_memory().total
        ram_gb = ram_bytes / (1024 ** 3)
        available_ram_gb = psutil.virtual_memory().available / (1024 ** 3)

        return {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "cpu_count_physical": psutil.cpu_count(logical=False),
            "total_ram_gb": round(ram_gb, 2),
            "available_ram_gb": round(available_ram_gb, 2),
            "is_kaggle": self.is_kaggle,
        }

    def _get_gpu_info(self) -> Dict[str, Any]:
        """Inspects CUDA device properties if available."""
        cuda_available = torch.cuda.is_available()
        if not cuda_available:
            return {
                "cuda_available": False,
                "device_count": 0,
                "devices": [],
                "primary_device_name": "CPU",
                "primary_vram_gb": 0.0,
            }

        devices = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total_mem_gb = props.total_memory / (1024 ** 3)
            devices.append({
                "index": i,
                "name": props.name,
                "total_memory_gb": round(total_mem_gb, 2),
                "major_version": props.major,
                "minor_version": props.minor,
                "multi_processor_count": getattr(props, "multi_processor_count", None),
            })

        return {
            "cuda_available": True,
            "device_count": torch.cuda.device_count(),
            "devices": devices,
            "primary_device_name": devices[0]["name"] if devices else "None",
            "primary_vram_gb": devices[0]["total_memory_gb"] if devices else 0.0,
            "cuda_version": torch.version.cuda,
            "cudnn_enabled": torch.backends.cudnn.enabled if hasattr(torch.backends, "cudnn") else False,
        }

    def _configure_device(self) -> Tuple[torch.device, torch.dtype]:
        """Selects optimal torch.device and precision dtype."""
        if self.gpu_info["cuda_available"]:
            device = torch.device("cuda:0")
            # T4 (Compute 7.5), P100 (Compute 6.0), Ampere+ (Compute 8.0+)
            # FP16 is standard for T4 and P100.
            dtype = torch.float16
        else:
            device = torch.device("cpu")
            dtype = torch.float32

        return device, dtype

    def _get_recommended_settings(self) -> Dict[str, Any]:
        """Computes dynamic batch size and accumulation steps to prevent OOM."""
        if not self.gpu_info["cuda_available"]:
            return {
                "vision_batch_size": 4,
                "eval_batch_size": 8,
                "gradient_accumulation_steps": 4,
                "use_amp": False,
                "dataloader_workers": 0 if platform.system() == "Windows" else 2,
            }

        vram = self.gpu_info["primary_vram_gb"]

        # Adjust based on detected GPU memory (e.g. T4 ~15GB, P100 ~16GB, or smaller/larger)
        if vram >= 14.0:
            return {
                "vision_batch_size": 16,
                "eval_batch_size": 32,
                "gradient_accumulation_steps": 2,
                "use_amp": True,
                "dataloader_workers": 2,
            }
        elif vram >= 8.0:
            return {
                "vision_batch_size": 8,
                "eval_batch_size": 16,
                "gradient_accumulation_steps": 4,
                "use_amp": True,
                "dataloader_workers": 2,
            }
        else:
            return {
                "vision_batch_size": 4,
                "eval_batch_size": 8,
                "gradient_accumulation_steps": 8,
                "use_amp": True,
                "dataloader_workers": 1,
            }

    def print_summary(self) -> str:
        """Returns a formatted diagnostic string and prints to stdout."""
        lines = [
            "=" * 60,
            "HALLUCINATION-RESISTANT MEDICAL AI: ENVIRONMENT DIAGNOSTICS",
            "=" * 60,
            f"Kaggle Environment:  {'YES (/kaggle detected)' if self.is_kaggle else 'NO (Local/Standalone)'}",
            f"Platform:            {self.system_info['platform']}",
            f"Python Version:      {self.system_info['python_version']}",
            f"PyTorch Version:     {torch.__version__}",
            f"Total System RAM:    {self.system_info['total_ram_gb']} GB (Available: {self.system_info['available_ram_gb']} GB)",
            f"CPU Cores:           {self.system_info['cpu_count_physical']} physical, {self.system_info['cpu_count_logical']} logical",
            "-" * 60,
        ]

        if self.gpu_info["cuda_available"]:
            lines.append(f"CUDA Available:      YES ({self.gpu_info['device_count']} device(s))")
            for dev in self.gpu_info["devices"]:
                lines.append(f"  - Device [{dev['index']}]: {dev['name']} ({dev['total_memory_gb']} GB VRAM)")
            lines.append(f"Target Device:       {self.device} (Precision: {self.dtype})")
        else:
            lines.append("CUDA Available:      NO -> Falling back to CPU execution")
            lines.append(f"Target Device:       {self.device} (Precision: {self.dtype})")

        lines.extend([
            "-" * 60,
            "Recommended Execution Profile:",
            f"  - Vision Batch Size:       {self.recommended_settings['vision_batch_size']}",
            f"  - Eval Batch Size:         {self.recommended_settings['eval_batch_size']}",
            f"  - Grad Accumulation Steps: {self.recommended_settings['gradient_accumulation_steps']}",
            f"  - Mixed Precision (AMP):   {self.recommended_settings['use_amp']}",
            f"  - DataLoader Workers:      {self.recommended_settings['dataloader_workers']}",
            "=" * 60,
        ])
        summary = "\n".join(lines)
        print(summary)
        return summary


def get_environment() -> EnvironmentConfig:
    """Singleton-like accessor for current environment configuration."""
    return EnvironmentConfig()


if __name__ == "__main__":
    env = get_environment()
    env.print_summary()

