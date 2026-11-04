"""Make NVIDIA's CUDA/cuDNN DLLs findable on Windows before GPU libraries load.

CTranslate2 (faster-whisper) looks for cuBLAS 12 / cuDNN 9 on the DLL search path, and onnxruntime-gpu
needs the same runtime. The installer puts them in the venv (torch's cu128 wheels ship them in torch/lib;
nvidia-* wheels put them under site-packages/nvidia/*/bin), but Windows does not search there by itself.
Call ``setup_cuda_dlls()`` once at process start, before importing faster_whisper/onnxruntime/torch."""

from __future__ import annotations

import logging
import os
import site
import sys
from pathlib import Path

log = logging.getLogger(__name__)
_dirs: list[str] | None = None
_preloaded = False


def candidate_dirs() -> list[Path]:
    roots = [Path(p) for p in (*site.getsitepackages(), site.getusersitepackages()) if p]
    out: list[Path] = []
    for r in roots:
        out.append(r / "torch" / "lib")
        nv = r / "nvidia"
        if nv.is_dir():
            out += sorted(d / "bin" for d in nv.iterdir())
    cuda = os.environ.get("CUDA_PATH")
    if cuda:
        out.append(Path(cuda) / "bin")
    return [d for d in dict.fromkeys(out) if d.is_dir()]


def setup_cuda_dlls(preload_onnxruntime: bool = True) -> list[str]:
    """Idempotent. Returns the directories added (empty on non-Windows)."""
    global _dirs, _preloaded
    if sys.platform != "win32":
        return []
    if _dirs is None:
        _dirs = []
        for d in candidate_dirs():
            try:
                os.add_dll_directory(str(d))  # type: ignore[attr-defined]
            except OSError:
                continue
            os.environ["PATH"] = f"{d}{os.pathsep}{os.environ.get('PATH', '')}"  # CTranslate2 uses the PATH search
            _dirs.append(str(d))
        if _dirs:
            log.info("CUDA DLL directories: %s", "; ".join(_dirs))
    if preload_onnxruntime and not _preloaded:
        _preloaded = True
        try:
            import onnxruntime as ort

            if hasattr(ort, "preload_dlls"):  # onnxruntime >= 1.21
                torch_lib = next((d for d in _dirs if d.endswith(os.path.join("torch", "lib"))), None)
                ort.preload_dlls(directory=torch_lib or "")
        except Exception as e:  # CPU-only installs are fine
            log.debug("onnxruntime DLL preload skipped: %s", e)
    return _dirs
