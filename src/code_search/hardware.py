"""Find the machine's GPU, install the PyTorch build that can use it, and pick the device at runtime.

PyPI's torch wheels only cover CPU (Windows), CUDA (Linux) and Apple MPS (macOS). `code-search setup`
fills the gaps: CUDA wheels via uv's `--torch-backend auto` on NVIDIA, AMD's ROCm wheels for the
exact gfx target on Radeon (Windows and Linux, no system ROCm needed), nothing extra on Apple Silicon.
"""
from __future__ import annotations

import functools
import json
import glob
import os
import platform
import re
import shutil
import subprocess
import sys

# must be set before torch is imported: ops missing on MPS fall back to CPU instead of raising
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

AMD_INDEX = "https://stable.repo.amd.com/rocm/whl-next/"

# marketing name -> gfx target; AMD ships one torch "device-<gfx>" extra per target
_AMD_GFX = [
    (r"RX 9070|R9700", "gfx1201"),
    (r"RX 9060|R9600", "gfx1200"),
    (r"RX 7900|W7900|W7800", "gfx1100"),
    (r"RX 7800|RX 7700|W7700", "gfx1101"),
    (r"RX 7600|W7600|W7500", "gfx1102"),
    (r"780M|760M|740M", "gfx1103"),
    (r"8060S|8050S|8040S", "gfx1151"),
    (r"890M|880M", "gfx1150"),
    (r"860M|840M|820M", "gfx1152"),
    (r"RX 69[05]0|RX 6800|W6800", "gfx1030"),
    (r"RX 67[05]0|RX 6700", "gfx1031"),
    (r"RX 66[05]0|RX 6600", "gfx1032"),
]


def amd_gfx(name: str) -> str | None:
    """gfx target for an AMD GPU name, e.g. 'AMD Radeon RX 9070 XT' -> 'gfx1201'."""
    return next((gfx for pat, gfx in _AMD_GFX if re.search(pat, name, re.I)), None)


def kfd_gfx(version: int) -> str | None:
    """Linux kfd gfx_target_version (e.g. 120001, 90010) -> 'gfx1201', 'gfx90a'."""
    if not version:
        return None
    return f"gfx{version // 10000}{(version // 100) % 100:x}{version % 100:x}"


def _run(cmd: list[str], timeout: float = 20) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


@functools.lru_cache(maxsize=1)
def gpus() -> tuple[dict, ...]:
    """GPUs on this machine as {vendor, name, gfx}; works without torch."""
    if sys.platform == "darwin":
        return ({"vendor": "apple", "name": "Apple Silicon", "gfx": None},) if platform.machine() == "arm64" else ()
    found = []
    smi = shutil.which("nvidia-smi") or (r"C:\Windows\System32\nvidia-smi.exe" if os.name == "nt" else None)
    if smi and os.path.exists(smi):
        found += [{"vendor": "nvidia", "name": n.strip(), "gfx": None}
                  for n in _run([smi, "--query-gpu=name", "--format=csv,noheader"]).splitlines() if n.strip()]
    if os.name == "nt":
        out = _run(["powershell", "-NoProfile", "-Command",
                    "Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name }"])
        for n in (n.strip() for n in out.splitlines()):
            if re.search(r"\bAMD\b|Radeon", n, re.I):
                found.append({"vendor": "amd", "name": n, "gfx": amd_gfx(n)})
            elif re.search(r"\bIntel\b.*\bArc\b", n, re.I):
                found.append({"vendor": "intel", "name": n, "gfx": None})
    else:
        for props in glob.glob("/sys/class/kfd/kfd/topology/nodes/*/properties"):
            try:
                text = open(props).read()
            except OSError:
                continue
            m = re.search(r"gfx_target_version\s+(\d+)", text)
            gfx = kfd_gfx(int(m.group(1))) if m else None
            if gfx:
                found.append({"vendor": "amd", "name": f"AMD GPU ({gfx})", "gfx": gfx})
    return tuple(found)


def best_gpu(found: tuple[dict, ...] | list[dict]) -> dict | None:
    """The GPU worth using: NVIDIA, then an AMD GPU with a known gfx target, then Apple, then Intel Arc."""
    rank = {"nvidia": 0, "amd": 1, "apple": 2, "intel": 3}
    usable = [g for g in found if g["vendor"] != "amd" or g["gfx"]]
    return min(usable, key=lambda g: rank[g["vendor"]], default=None)


def install_plan(gpu: dict | None, uv: str = "uv", python: str = sys.executable) -> tuple[str, list[str] | None]:
    """(description, uv command or None) that installs the torch build for this GPU."""
    # --no-config: the project's exclude-dependencies would otherwise skip torch here too
    base = [uv, "pip", "install", "--no-config", "--python", python,
            "--reinstall-package", "torch", "--reinstall-package", "torchvision"]
    if gpu is None:
        return ("no supported GPU found: installing the CPU build of PyTorch",
                base + ["--torch-backend", "cpu", "torch", "torchvision"])
    if gpu["vendor"] == "apple":
        return "Apple Silicon: installing PyTorch, which runs on the GPU (MPS)", base + ["torch", "torchvision"]
    if gpu["vendor"] == "nvidia":
        return (f"{gpu['name']}: installing the CUDA build of PyTorch",
                base + ["--torch-backend", "auto", "torch", "torchvision"])
    if gpu["vendor"] == "amd":
        # --index outranks PyPI; AMD labels the matching torchvision a pre-release (0.29.0a0)
        return (f"{gpu['name']}: installing AMD's ROCm build of PyTorch for {gpu['gfx']}",
                base + ["--index", AMD_INDEX, f"torch[device-{gpu['gfx']}]", "torchvision>=0.29.0a0"])
    return (f"{gpu['name']}: installing the Intel XPU build of PyTorch",
            base + ["--torch-backend", "xpu", "torch", "torchvision"])


def setup(dry_run: bool = False) -> dict:
    """Detect the GPU, install the matching PyTorch into this environment, and check that it works."""
    found = gpus()
    gpu = best_gpu(found)
    uv = os.environ.get("UV") or shutil.which("uv") or "uv"
    desc, cmd = install_plan(gpu, uv=uv)
    out = {"gpus": [g["name"] + (f" ({g['gfx']})" if g["gfx"] else "") for g in found], "plan": desc,
           "command": " ".join(cmd) if cmd else None}
    unknown_amd = [g["name"] for g in found if g["vendor"] == "amd" and not g["gfx"]]
    if unknown_amd and not gpu:
        out["note"] = (f"Unknown AMD GPU {unknown_amd}: find its gfx target and run the command above with "
                       f"torch[device-gfxNNNN] from {AMD_INDEX}")
    if cmd and not dry_run:
        print(desc, file=sys.stderr, flush=True)
        if subprocess.run(cmd).returncode != 0:
            raise SystemExit("PyTorch install failed; see the uv output above.")
    if not dry_run:
        check = _run([sys.executable, "-c", "from code_search import hardware; print(hardware.device()[1])"],
                     timeout=600)  # the first ROCm/CUDA import can take minutes
        out["torch_device"] = check.strip() or "torch failed to import"
        if cmd and out["torch_device"] in ("CPU", "torch failed to import"):
            out["warning"] = "PyTorch still can't use the GPU; check the uv output above."
            print(json.dumps(out, indent=2), file=sys.stderr)
            raise SystemExit(1)
    return out


@functools.lru_cache(maxsize=1)
def device(forced: str | None = None) -> tuple[str, str]:
    """(torch device, human description): `forced` (CODE_EMBED_DEVICE) or the best accelerator torch can use."""
    try:
        import torch
    except ImportError:
        raise RuntimeError("PyTorch isn't installed. Run `uv run code-search setup` "
                           "(a plain `uv sync` removes it; `uv run` keeps it).") from None

    if forced:
        return forced, forced
    if torch.cuda.is_available():
        # largest-memory GPU, so a discrete card wins over an integrated one
        i = max(range(torch.cuda.device_count()), key=lambda d: torch.cuda.get_device_properties(d).total_memory)
        stack = f"ROCm {torch.version.hip}" if torch.version.hip else f"CUDA {torch.version.cuda}"
        return f"cuda:{i}", f"{torch.cuda.get_device_name(i)}, {stack}"
    xpu = getattr(torch, "xpu", None)
    if xpu is not None and xpu.is_available():
        return "xpu", f"{xpu.get_device_name(0)}, XPU"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps", "Apple GPU, MPS"
    return "cpu", "CPU"


def dtype(device: str):
    """bf16 on CUDA/ROCm GPUs that support it, else fp32 (fp16 degrades these models)."""
    import torch

    return torch.bfloat16 if device.startswith("cuda") and torch.cuda.is_bf16_supported() else torch.float32


def be_gentle(device: str) -> None:
    """Keep the machine usable while the model works: below-normal CPU priority, and a cap on how much
    GPU memory this process may take (CODE_EMBED_GPU_MEMORY, a fraction; default 0.5). Hitting the cap
    makes batches smaller instead of starving games, browsers and the desktop."""
    try:
        if os.name == "nt":
            import ctypes

            k32 = ctypes.windll.kernel32
            k32.SetPriorityClass(k32.GetCurrentProcess(), 0x4000)  # BELOW_NORMAL_PRIORITY_CLASS
        else:
            os.nice(10)
    except Exception:
        pass
    frac = float(os.environ.get("CODE_EMBED_GPU_MEMORY", "0.5"))
    if not 0 < frac < 1:
        return
    try:
        import torch

        if device.startswith("cuda"):
            torch.cuda.set_per_process_memory_fraction(frac, int(device.partition(":")[2] or 0))
        elif device == "mps":
            torch.mps.set_per_process_memory_fraction(frac)
    except Exception:
        pass


def cpu_fallback_hint() -> str | None:
    """Why a GPU machine is running on CPU, and how to fix it."""
    gpu = best_gpu(gpus())
    if gpu is None or gpu["vendor"] == "apple":
        return None
    return f"{gpu['name']} found but this PyTorch build can't use it. Run `uv run code-search setup`."
