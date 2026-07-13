#!/usr/bin/env python3
"""Diagnose the TensorRT dependency stack and flag known-bad combinations.

Companion to ``tensorrt_versioning.md``. TensorRT's runtime behaviour is decided by a
whole chain — GPU driver -> CUDA -> cuDNN -> TensorRT -> the serialized engine (.plan) —
and a mismatch anywhere fails at deserialize time, often with a cryptic message. Run this
FIRST on any new machine/container/Jetson to print the full stack and catch mismatches
before they cost you an afternoon.

    python check_trt_env.py                 # dump the version stack + verdicts
    python check_trt_env.py --engine m.plan # also test whether a .plan deserializes here

Pure standard library + optional imports (torch / tensorrt are used only if present), so it
runs anywhere: an x86 CPU box (reports what is missing) or a full Jetson.
"""
from __future__ import annotations

import argparse
import platform
import re
import subprocess


def _run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        return (out.stdout + out.stderr).strip() or None
    except (FileNotFoundError, subprocess.SubprocessError):
        return None


def _first(pattern: str, text: str | None) -> str | None:
    if not text:
        return None
    m = re.search(pattern, text)
    return m.group(1) if m else None


def collect() -> dict:
    info: dict[str, str | None] = {}
    info["python"] = platform.python_version()
    info["os"] = f"{platform.system()} {platform.release()}"
    info["arch"] = platform.machine()  # x86_64 vs aarch64 (Jetson)

    # --- GPU driver + CUDA driver API version (from nvidia-smi) ---
    smi = _run(["nvidia-smi", "--query-gpu=driver_version,name,compute_cap",
                "--format=csv,noheader"])
    if smi:
        parts = [p.strip() for p in smi.splitlines()[0].split(",")]
        info["driver"] = parts[0] if len(parts) > 0 else None
        info["gpu"] = parts[1] if len(parts) > 1 else None
        info["compute_cap"] = parts[2] if len(parts) > 2 else None
    info["cuda_driver_api"] = _first(r"CUDA Version:\s*([\d.]+)", _run(["nvidia-smi"]))

    # --- CUDA toolkit (nvcc) ---
    info["cuda_toolkit"] = _first(r"release ([\d.]+)", _run(["nvcc", "--version"]))

    # --- TensorRT (Python binding) + libnvinfer soname on the loader path ---
    try:
        import tensorrt as trt  # type: ignore
        info["tensorrt_python"] = trt.__version__
    except Exception:
        info["tensorrt_python"] = None
    ldc = _run(["ldconfig", "-p"])
    if ldc:
        sos = sorted({m for m in re.findall(r"libnvinfer\.so\.[\d.]+", ldc)})
        info["libnvinfer_soname"] = ", ".join(sos) if sos else None

    # --- cuDNN + framework versions (via torch if available) ---
    try:
        import torch  # type: ignore
        info["torch"] = torch.__version__
        info["torch_cuda_build"] = torch.version.cuda
        cudnn = torch.backends.cudnn.version()
        info["cudnn"] = str(cudnn) if cudnn else None
    except Exception:
        info["torch"] = info.get("torch")
    for mod in ("torchvision", "onnx", "onnxruntime"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = None

    # --- JetPack / L4T (Jetson only) ---
    tegra = _run(["cat", "/etc/nv_tegra_release"])
    info["l4t"] = _first(r"R(\d+)", tegra)
    jp = _run(["dpkg-query", "-W", "-f=${Version}", "nvidia-jetpack"])
    info["jetpack"] = jp if jp and "no packages" not in jp.lower() else None

    return info


# Reference: what each JetPack pins (you do NOT pick these independently on Jetson).
JETPACK_MATRIX = [
    # jetpack,  cuda,     cudnn,  tensorrt,      notes
    ("5.x", "11.4", "8.x", "8.4-8.6", "Orin, TRT 8 series"),
    ("6.0", "12.2", "8.9", "8.6", "Orin, still TRT 8"),
    ("6.2 / 6.2.1", "12.6", "9.3", "10.3", "Orin, jumps to TRT 10"),
    ("7.x", "13.x", "9.x", "10.13-10.16", "Thor (Blackwell, CC 11.0), CUDA 13"),
]


def verdicts(info: dict) -> list[str]:
    out: list[str] = []
    trt = info.get("tensorrt_python")
    arch = info.get("arch")
    cc = info.get("compute_cap")

    if trt is None:
        out.append("NOTE  TensorRT Python binding not importable here "
                   "(expected on an x86 CPU box; on Jetson it ships via JetPack apt, not pip).")
    else:
        major = trt.split(".")[0]
        if major == "10" and (info.get("cuda_toolkit") or "").startswith("11"):
            out.append("WARN  TensorRT 10 requires CUDA 12.x, but nvcc reports CUDA 11.x — "
                       "mismatch. TRT 10 will not load against a CUDA 11 toolkit.")
        # Jetson Thor FP8/FP4 silent-fallback trap (issue #4590)
        if cc == "11.0" and trt.startswith("10.13"):
            out.append("WARN  Thor (CC 11.0) + TensorRT 10.13.x: FP8/FP4 BuilderFlags may be "
                       "silently ignored and build FP32 (issue #4590). Verify actual layer "
                       "precision; consider a newer TRT point release.")

    if arch == "aarch64" and info.get("jetpack") is None:
        out.append("NOTE  aarch64 but nvidia-jetpack package not found — this may be a non-Jetson "
                   "ARM host, or JetPack metadata is missing.")

    if trt and info.get("libnvinfer_soname"):
        so = info["libnvinfer_soname"]
        trt_major = trt.split(".")[0]
        if trt_major not in so:  # e.g. python trt 10.x but only libnvinfer.so.8 on the path
            out.append(f"WARN  tensorrt python {trt} but libnvinfer soname '{so}' — a mixed "
                       "install (pip wheel vs system .so) can crash at load. Align them.")

    if not out:
        out.append("OK    no obvious version conflicts detected in the collected stack.")
    return out


def test_engine(path: str) -> None:
    """Try to deserialize a .plan here; surface the version-mismatch message if any."""
    try:
        import tensorrt as trt  # type: ignore
    except Exception:
        print(f"\n[engine] cannot test {path}: TensorRT not importable here.")
        return
    logger = trt.Logger(trt.Logger.ERROR)
    try:
        with open(path, "rb") as f:
            data = f.read()
        eng = trt.Runtime(logger).deserialize_cuda_engine(data)
        if eng is None:
            print(f"\n[engine] {path}: deserialize returned None — likely a version/arch "
                  "mismatch. Look for 'expecting library version X got Y' above and REBUILD.")
        else:
            print(f"\n[engine] {path}: deserialized OK with TensorRT {trt.__version__}.")
    except Exception as e:  # noqa: BLE001
        print(f"\n[engine] {path}: {e}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine", help="path to a .plan/.engine to test-deserialize here")
    args = ap.parse_args()

    info = collect()
    order = ["python", "os", "arch", "gpu", "compute_cap", "driver", "cuda_driver_api",
             "cuda_toolkit", "cudnn", "tensorrt_python", "libnvinfer_soname",
             "torch", "torch_cuda_build", "torchvision", "onnx", "onnxruntime",
             "l4t", "jetpack"]
    print("=== TensorRT dependency stack ===")
    for k in order:
        print(f"  {k:20s}: {info.get(k)}")

    print("\n=== JetPack pin reference (Jetson) ===")
    print(f"  {'jetpack':14s} {'cuda':6s} {'cudnn':6s} {'tensorrt':12s} notes")
    for jp, cuda, cudnn, trt_, notes in JETPACK_MATRIX:
        print(f"  {jp:14s} {cuda:6s} {cudnn:6s} {trt_:12s} {notes}")

    print("\n=== verdicts ===")
    for line in verdicts(info):
        print(f"  {line}")

    if args.engine:
        test_engine(args.engine)


if __name__ == "__main__":
    main()
