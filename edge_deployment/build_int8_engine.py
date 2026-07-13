#!/usr/bin/env python3
"""Build and benchmark an INT8 TensorRT engine for a small AD-perception model on Jetson.

Companion script to ``tensorrt_quantization_jetson.md``. Mirrors the role of
``driving_benchmarks/run_pdm_singlestage.py``: a device-side hands-on script, NOT run in
the shared CPU uv environment (TensorRT requires CUDA + aarch64 on Jetson).

End-to-end flow (run on the Jetson):

    # 0. one-time: create a small calibration set of representative inputs
    python build_int8_engine.py calib --n 256 --out calib/

    # 1. export the PyTorch perception model to ONNX (runs anywhere torch is installed)
    python build_int8_engine.py export --onnx tinybev.onnx

    # 2. build INT8 + FP16 engines and compare (needs tensorrt on the Jetson)
    python build_int8_engine.py build --onnx tinybev.onnx --calib calib/ --int8 --engine tinybev_int8.plan
    python build_int8_engine.py build --onnx tinybev.onnx --fp16               --engine tinybev_fp16.plan

    # 3. benchmark latency/throughput
    python build_int8_engine.py bench --engine tinybev_int8.plan
    python build_int8_engine.py bench --engine tinybev_fp16.plan

trtexec equivalents (quickest path, no Python needed for build/bench):

    trtexec --onnx=tinybev.onnx --fp16 --saveEngine=tinybev_fp16.plan
    trtexec --onnx=tinybev.onnx --int8 --calib=calib.cache --saveEngine=tinybev_int8.plan
    trtexec --loadEngine=tinybev_int8.plan --iterations=1000 --avgRuns=1000

Tested locally (x86, CPU torch): the model forward pass and ONNX export. The build/bench
paths require a Jetson and are intentionally guarded so importing this file never needs
tensorrt/pycuda.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

# ----------------------------------------------------------------------------------------
# 1. A small AD-perception model: a compact camera -> BEV-ish semantic segmentation head.
#    Stands in for a real perception backbone (ResNet/RegNet + seg/occupancy head).
# ----------------------------------------------------------------------------------------
INPUT_SHAPE = (1, 3, 256, 512)   # (N, C, H, W) — a downsized front-camera frame
NUM_CLASSES = 10                 # e.g. road / lane / vehicle / pedestrian / ...


def build_model():
    import torch
    import torch.nn as nn

    def block(cin, cout, stride):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    class TinyBEVSegNet(nn.Module):
        """4-stage conv encoder + a 1x1 segmentation head (output stride 8)."""

        def __init__(self, num_classes=NUM_CLASSES):
            super().__init__()
            self.stem = block(3, 32, stride=2)     # /2
            self.enc1 = block(32, 64, stride=2)    # /4
            self.enc2 = block(64, 128, stride=2)   # /8
            self.enc3 = block(128, 128, stride=1)
            self.head = nn.Conv2d(128, num_classes, 1)

        def forward(self, x):
            x = self.stem(x)
            x = self.enc1(x)
            x = self.enc2(x)
            x = self.enc3(x)
            return self.head(x)               # (N, num_classes, H/8, W/8) logits

    return TinyBEVSegNet()


# ----------------------------------------------------------------------------------------
# 2. Calibration set: representative inputs the INT8 calibrator observes to pick scales.
#    In practice, dump real preprocessed frames here instead of random noise.
# ----------------------------------------------------------------------------------------
def cmd_calib(args):
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(0)
    for i in range(args.n):
        # Placeholder: replace with real normalized camera frames from your dataset.
        x = rng.standard_normal(INPUT_SHAPE, dtype=np.float32)
        np.save(os.path.join(args.out, f"calib_{i:04d}.npy"), x)
    print(f"wrote {args.n} calibration tensors to {args.out}/  "
          f"(REPLACE with real frames for meaningful scales)")


# ----------------------------------------------------------------------------------------
# 3. ONNX export (runs anywhere torch is installed — verified on x86 CPU torch).
# ----------------------------------------------------------------------------------------
def cmd_export(args):
    import torch

    model = build_model().eval()
    dummy = torch.randn(*INPUT_SHAPE)
    torch.onnx.export(
        model, dummy, args.onnx,
        input_names=["image"], output_names=["seg_logits"],
        opset_version=17, dynamo=False,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"exported {args.onnx}  ({n_params/1e6:.2f}M params, input {INPUT_SHAPE})")


# ----------------------------------------------------------------------------------------
# 4. INT8 entropy calibrator (IInt8EntropyCalibrator2) — the class TensorRT calls to
#    estimate per-tensor activation scales from the calibration set. Jetson-only.
# ----------------------------------------------------------------------------------------
def _make_calibrator(calib_dir, cache_path):
    import tensorrt as trt
    import pycuda.autoinit  # noqa: F401  (initializes CUDA context)
    import pycuda.driver as cuda

    files = sorted(glob.glob(os.path.join(calib_dir, "*.npy")))
    if not files:
        raise FileNotFoundError(f"no *.npy calibration tensors in {calib_dir}")

    class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self):
            super().__init__()
            self.files = files
            self.idx = 0
            self.cache = cache_path
            self.dbytes = int(np.prod(INPUT_SHAPE)) * 4
            self.dptr = cuda.mem_alloc(self.dbytes)

        def get_batch_size(self):
            return INPUT_SHAPE[0]

        def get_batch(self, names):
            if self.idx >= len(self.files):
                return None
            x = np.load(self.files[self.idx]).astype(np.float32)
            cuda.memcpy_htod(self.dptr, np.ascontiguousarray(x))
            self.idx += 1
            return [int(self.dptr)]

        def read_calibration_cache(self):
            if os.path.exists(self.cache):
                with open(self.cache, "rb") as f:
                    return f.read()
            return None

        def write_calibration_cache(self, cache):
            with open(self.cache, "wb") as f:
                f.write(cache)

    return EntropyCalibrator()


def cmd_build(args):
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    with open(args.onnx, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("ONNX parse failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 GiB
    if args.fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    if args.int8:
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = _make_calibrator(args.calib, args.cache)

    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise RuntimeError("engine build failed")
    with open(args.engine, "wb") as f:
        f.write(engine)
    print(f"wrote {args.engine}  (int8={args.int8}, fp16={args.fp16})")


# ----------------------------------------------------------------------------------------
# 5. Latency/throughput benchmark. Jetson-only. Lock clocks first: sudo jetson_clocks.
# ----------------------------------------------------------------------------------------
def cmd_bench(args):
    import time
    import tensorrt as trt
    import pycuda.autoinit  # noqa: F401
    import pycuda.driver as cuda

    logger = trt.Logger(trt.Logger.WARNING)
    with open(args.engine, "rb") as f:
        engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    in_shape = INPUT_SHAPE
    out_shape = (INPUT_SHAPE[0], NUM_CLASSES, INPUT_SHAPE[2] // 8, INPUT_SHAPE[3] // 8)
    d_in = cuda.mem_alloc(int(np.prod(in_shape)) * 4)
    d_out = cuda.mem_alloc(int(np.prod(out_shape)) * 4)
    bindings = [int(d_in), int(d_out)]
    stream = cuda.Stream()

    x = np.ascontiguousarray(np.random.randn(*in_shape).astype(np.float32))
    cuda.memcpy_htod(d_in, x)

    for _ in range(50):  # warmup
        context.execute_v2(bindings)
    cuda.Context.synchronize()

    t0 = time.perf_counter()
    N = args.iterations
    for _ in range(N):
        context.execute_v2(bindings)
    cuda.Context.synchronize()
    dt = (time.perf_counter() - t0) / N * 1e3
    print(f"{args.engine}: {dt:.3f} ms/inf   {1000.0/dt:.1f} inf/s   (N={N})")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("calib"); p.add_argument("--n", type=int, default=256)
    p.add_argument("--out", default="calib/"); p.set_defaults(func=cmd_calib)

    p = sub.add_parser("export"); p.add_argument("--onnx", default="tinybev.onnx")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("build")
    p.add_argument("--onnx", default="tinybev.onnx")
    p.add_argument("--engine", default="tinybev.plan")
    p.add_argument("--calib", default="calib/")
    p.add_argument("--cache", default="calib.cache")
    p.add_argument("--int8", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("bench"); p.add_argument("--engine", default="tinybev.plan")
    p.add_argument("--iterations", type=int, default=1000); p.set_defaults(func=cmd_bench)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
