from workers.fp32_worker import benchmark_fp32
from workers.fp16_worker import benchmark_fp16
from workers.int8_worker import benchmark_int8
from workers.onnx_worker import benchmark_onnx

__all__ = ["benchmark_fp32", "benchmark_fp16", "benchmark_int8", "benchmark_onnx"]
