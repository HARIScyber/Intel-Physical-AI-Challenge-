import openvino as ov
import time
import numpy as np
import logging
import torch
from pathlib import Path
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def benchmark_pytorch(model, input_shape: Dict[str, Any]):
    """
    Benchmarks the PyTorch model on CPU.
    """
    model.eval()
    # Create dummy input data
    # We use the keys used in denoise_step
    inputs = {
        "prefix_pad_masks": torch.randn(*input_shape["input_0"]),
        "past_key_values": None, # Simplified
        "x_t": torch.randn(*input_shape["input_2"]),
        "timestep": torch.randn(*input_shape["input_3"]),
    }

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            model.denoise_step(**inputs)

    # Benchmark
    iterations = 50
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            model.denoise_step(**inputs)
    end_time = time.perf_counter()

    latency = (end_time - start_time) / iterations
    throughput = 1 / latency

    return {
        "latency": latency,
        "throughput": throughput,
        "device": "cpu (pytorch)",
        "precision": "fp32",
        "model": "pytorch_baseline"
    }

def benchmark_model(model_path: str, device: str, input_shape: Dict[str, Any]):
    """
    Benchmarks an OpenVINO model on a specific device.
    """
    core = ov.Core()
    model = core.read_model(model_path)
    compiled_model = core.compile_model(model, device)
    infer_request = compiled_model.create_infer_request()

    # Create dummy input data
    inputs = {}
    for input_name, shape in input_shape.items():
        inputs[input_name] = np.random.randn(*shape).astype(np.float32)

    # Warmup
    for _ in range(5):
        infer_request.infer(inputs)

    # Benchmark
    iterations = 50
    start_time = time.perf_counter()
    for _ in range(iterations):
        infer_request.infer(inputs)
    end_time = time.perf_counter()

    latency = (end_time - start_time) / iterations
    throughput = 1 / latency

    return {
        "latency": latency,
        "throughput": throughput,
        "device": device
    }

def run_benchmarks(model_dir: str = "outputs/openvino", pytorch_model=None):
    """
    Runs benchmarks for all available models on all available devices.
    """
    model_path = Path(model_dir)
    core = ov.Core()
    devices = core.available_devices
    models = [f for f in model_path.glob("*.xml")]

    # Define dummy input shapes based on SmolVLA expectations
    # This needs to match the exported model's inputs
    # For our OpenVINOExportWrapper, inputs are:
    # prefix_pad_masks, past_key_values, x_t, timestep
    input_shapes = {
        "input_0": (1, 224), # prefix_pad_masks
        "input_1": (1, 100, 512), # past_key_values (simplified)
        "input_2": (1, 10, 12), # x_t
        "input_3": (1,), # timestep
    }

    results = []

    if pytorch_model is not None:
        logger.info("Benchmarking PyTorch baseline...")
        results.append(benchmark_pytorch(pytorch_model, input_shapes))

    for model_file in models:
        precision = "fp32" if "fp32" in model_file.name else "fp16" if "fp16" in model_file.name else "int8"
        logger.info(f"Benchmarking {model_file.name} ({precision})...")

        for device in devices:
            try:
                logger.info(f"  Device: {device}")
                res = benchmark_model(str(model_file), device, input_shapes)
                res["precision"] = precision
                res["model"] = model_file.name
                results.append(res)
            except Exception as e:
                logger.error(f"  Failed on {device}: {e}")

    return results

if __name__ == "__main__":
    # In a real scenario, we'd load the checkpoint from config
    # Here we'll assume the user provides it or we use a dummy
    pytorch_model = None
    try:
        from policy.vla_policy import LeRobotVLA
        # Load with a dummy checkpoint or real one if provided in sys.argv
        import sys
        checkpoint = sys.argv[1] if len(sys.argv) > 1 else "lerobot/smolvla_base"
        policy = LeRobotVLA({"checkpoint": checkpoint})
        policy.load_checkpoint(checkpoint)
        pytorch_model = policy.model
    except Exception as e:
        logger.warning(f"Could not load PyTorch baseline: {e}")

    results = run_benchmarks(pytorch_model=pytorch_model)
    print("\n--- OpenVINO Benchmark Results ---")
    print(f"{'Model':<20} | {'Device':<15} | {'Precision':<10} | {'Latency (s)':<15} | {'Throughput (fps)':<15}")
    print("-" * 80)
    for r in results:
        print(f"{r['model']:<20} | {r['device']:<15} | {r['precision']:<10} | {r['latency']:<15.4f} | {r['throughput']:<15.4f}")
    print("----------------------------------\n")
