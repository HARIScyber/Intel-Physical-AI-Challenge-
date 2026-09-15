from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import openvino as ov
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SOURCE_ROOT = ROOT / "lerobot" / "src"
if SOURCE_ROOT.is_dir() and str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from lerobot.policies import make_policy, make_policy_config, make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks

LOGGER = logging.getLogger(__name__)

INPUT_NAMES = (
    "image",
    "img_mask",
    "lang_tokens",
    "lang_masks",
    "state",
    "x_t",
    "timestep",
)

INPUT_SHAPES = (
    (1, 3, 256, 256),
    (1,),
    (1, 48),
    (1, 48),
    (1, 32),
    (1, 50, 32),
    (1,),
)


@dataclass(frozen=True)
class BenchmarkSample:
    image: np.ndarray
    img_mask: np.ndarray
    lang_tokens: np.ndarray
    lang_masks: np.ndarray
    state: np.ndarray
    x_t: np.ndarray
    timestep: np.ndarray

    def torch_inputs(self) -> tuple[torch.Tensor, ...]:
        return (
            torch.from_numpy(self.image),
            torch.from_numpy(self.img_mask),
            torch.from_numpy(self.lang_tokens),
            torch.from_numpy(self.lang_masks),
            torch.from_numpy(self.state),
            torch.from_numpy(self.x_t),
            torch.from_numpy(self.timestep),
        )

    def openvino_inputs(self) -> tuple[np.ndarray, ...]:
        return (
            self.image,
            self.img_mask,
            self.lang_tokens,
            self.lang_masks,
            self.state,
            self.x_t,
            self.timestep,
        )


class SmolVLABenchmarkForward(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self,
        image: torch.Tensor,
        img_mask: torch.Tensor,
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: torch.Tensor,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.model.embed_prefix(
            [image],
            [img_mask],
            lang_tokens,
            lang_masks,
            state=state,
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.model.embed_suffix(x_t, timestep)
        pad_masks = torch.cat((prefix_pad_masks, suffix_pad_masks), dim=1)
        att_masks = torch.cat((prefix_att_masks, suffix_att_masks), dim=1)
        attention_mask = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        (_, suffix_out), _ = self.model.vlm_with_expert.forward(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=(prefix_embs, suffix_embs),
            use_cache=False,
        )
        suffix_out = suffix_out[:, -self.model.config.chunk_size:]
        return self.model.action_out_proj(suffix_out.to(dtype=torch.float32))


def make_benchmark_sample(seed: int = 123) -> BenchmarkSample:
    rng = np.random.default_rng(seed)
    return BenchmarkSample(
        image=rng.uniform(-1.0, 1.0, INPUT_SHAPES[0]).astype(np.float32),
        img_mask=np.ones(INPUT_SHAPES[1], dtype=np.bool_),
        lang_tokens=rng.integers(0, 49280, INPUT_SHAPES[2], dtype=np.int64),
        lang_masks=np.ones(INPUT_SHAPES[3], dtype=np.bool_),
        state=rng.standard_normal(INPUT_SHAPES[4]).astype(np.float32),
        x_t=rng.standard_normal(INPUT_SHAPES[5]).astype(np.float32),
        timestep=np.asarray([0.5], dtype=np.float32),
    )


def load_smolvla_model(checkpoint: str, device: str = "cpu") -> tuple[torch.nn.Module, Any]:
    policy_config = make_policy_config(
        "smolvla",
        pretrained_path=checkpoint,
        device=device,
    )
    policy = make_policy(policy_config)
    make_pre_post_processors(
        policy_config,
        pretrained_path=checkpoint,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    model = policy.model
    model.eval()
    return model, policy_config


def _set_input_names(model: ov.Model) -> None:
    for port, name in zip(model.inputs, INPUT_NAMES):
        port.get_tensor().set_names({name})


def _set_input_names(model: ov.Model) -> None:
    for port, name in zip(model.inputs, INPUT_NAMES):
        port.get_tensor().set_names({name})


def _artifact(path: Path, precision: str, status: str, supported: bool, message: str) -> dict[str, Any]:
    return {
        "path": path.as_posix() if path.is_relative_to(ROOT) else str(path),
        "precision": precision,
        "status": status,
        "supported": supported,
        "message": message,
        "size_bytes": path.stat().st_size if path.exists() else None,
    }


def convert_model(
    checkpoint: str,
    output_dir: str | Path = "outputs/openvino",
    device: str = "cpu",
    force: bool = False,
) -> list[dict[str, Any]]:
    output_path = Path(output_dir)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.mkdir(parents=True, exist_ok=True)

    fp32_xml = output_path / "smolvla_benchmark_fp32.xml"
    fp16_xml = output_path / "smolvla_benchmark_fp16.xml"
    int8_xml = output_path / "smolvla_benchmark_int8.xml"

    if force or not fp32_xml.exists() or not fp32_xml.with_suffix(".bin").exists():
        LOGGER.info("Loading SmolVLA for OpenVINO conversion")
        model, _ = load_smolvla_model(checkpoint, device)
        wrapper = SmolVLABenchmarkForward(model).eval()
        sample = make_benchmark_sample()
        with torch.inference_mode():
            expected = wrapper(*sample.torch_inputs())
        if tuple(expected.shape) != (1, 50, 32):
            raise RuntimeError(f"Unexpected model output shape: {tuple(expected.shape)}")
        LOGGER.info("Converting SmolVLA forward graph to OpenVINO FP32")
        ov_model = ov.convert_model(wrapper, example_input=sample.torch_inputs())
        _reshape_static(ov_model)
        _set_input_names(ov_model)
        ov.save_model(ov_model, fp32_xml)
    else:
        LOGGER.info("Reusing OpenVINO FP32 artifact: %s", fp32_xml)

    if force or not fp16_xml.exists() or not fp16_xml.with_suffix(".bin").exists():
        LOGGER.info("Creating OpenVINO FP16 artifact")
        fp16_model = _reshape_static(ov.Core().read_model(fp32_xml))
        ov.save_model(fp16_model, fp16_xml, compress_to_fp16=True)

    artifacts = [
        _artifact(fp32_xml, "fp32", "ready", True, "OpenVINO IR exported from the identical-input forward graph"),
        _artifact(fp16_xml, "fp16", "ready", True, "FP16-compressed OpenVINO IR"),
    ]

    if importlib.util.find_spec("nncf") is None:
        artifacts.append(
            _artifact(
                int8_xml,
                "int8",
                "unsupported",
                False,
                "NNCF is not installed; calibrated INT8 post-training quantization was not run",
            )
        )
        return artifacts

    try:
        import nncf

        fp32_model = _reshape_static(ov.Core().read_model(fp32_xml))
        calibration_data = [make_benchmark_sample().openvino_inputs()]
        calibration_dataset = nncf.Dataset(calibration_data)
        quantized_model = nncf.quantize(fp32_model, calibration_dataset, subset_size=1)
        _reshape_static(quantized_model)
        ov.save_model(quantized_model, int8_xml)
        artifacts.append(_artifact(int8_xml, "int8", "ready", True, "NNCF post-training quantized OpenVINO IR"))
    except Exception as exc:
        artifacts.append(
            _artifact(
                int8_xml,
                "int8",
                "unsupported",
                False,
                f"INT8 quantization failed: {exc}",
            )
        )

    return artifacts


def main() -> int:
    parser = argparse.ArgumentParser(description="Export the SmolVLA benchmark forward graph to OpenVINO")
    parser.add_argument("checkpoint", nargs="?", default="lerobot/smolvla_base")
    parser.add_argument("--output-dir", default="outputs/openvino")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    artifacts = convert_model(args.checkpoint, args.output_dir, args.device, args.force)
    for artifact in artifacts:
        print(f"{artifact['precision']}: {artifact['status']} - {artifact['message']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
