#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import pathlib
import shutil
import sys
import tempfile

import litert_torch
import numpy as np
import tensorflow as tf
import torch


def _patch_broadcast_repeat(model_module):
    """Patch BCResNet blocks for litert_torch export compatibility."""

    def normal_forward(self, x):
        x1 = self.f2(x)
        x2 = torch.mean(x1, dim=2, keepdim=True)
        x2 = self.f1(x2)
        return self.activation(x + x1 + x2)

    def transition_forward(self, x):
        x = self.f2(x)
        x1 = torch.mean(x, dim=2, keepdim=True)
        x1 = self.f1(x1)
        return self.activation(x + x1)

    model_module.NormalBlock.forward = normal_forward
    model_module.TransitionBlock.forward = transition_forward


def _representative_dataset(samples: int):
    for _ in range(samples):
        sample = np.random.randn(1, 1, 40, 101).astype(np.float32)
        yield [sample]


def _summarize_tflite(path: pathlib.Path):
    interpreter = tf.lite.Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    ops = sorted({d["op_name"] for d in interpreter._get_ops_details()})
    print(f"Input dtype={input_detail['dtype']} shape={input_detail['shape']}")
    print(f"Output dtype={output_detail['dtype']} shape={output_detail['shape']}")
    print(f"Ops ({len(ops)}): {ops}")
    print(
        "Input quantization:",
        input_detail.get("quantization_parameters", {}).get("scales", []),
        input_detail.get("quantization_parameters", {}).get("zero_points", []),
    )
    print(
        "Output quantization:",
        output_detail.get("quantization_parameters", {}).get("scales", []),
        output_detail.get("quantization_parameters", {}).get("zero_points", []),
    )


def export_int8_tflite(repo_root: pathlib.Path, output_path: pathlib.Path, calib_samples: int):
    third_party_repo = repo_root / "third_party" / "bcresnet_re9ulus"
    if not third_party_repo.exists():
        raise FileNotFoundError(f"Missing repo: {third_party_repo}")

    sys.path.insert(0, str(third_party_repo))
    import bc_resnet_model  # pylint: disable=import-error

    _patch_broadcast_repeat(bc_resnet_model)
    model = bc_resnet_model.BcResNetModel(n_class=35, scale=2)
    checkpoint_path = third_party_repo / "example_model" / "model-sc-2.pt"
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    sample_input = torch.randn(1, 1, 40, 101)
    with tempfile.TemporaryDirectory(prefix="bcresnet_saved_model_") as saved_model_dir:
        _ = litert_torch.convert(
            model,
            (sample_input,),
            _saved_model_dir=saved_model_dir,
        )

        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = lambda: _representative_dataset(calib_samples)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        int8_model = converter.convert()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(int8_model)
    print(f"Wrote int8 model to: {output_path}")
    _summarize_tflite(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="tests/npusim_examples/models/bcresnet_sc35_int8.tflite",
        help="Path to output int8 TFLite model.",
    )
    parser.add_argument(
        "--calib_samples",
        type=int,
        default=128,
        help="Representative dataset samples for PTQ calibration.",
    )
    args = parser.parse_args()

    repo_root = pathlib.Path(__file__).resolve().parents[2]
    output_path = (repo_root / args.output).resolve()
    export_int8_tflite(repo_root, output_path, args.calib_samples)


if __name__ == "__main__":
    main()
