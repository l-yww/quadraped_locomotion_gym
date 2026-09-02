"""
Convert a TorchScript (.pt) policy to ONNX format.

Usage:
    python sim2sim/export_onnx.py --pt_path <path_to_pt> [--onnx_path <output_path>]

Example:
    python sim2sim/export_onnx.py \
        --pt_path logs/quadruped_wtw_slope/exported/Jun05_12-21-19_test/polices/Jun05_12-21-19_test.pt \
        --onnx_path sim2sim/models/policy.onnx
"""

import sys
from pathlib import Path

PATH_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PATH_ROOT))

import torch
import argparse
import yaml
LEGGED_GYM_ROOT_DIR = str(PATH_ROOT)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert TorchScript policy to ONNX")
    parser.add_argument("--pt_path", type=str, required=True,
                        help="Path to the .pt TorchScript policy file")
    parser.add_argument("--onnx_path", type=str, default=None,
                        help="Output ONNX path (default: same dir as .pt, with .onnx extension)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to sim2sim config YAML (to auto-detect input dims)")
    parser.add_argument("--input_dim", type=int, default=None,
                        help="Input observation dimension (auto-detected if not provided)")
    parser.add_argument("--dynamic_batch", action="store_true",
                        help="Export with dynamic batch dimension")
    return parser.parse_args()


def main():
    args = parse_args()

    pt_path = Path(args.pt_path)
    if not pt_path.exists():
        # Try relative to project root
        pt_path = Path(LEGGED_GYM_ROOT_DIR) / args.pt_path
    if not pt_path.exists():
        raise FileNotFoundError(f"Policy file not found: {args.pt_path}")

    print(f"[1/4] Loading TorchScript model from: {pt_path}")
    model = torch.jit.load(str(pt_path))
    model.eval()
    model.to("cpu")

    # Determine input dimension
    if args.input_dim:
        input_dim = args.input_dim
    elif args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            config_path = Path(LEGGED_GYM_ROOT_DIR) / args.config
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        input_dim = config["num_obs"] * 5  # num_single_obs * frame_stack
        print(f"  Input dim from config: num_obs={config['num_obs']}, frame_stack=5 → {input_dim}")
    else:
        # Try to infer by running with a fake input
        print("  Auto-detecting input dim...")
        for dim in [295, 210, 235, 59]:
            try:
                dummy = torch.zeros(1, dim)
                with torch.no_grad():
                    model(dummy)
                input_dim = dim
                print(f"  Input dim detected: {input_dim}")
                break
            except Exception:
                continue
        else:
            # Default for quadruped_wtw_slope
            input_dim = 295
            print(f"  Using default input dim: {input_dim}")

    # Create dummy input
    dummy_input = torch.zeros(1, input_dim, dtype=torch.float32)

    # Verify model runs
    print(f"[2/4] Verifying model with input shape: (1, {input_dim})")
    with torch.no_grad():
        output = model(dummy_input)
    print(f"  Output shape: {output.shape}")
    print(f"  Output sample: {output[0, :6].tolist()}...")

    # Determine output path
    if args.onnx_path:
        onnx_path = Path(args.onnx_path)
    else:
        onnx_path = pt_path.with_suffix(".onnx")
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    # Export to ONNX
    print(f"[3/4] Exporting to ONNX: {onnx_path}")
    dynamic_axes = None
    if args.dynamic_batch:
        dynamic_axes = {
            "obs": {0: "batch_size"},
            "actions": {0: "batch_size"},
        }

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=["obs"],
        output_names=["actions"],
        dynamic_axes=dynamic_axes,
        opset_version=14,
        do_constant_folding=True,
    )

    # Verify ONNX model
    print(f"[4/4] Verifying ONNX model...")
    try:
        import onnx
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        print(f"  ONNX model is valid!")
        print(f"  Input:  {onnx_model.graph.input[0].name}  shape={[d.dim_value for d in onnx_model.graph.input[0].type.tensor_type.shape.dim]}")
        print(f"  Output: {onnx_model.graph.output[0].name}  shape={[d.dim_value for d in onnx_model.graph.output[0].type.tensor_type.shape.dim]}")
    except ImportError:
        print("  (Install 'onnx' package for model verification)")

    print(f"\nDone! ONNX model saved to: {onnx_path}")
    print(f"  Input dim: {input_dim}")
    print(f"  Output dim: {output.shape[-1]}")


if __name__ == "__main__":
    main()
