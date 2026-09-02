"""
Verify that .pt and .onnx models produce identical outputs for the same input.

Usage:
    python sim2sim/verify_pt_onnx.py

Checks logs/quadruped_wtw_slope/exported/Jun05_12-21-19_test/polices/
"""
import sys
from pathlib import Path
PATH_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PATH_ROOT))

import torch
import numpy as np

PT_PATH = "logs/quadruped_wtw_slope/exported/负载-抬腿-ptich-对称性loss-高度-站立/polices/负载-抬腿-ptich-对称性loss-高度-站立_model_14900.pt"
ONNX_PATH = "logs/quadruped_wtw_slope/exported/负载-抬腿-ptich-对称性loss-高度-站立/polices/负载-抬腿-ptich-对称性loss-高度-站立_model_14900.onnx"
INPUT_DIM = 295  # frame_stack=5 × num_single_obs=59

def main():
    # Load models
    print(f"Loading PT:  {PT_PATH}")
    pt_model = torch.jit.load(PT_PATH)
    pt_model.eval()
    pt_model.to("cpu")

    print(f"Loading ONNX: {ONNX_PATH}")
    import onnx
    import onnxruntime as ort
    onnx_model = onnx.load(ONNX_PATH)
    onnx.checker.check_model(onnx_model)
    ort_session = ort.InferenceSession(str(ONNX_PATH))

    # Generate test inputs
    torch.manual_seed(42)
    np.random.seed(42)

    test_inputs = [
        torch.randn(1, INPUT_DIM, dtype=torch.float32),           # random
        torch.zeros(1, INPUT_DIM, dtype=torch.float32),            # zeros
        torch.ones(1, INPUT_DIM, dtype=torch.float32),             # ones
        torch.randn(1, INPUT_DIM, dtype=torch.float32) * 0.1,     # small random
        torch.randn(1, INPUT_DIM, dtype=torch.float32) * 10.0,    # large random
    ]

    print(f"\n{'='*70}")
    print(f"{'Test':<10} {'Max diff':<15} {'Mean diff':<15} {'Match':<8}")
    print(f"{'='*70}")
    all_pass = True
    for idx, inp in enumerate(test_inputs):
        # PT forward
        with torch.no_grad():
            pt_out = pt_model(inp).cpu().numpy()

        # ONNX forward
        onnx_out = ort_session.run(None, {"obs": inp.numpy()})[0]

        max_diff = np.max(np.abs(pt_out - onnx_out))
        mean_diff = np.mean(np.abs(pt_out - onnx_out))
        match = "✓" if max_diff < 1e-4 else "✗ FAIL"
        if max_diff >= 1e-4:
            all_pass = False
        print(f"{idx+1:<10} {max_diff:<15.6e} {mean_diff:<15.6e} {match:<8}")
    print(f"{'='*70}")

    # Also show sample outputs
    print(f"\nSample output comparison (test 0):")
    print(f"  PT   [:6]: {pt_out[0, :6]}")
    print(f"  ONNX [:6]: {onnx_out[0, :6]}")

    if all_pass:
        print("\n✅ PT and ONNX outputs are IDENTICAL (max diff < 1e-4)")
    else:
        print("\n❌ PT and ONNX outputs DIFFER! Check above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
