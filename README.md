# Supplementary Materials: Act–Judge Framework for PolSAR Decomposition

This package provides implementations of three complete model-based decomposition (CMD) variants, selected Act–Judge prompts, model information and representative tool descriptions. The materials support understanding and further exploration of physics-constrained polarimetric synthetic aperture radar (PolSAR) decomposition.

## Package Contents

| File | Description |
| --- | --- |
| [1_cmd_variants.py](1_cmd_variants.py) | Standalone implementations of SGSD-CMD, PGPA-CMD and DSP-CMD, with input/output documentation and a synthetic demonstration. |
| [2_selected_prompts.md](2_selected_prompts.md) | Selected instructions for the two top-level functional agents: the Act-Agent and Judge-Agent. Includes physical constraints, iterative refinement, numerical and visual assessment, few-shot guidance and feedback synthesis. |
| [3_model_configuration.md](3_model_configuration.md) | Qwen-Plus and Qwen-VL-Plus model information, API access, and the main experimental period of November 2025 to February 2026. |
| [4_tool_descriptions.md](4_tool_descriptions.md) | Examples of tools for data loading, coherency-matrix reconstruction, code execution, normalized visualization and numerical assessment. |

The Python implementation runs independently with NumPy. API access is not required to apply the three CMD variants. The Markdown files provide methodological context and can be read independently.

## Quick Start

Use Python 3 with NumPy installed. From this folder, run:

```bash
python -m pip install numpy
python 1_cmd_variants.py
```

The script generates a small synthetic coherency array and prints the output shape and maximum absolute SPAN error for each method. This demonstration illustrates the implementation interface; supply your own coherency data for a real scene.

## Decomposition Functions

| Function | Method |
| --- | --- |
| `decompose1(T)` | Sigmoid-Guided Soft Decision CMD (SGSD-CMD) |
| `decompose2(T)` | Prior-Guided Proportional Allocation CMD (PGPA-CMD) |
| `decompose3(T)` | Dominant-Scattering-Preserving CMD (DSP-CMD) |

**Input:** `T` is a floating-point or complex NumPy array of shape `(H, W, 3, 3)`. Each pixel represents a finite Hermitian positive semidefinite coherency matrix in the Pauli basis. Supply linear power values. Data loading, cropping, filtering and orientation compensation are performed separately from these functions.

**Output:** Each function returns an array of shape `(H, W, 3)` containing linear scattering powers in the order **`[Pd, Pv, Ps]`**: double-bounce, volume and surface scattering. These are raw powers before display normalization.

The three methods use the same volume-extraction formulas when supplied with the same input array. Their differences concern the allocation of the remainder power between surface and double-bounce scattering.

### Applying a Method to Your Data

The example below loads the functions from the numbered Python file using the standard-library `runpy` module. Run it from this folder, with your assembled coherency array saved as `T_matrix.npy`:

```python
import runpy
import numpy as np

methods = runpy.run_path("1_cmd_variants.py")
T = np.load("T_matrix.npy", allow_pickle=False)

powers = methods["decompose1"](T)  # SGSD-CMD
# Use "decompose2" for PGPA-CMD or "decompose3" for DSP-CMD.

Pd = powers[..., 0]
Pv = powers[..., 1]
Ps = powers[..., 2]

np.save("SGSD_CMD_powers.npy", powers)
```

Preserve the raw power arrays for numerical analysis. For comparisons between methods, apply the same preprocessing and provide the same coherency array to all three functions.
