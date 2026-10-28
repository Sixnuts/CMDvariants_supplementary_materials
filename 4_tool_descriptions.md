# Representative PolSAR Tools

The Act–Judge framework uses a custom PolSAR toolset for data preparation, code execution, visualization and numerical assessment. Representative tools are described below.

| Tool | Function | Brief description |
| --- | --- | --- |
| `read_mat_file` | Data loading | Reads a coherency-matrix channel from a MATLAB `.mat` file and saves it as a NumPy array for subsequent processing. |
| `assemble_T_matrix` | Coherency-matrix reconstruction | Combines the six independent channels into an array of shape `(H, W, 3, 3)`, filling the lower-triangular entries by complex conjugation. |
| `save_code_to_file` | Script generation and execution | Saves the generated Python code, executes the script and returns execution information for inspection and refinement. |
| `norm_draw_rgb` | Normalized pseudo-colour rendering | Normalizes each pixel's scattering powers by their sum and generates an RGB image with double-bounce, volume and surface scattering assigned to red, green and blue, respectively. |
| `check_power_invariance` | Power-conservation assessment | Compares the input SPAN with `Ps + Pd + Pv` and reports the mean absolute error, root mean square error and maximum absolute error. |
| `analyze_pixel_validity` | Abnormal-power diagnosis | Calculates negative- and zero-power percentages for each component and identifies regions with concentrated negative-power pixels. |
