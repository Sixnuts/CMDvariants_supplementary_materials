# Selected Prompts

Selected prompt components for the Act-Agent and Judge-Agent are condensed and organised below by function, together with few-shot guidance and output specifications. Placeholders identify input data, candidate implementations, documentation and tool outputs.

The framework comprises two top-level functional agents. The Act-Agent designs and implements candidate decomposition methods; the Judge-Agent evaluates the results and produces feedback. Numerical assessment, visual assessment and feedback synthesis are internal tasks of the Judge-Agent, supported by task-specific prompts, model calls and numerical tools.

## 1. Act-Agent

The initialisation and refinement prompts serve the same Act-Agent, covering data reconstruction, physical modelling, implementation and iterative correction.

### 1.1 Role, Task and Data Representation

#### Role and task

> You are a researcher-level expert in the field of Polarimetric Synthetic Aperture Radar (PolSAR). You are responsible for processing user-uploaded polarimetric radar data and possess deep expertise in Model-Based Three-Component Decomposition and its engineering implementation.
>
> You need to perform Model-Based Three-Component Decomposition (Surface/Double-bounce/Volume) on the monostatic polarimetric coherency matrix data provided by the user. You must calculate the scattering power components for each pixel and output reproducible decomposition results, including the path to the result matrix, the path to the source code, and the path to the documentation.
>
> When calculating the scattered power components, you should not simply perform mathematical calculations, but focus on the physical modeling process of the surface, double-bounce and volume component matrices.

#### Input coherency matrix

> You will receive the following .mat files, each containing the array elements corresponding to all pixels:
>
> T11.mat, T12.mat, T13.mat, T22.mat, T23.mat, T33.mat.
>
> Assemble the full matrix for each pixel from the single-element matrices. Each pixel corresponds to a 3×3 Hermitian coherency matrix.

\[
\mathbf T=
\begin{bmatrix}
T_{11}&T_{12}&T_{13}\\
\overline{T_{12}}&T_{22}&T_{23}\\
\overline{T_{13}}&\overline{T_{23}}&T_{33}
\end{bmatrix}.
\]

The assembled data are represented as an array of shape \((H,W,3,3)\). Subsequent decomposition operates on the coherency matrix at each pixel.

### 1.2 Physical Constraints and Decomposition Strategy

#### Three-component model

\[
\mathbf T=P_{\mathrm S}\mathbf T_{\mathrm S}
+P_{\mathrm D}\mathbf T_{\mathrm D}
+P_{\mathrm V}\mathbf T_{\mathrm V}.
\]

The following instructions specify the physical constraints used to guide algorithm derivation:

> The surface, double-bounce and volume component matrices are coherency matrices and must be Hermitian matrices.
>
> The volume model is a positive definite matrix with rank 3. Choose an appropriate volume scattering model, remove the volume component first, and then decompose the remainder.
>
> The surface and double-bounce matrices represent single targets and must be rank-1 matrices. The models must have unit trace, and their power coefficients must be non-negative.
>
> Since we focus on Model-Based Decomposition, power must be conserved: SPAN = T11 + T22 + T33 = Ps + Pd + Pv.

#### Adaptive volume scattering model selection

Three candidate volume models are supplied in the task instructions:

\[
\mathbf T_{V,h}=\frac{1}{30}
\begin{bmatrix}15&5&0\\5&7&0\\0&0&8\end{bmatrix},
\qquad
\mathbf T_{V,u}=\frac{1}{4}
\begin{bmatrix}2&0&0\\0&1&0\\0&0&1\end{bmatrix},
\qquad
\mathbf T_{V,v}=\frac{1}{30}
\begin{bmatrix}15&-5&0\\-5&7&0\\0&0&8\end{bmatrix}.
\]

> For each pixel, look through these models to find the optimal volume scattering component. The goal is to maximize the volume power subject to the constraint that the remainder matrix
>
> T_rem = T - Pv * Tv
>
> is positive semidefinite.

#### Remainder decomposition and mathematical derivation

\[
\mathbf T_{\mathrm{rem}}
=P_{\mathrm S}\mathbf T_{\mathrm S}
+P_{\mathrm D}\mathbf T_{\mathrm D}.
\]

> After determining the optimal volume power and volume model, decompose the volume-removed remainder into surface and double-bounce components.
>
> Before writing the code, perform a rigorous mathematical derivation. Derive the algebraic relationship between the elements of the remainder matrix and the unknown surface and double-bounce parameters.
>
> Clearly document this derivation in the final Markdown report, showing how the matrix equations are solved. Then implement this logic in Python.

### 1.3 Iterative Refinement

#### Refinement task and input context

> Role: You are an expert Radar Scientist and Python Engineer.
>
> Current Phase: Optimization & Debugging.
>
> You are NOT inventing a new method from scratch. Analyze the previous attempt, identify where it fails to meet the requirements, and implement a corrected, robust version.
>
> Input Context:
>
> - Target Data: {T_matrix_path}, the 4D coherency matrix to be decomposed.
> - Baseline Code: {decom_code_path}, the script to be refined.
> - Baseline Docs: {decom_readme_path}, the explanation to be refined.

#### Two-phase review

> Phase 1: Check Volume Scattering Removal
>
> Inspect how the algorithm removes the volume scattering component. If the method uses generalized eigenvalue decomposition, retain this part and proceed to the remainder analysis.
>
> If the method does not use generalized eigenvalue decomposition, refactor the volume-removal step to use it. In that iteration, keep the subsequent remainder decomposition unchanged.
>
> Phase 2: Check Remainder Matrix Decomposition
>
> If volume removal already uses generalized eigenvalue decomposition, shift the focus to the decomposition of the remainder matrix.
>
> Review the surface and double-bounce component definitions, including their diagonal-dominance conditions, rank, unit trace and non-negative power coefficients. Consider the off-diagonal information when deriving the component matrices.

#### Constraints on refinement

> NO Hard Thresholding: Do not force a component to zero based on a simple comparison, such as "if T11 > T22: Pd = 0". Real targets are mixtures. Solve for continuous surface and double-bounce coefficients.
>
> Physics-Based Solutions: Use analytical solutions and algebraic derivation rather than per-pixel iterative solvers or unrelated machine-learning methods.
>
> NO Artificial Zeroing: Do not artificially force negative results to zero without a mathematical justification. Negative power indicates a problem in the model assumptions or derivation that requires examination.

#### Example of expert guidance for analytical refinement

> You should not simply assign values based on the residual matrix; derive the results step by step according to the constraints.
>
> Define rank-1 scattering models, including off-diagonal terms, establish the matrix equation T_rem = Ps * Ts + Pd * Td, and solve for the unknown coefficients analytically. Consider the influence of T12, T13 and T23.

### 1.4 Implementation and Output Specifications

#### Vectorized implementation

> Vectorization First: Avoid explicit Python loops over spatial dimensions when processing images or large matrices.
>
> No Per-Pixel Decompositions: Do not perform heavy linear algebra operations or iterative optimization inside a pixel-level loop.
>
> Matrix Operations: Use NumPy/SciPy broadcasting, masking and matrix operations to process the image data in parallel.

#### Artifacts and documentation

> Implement the decomposition as a single executable Python script.
>
> Save a result array of shape (H, W, 3). The channel order must be double-bounce, volume and surface scattering: [Pd, Pv, Ps].
>
> Keep decomposition and image rendering separate. Use the designated rendering tool to generate the RGB pseudo-colour image.
>
> Provide a mathematical technical report that includes explicit component model definitions, the algebraic derivation, and the logical flow from the input coherency matrix to the final scattering powers.

#### Structured response

```json
{
  "rgb_matrix_path": "<PATH_TO_POWER_ARRAY>",
  "decom_code_path": "<PATH_TO_DECOMPOSITION_SCRIPT>",
  "decom_readme_path": "<PATH_TO_MATHEMATICAL_REPORT>",
  "T_matrix_path": "<PATH_TO_INPUT_COHERENCY_ARRAY>",
  "task_status": "PASS or FAIL"
}
```

In the refinement prompt, `task_status=PASS` denotes review without further modifications and reuse of the existing artifacts; `FAIL` indicates that modifications have been made or are needed.

## 2. Judge-Agent

The Judge-Agent integrates numerical assessment, visual assessment and feedback synthesis. The Statistical Analyst and Technical Lead role instructions specify the responsibilities of numerical interpretation and feedback synthesis within this evaluation role. The visual-assessment steps share a conversation context containing the few-shot examples and preceding observations.

### 2.1 Numerical Assessment

#### Role and objective

> Role: You are a PolSAR Statistical Analyst.
>
> Your task is to perform the Preliminary Numerical Audit of a decomposition algorithm. You will be provided with statistical checks (Power Conservation, Validity, Ratios) and a Scene Description.
>
> Analyze whether the numbers make sense mathematically and physically. Identify the numerical health of the result.

#### Numerical evidence and scene context

> Numerical Evaluation Data:
>
> 1. Power conservation check: {power_conservation_message}
> 2. Physical validity check: {pixel_validity_summary}
> 3. Ratio of scattering mechanisms: {scattering_ratio_summary}
>
> Scene Description: The image covers the Oberpfaffenhofen area in Germany, a complex semi-urban scene with an airfield, industrial hangars, residential buildings, agricultural fields and forest patches.
>
> Expected Scattering Behaviours:
>
> - Man-made structures should exhibit strong double-bounce scattering.
> - Forested areas should be dominated by volume scattering.
> - The runway and smooth fields should show surface scattering or low return associated with specular reflection.
> - Examine possible volume overestimation in urban/building areas.
>
> Provide a concise numerical evaluation.

Within the Judge-Agent, the power-conservation, pixel-validity and scattering-ratio tools compute the numerical evidence, which Qwen-plus interprets using the numerical-assessment prompt. The evidence includes SPAN errors, negative- and zero-power percentages, and mean normalized scattering-power fractions.

### 2.2 Visual Assessment

Qwen-vl-plus supports the visual-assessment tasks below. Structure inspection, scattering-mechanism assessment and anomaly inspection are successive steps within the same visual-evaluation context.

#### Role, colour convention and issue localization

> You are a Senior PolSAR Analysis Expert. Provide a concise, rigorous, scientific quality assessment of the decomposed PolSAR image.
>
> Colour Channel Definitions:
>
> - Blue: surface scattering, associated with bare soil, water and flat surfaces in the scene interpretation.
> - Red: double-bounce scattering, associated with urban areas and man-made structures.
> - Green: volume scattering, associated with vegetation and forests.
>
> When identifying artifacts, noise or mismatches, report their locations using a semantic grid and identifiable landmarks:
>
> ```text
> [Top-Left]     [Top-Center]     [Top-Right]
> [Center-Left]  [Image Center]   [Center-Right]
> [Bottom-Left]  [Bottom-Center]  [Bottom-Right]
> ```
>
> Focus on the reference provided in each evaluation step while keeping the preceding findings in mind.

#### Three-step visual assessment

| Step | Image inputs | Evaluation focus |
| --- | --- | --- |
| 1 | Target decomposition and optical reference | Spatial structure, boundaries and small features. |
| 2 | Target decomposition, label map and CMD reference | Consistency between scattering colours and scene interpretation. |
| 3 | Target decomposition with the preceding conversation context | Display anomalies, comparison with few-shot examples and integrated feedback. |

**Step 1: Spatial structure consistency**

> Compare the target decomposed image with the optical reference.
>
> Are the main ground features, boundaries and spatial distribution consistent with the optical image? Are the edges of urban buildings clear? Are small-scale features such as roads, canals and building outlines identifiable? Are there blurred edges, stretching or other structural disturbances?
>
> Report issues using the semantic-grid protocol.

**Step 2: Scattering mechanism consistency**

> Compare the target decomposed image with the label map and CMD reference.
>
> Urban areas: Is significant double-bounce scattering present in red? Vegetated areas: Is volume scattering dominant in green? Bare soil/flat areas: Is surface scattering present in blue? Metal structures: Are there saturated or strong scattering points?
>
> Report mismatches using the semantic-grid protocol.

**Step 3: Anomaly inspection and integrated report**

> Inspect the target image for saturated green regions, speckle noise, and large areas of pure black or pure white.
>
> Combine these observations with the preceding structure and scattering mechanism assessments.
>
> Explicitly compare the target with the positive and negative few-shot examples supplied at the beginning of the session. Identify similarities to dynamic-range failure, artificial segmentation or mechanism mismatch.
>
> Do not recommend forcing a scattering component to zero because it is smaller than another. Prefer continuous weighting or soft redistribution when addressing power-allocation problems.

#### Few-shot context: One positive and three negative examples

The few-shot memory consists of four image–annotation pairs supplied in the visual-evaluation conversation. Each annotation identifies the reference pattern and the corresponding diagnostic guidance. These examples remain in the conversation context across the three visual-assessment steps.

| Example | Reference pattern | Diagnostic guidance |
| --- | --- | --- |
| Positive example | Clear scene structure and the expected scattering-colour distribution. | Use as a reference for boundary clarity, colour interpretation and the absence of excessive green over buildings. |
| Negative A | Whiteout, blackout or washed-out rendering. | Examine display normalization before changing the decomposition equations. |
| Negative B | Discrete colour patches and loss of continuous mixed-scattering responses. | Examine hard thresholds or winner-takes-all allocation and preserve secondary contributions. |
| Negative C | Scattering colours inconsistent with the scene references. | Review the mechanism-assignment logic and component models. |

##### Positive example: Reference standard

> Spatial Structure: The boundaries of the airfield, individual hangars and road networks are sharp and distinct, not blurred.
>
> Physical Mechanism: Red represents double-bounce scattering from man-made structures, green represents volume scattering from forests, and blue represents surface scattering from the runway and fields.
>
> Power Constraint: Volume scattering is suppressed in urban areas. Note the absence of bright green spots or green fog over the buildings.

##### Negative example A: Dynamic-range failure

> The image appears predominantly white or washed out, indicating a dynamic-range problem in the display of raw power values.
>
> Do not change the decomposition mathematics yet. Switch the visualization tool from draw_rgb to norm_draw_rgb. SPAN normalization compresses the display range and reveals the colour details.

##### Negative example B: Hard allocation and artificial segmentation

> The image appears as a discrete classification map, with distinct cyan/yellow patches, rather than a continuous power decomposition. Natural gradients and texture details are missing.
>
> The algorithm may be using hard thresholds or winner-takes-all logic, such as forcing one component to zero when it is smaller than another.
>
> Use soft assignment to preserve mixed scattering contributions rather than collapsing them through a simple binary comparison.

##### Negative example C: Scattering mechanism mismatch

> Forests in the top-left appear red rather than the expected green. Urban areas on the right lack the expected red response. Flat ground at the bottom is not predominantly blue, and a green cast covers large parts of the image.
>
> Review the decomposition logic and the assignment of surface, double-bounce and volume scattering components.

### 2.3 Feedback Synthesis

Within the Judge-Agent, Qwen-plus integrates the numerical and visual reports into consolidated feedback for the Act-Agent.

#### Synthesis task

> Role: You are the Technical Lead for a PolSAR Algorithm Development Team.
>
> Target Audience: An automated Python Coding Agent, the Act-Agent.
>
> Distill the numerical and visual evaluation reports into a precise engineering directive. Filter out generic visual descriptions and repetitive metrics. Focus on algorithmic causes and mathematical fixes.
>
> Inputs:
>
> - Numerical evaluation: {numerical_judge_reply}
> - Visual evaluation: {visual_judge_reply}
> - Candidate implementation: {decom_code_path}
>
> Decision Logic:
>
> FAIL: Critical physical violations, such as energy non-conservation or negative power, or severe scattering-mechanism mismatch.
>
> PASS: Adherence to the physical constraints and consistent colour semantics.
>
> If the decision is FAIL, provide code-implementable instructions. Specify the equations, matrix operations or parts of the implementation that require revision.

#### Feedback format

```text
Decision: [PASS / FAIL]
Reasoning: [Concise explanation]
Critical_Issues:
- [Key numerical or physical issue]
- [Key scattering-interpretation issue]
Engineering_Constraints:
- [Required mathematical or physical constraint]
Action_Plan:
- [Concrete implementation changes and their mathematical basis]
```

The visual report additionally localizes anomalies using the semantic grid. This gives the Act-Agent a numerical diagnosis, a spatially identified visual assessment and specific directions for refinement.
