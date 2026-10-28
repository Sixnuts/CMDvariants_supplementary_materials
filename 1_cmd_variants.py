"""Standalone implementations of three complete model-based decomposition variants.

Dependency
-----------------
Python 3 and NumPy. Verification environment: Python 3.13.5, NumPy 2.1.3.
This is the verification environment, not a recovered historical environment.

Input
------------
T : numpy.ndarray, shape (H, W, 3, 3), real or complex floating point.
    Each pixel is a finite Hermitian positive semidefinite coherency matrix
    in the Pauli basis: k = [S_HH+S_VV, S_HH-S_VV, 2*S_HV] / sqrt(2).
    T = E[k k^H]. Supply linear power, not dB values or display RGB values.
    Preprocessing, orientation compensation and MAT loading are caller choices;
    these functions do not perform any of them or change T in place.

Output
-------------
powers : numpy.ndarray, shape (H, W, 3), linear scattering powers.
    Channel 0 = Pd (double-bounce), channel 1 = Pv (volume),
    channel 2 = Ps (surface). The channel order is [Pd, Pv, Ps].
    Although the original variable is called rgb_matrix, this is RAW POWER,
    not normalized RGB. Preserve it before any display normalization.

Methods
--------------
decompose1(T): SGSD-CMD, Sigmoid-Guided Soft Decision CMD (k = 100).
decompose2(T): PGPA-CMD, Prior-Guided Proportional Allocation CMD.
decompose3(T): DSP-CMD, Dominant-Scattering-Preserving CMD.

Example
------------------
    import numpy as np
    from cmd_variants import decompose1, decompose2, decompose3
    T = np.load("T_matrix.npy", allow_pickle=False)  # your own assembled T
    powers = decompose2(T)
    Pd, Pv, Ps = powers[..., 0], powers[..., 1], powers[..., 2]
    span = np.real(np.trace(T, axis1=-2, axis2=-1))
    pd_fraction = np.divide(Pd, span, out=np.full_like(Pd, np.nan),
                            where=span > 0)
    # For display only; this creates a new array and leaves powers unchanged:
    total = powers.sum(axis=-1, keepdims=True)
    rgb = np.divide(powers, total, out=np.zeros_like(powers), where=total > 0)

"""

import numpy as np

def decompose1(T):
    """SGSD-CMD: T (H,W,3,3) -> raw powers (H,W,3), ordered [Pd,Pv,Ps]."""
    # Load the T-matrix (shape: H, W, 3, 3)
    H, W, _, _ = T.shape
    
    # Define volume scattering models (Hermitian, positive definite, trace=1)
    T_v_h = np.array([[15, 5, 0],
                     [5, 7, 0],
                     [0, 0, 8]], dtype=np.float64) / 30.0
    T_v_u = np.array([[2, 0, 0],
                     [0, 1, 0],
                     [0, 0, 1]], dtype=np.float64) / 4.0
    T_v_v = np.array([[15, -5, 0],
                     [-5, 7, 0],
                     [0, 0, 8]], dtype=np.float64) / 30.0
    
    T_v_models = np.stack([T_v_h, T_v_u, T_v_v], axis=0)
    
    T_flat = T.reshape(-1, 3, 3)
    N = T_flat.shape[0]
    

    P_V_candidates = np.zeros((N, 3), dtype=np.float64)
    
    for idx_model, T_v in enumerate([T_v_h, T_v_u, T_v_v]):
        try:
            T_v_inv = np.linalg.inv(T_v)
        except np.linalg.LinAlgError:
            T_v_reg = T_v + 1e-12 * np.eye(3)
            T_v_inv = np.linalg.inv(T_v_reg)

        M = np.einsum('nij,jk->nik', T_flat, T_v_inv)
        evals = np.linalg.eigvals(M) # (N, 3)
        min_evals = np.min(np.real(evals), axis=1)
        P_V_candidates[:, idx_model] = np.clip(min_evals, 0, None)

    best_idx = np.argmax(P_V_candidates, axis=1)
    
    P_V_flat = P_V_candidates[np.arange(N), best_idx]
    T_v_best = T_v_models[best_idx] # (N, 3, 3)
    

    T_rem_flat = T_flat - P_V_flat[:, None, None] * T_v_best
    T_rem_flat = (T_rem_flat + np.transpose(T_rem_flat, (0, 2, 1)).conj()) / 2.0
    evals_rem, evecs_rem = np.linalg.eigh(T_rem_flat) 

    lambda1 = evals_rem[:, 2] # max
    lambda2 = evals_rem[:, 1] 
    k1 = evecs_rem[:, :, 2]
    k2 = evecs_rem[:, :, 1]
    mag_k1_0 = np.abs(k1[:, 0])
    mag_k1_1 = np.abs(k1[:, 1])
    mag_k2_0 = np.abs(k2[:, 0])
    mag_k2_1 = np.abs(k2[:, 1])
    term_diff1 = (np.square(mag_k1_0) - np.square(mag_k1_1))
    term_diff2 = (np.square(mag_k2_0) - np.square(mag_k2_1))
    k = 100.0 # Softening parameter

    exp_val1 = np.exp(-k * term_diff1)
    alpha1 = 1.0 / (1.0 + exp_val1)
    exp_val2 = np.exp(-k * term_diff2)
    alpha2 = 1.0 / (1.0 + exp_val2)
    
    P_S_flat = alpha1 * lambda1 + alpha2 * lambda2
    P_D_flat = (1.0 - alpha1) * lambda1 + (1.0 - alpha2) * lambda2
    

    P_S = P_S_flat.reshape(H, W)
    P_D = P_D_flat.reshape(H, W)
    P_V = P_V_flat.reshape(H, W)
    rgb_matrix = np.stack([P_D, P_V, P_S], axis=-1)
    
    return rgb_matrix

def decompose2(T):
    """PGPA-CMD: T (H,W,3,3) -> raw powers (H,W,3), ordered [Pd,Pv,Ps]"""

    H, W, _, _ = T.shape
    T_v_h = np.array([[15, 5, 0],
                     [5, 7, 0],
                     [0, 0, 8]], dtype=np.float64) / 30.0
    T_v_u = np.array([[2, 0, 0],
                     [0, 1, 0],
                     [0, 0, 1]], dtype=np.float64) / 4.0
    T_v_v = np.array([[15, -5, 0],
                     [-5, 7, 0],
                     [0, 0, 8]], dtype=np.float64) / 30.0
    
    T_v_models = np.stack([T_v_h, T_v_u, T_v_v], axis=0)
    T_flat = T.reshape(-1, 3, 3)
    N = T_flat.shape[0]
    P_V_candidates = np.zeros((N, 3), dtype=np.float64)
    
    for idx_model, T_v in enumerate([T_v_h, T_v_u, T_v_v]):
        try:
            T_v_inv = np.linalg.inv(T_v)
        except np.linalg.LinAlgError:
            T_v_reg = T_v + 1e-12 * np.eye(3)
            T_v_inv = np.linalg.inv(T_v_reg)
        M = np.einsum('nij,jk->nik', T_flat, T_v_inv)

        evals = np.linalg.eigvals(M) # (N, 3)

        min_evals = np.min(np.real(evals), axis=1)
        P_V_candidates[:, idx_model] = np.clip(min_evals, 0, None)

    best_idx = np.argmax(P_V_candidates, axis=1)
    P_V_flat = P_V_candidates[np.arange(N), best_idx]
    T_v_best = T_v_models[best_idx] # (N, 3, 3)
    

    T_rem_flat = T_flat - P_V_flat[:, None, None] * T_v_best
    T_rem_flat = (T_rem_flat + np.transpose(T_rem_flat, (0, 2, 1)).conj()) / 2.0
    evals_rem, evecs_rem = np.linalg.eigh(T_rem_flat) 
    

    lambda1 = evals_rem[:, 2]
    lambda2 = evals_rem[:, 1]
    k1 = evecs_rem[:, :, 2]
    k2 = evecs_rem[:, :, 1]
    mag_k1_0 = np.abs(k1[:, 0])
    mag_k1_1 = np.abs(k1[:, 1])
    mag_k2_0 = np.abs(k2[:, 0])
    mag_k2_1 = np.abs(k2[:, 1])

    term_diff1 = np.square(mag_k1_0) / (np.square(mag_k1_0) + np.square(mag_k1_1))
    term_diff2 = np.square(mag_k2_0) / (np.square(mag_k2_0) + np.square(mag_k2_1))
    
    alpha1 = term_diff1
    alpha2 = term_diff2   
    P_S_flat = alpha1 * lambda1 + alpha2 * lambda2
    P_D_flat = (1.0 - alpha1) * lambda1 + (1.0 - alpha2) * lambda2
    
    P_S = P_S_flat.reshape(H, W)
    P_D = P_D_flat.reshape(H, W)
    P_V = P_V_flat.reshape(H, W)
    rgb_matrix = np.stack([P_D, P_V, P_S], axis=-1)
    
    return rgb_matrix

def decompose3(T):
    """DSP-CMD: T (H,W,3,3) -> raw powers (H,W,3), ordered [Pd,Pv,Ps]."""

    H, W, _, _ = T.shape
    T_v_h = np.array([[15, 5, 0],
                     [5, 7, 0],
                     [0, 0, 8]], dtype=np.float64) / 30.0
    T_v_u = np.array([[2, 0, 0],
                     [0, 1, 0],
                     [0, 0, 1]], dtype=np.float64) / 4.0
    T_v_v = np.array([[15, -5, 0],
                     [-5, 7, 0],
                     [0, 0, 8]], dtype=np.float64) / 30.0
    T_v_models = np.stack([T_v_h, T_v_u, T_v_v], axis=0)
    
    T_flat = T.reshape(-1, 3, 3)
    N = T_flat.shape[0]
    

    P_V_candidates = np.zeros((N, 3), dtype=np.float64)
    
    for idx_model, T_v in enumerate([T_v_h, T_v_u, T_v_v]):
        try:
            T_v_inv = np.linalg.inv(T_v)
        except np.linalg.LinAlgError:
            T_v_reg = T_v + 1e-12 * np.eye(3)
            T_v_inv = np.linalg.inv(T_v_reg)
        M = np.einsum('nij,jk->nik', T_flat, T_v_inv)
        
        evals = np.linalg.eigvals(M) # (N, 3)
        min_evals = np.min(np.real(evals), axis=1)
        P_V_candidates[:, idx_model] = np.clip(min_evals, 0, None)


    best_idx = np.argmax(P_V_candidates, axis=1)
    P_V_flat = P_V_candidates[np.arange(N), best_idx]
    T_v_best = T_v_models[best_idx] # (N, 3, 3)
    T_rem_flat = T_flat - P_V_flat[:, None, None] * T_v_best
    
    T_3 = (T_rem_flat + np.transpose(T_rem_flat, (0, 2, 1)).conj()) / 2.0
    
    t3_11 = np.real(T_3[:, 0, 0])
    t3_22 = np.real(T_3[:, 1, 1])
    t3_33 = np.real(T_3[:, 2, 2])
    t3_12 = T_3[:, 0, 1]
    t3_13 = T_3[:, 0, 2]
    t3_23 = T_3[:, 1, 2]
    trace_T3 = t3_11 + t3_22 + t3_33

    with np.errstate(divide='ignore', invalid='ignore'):
        ps_branch1 = t3_11 + (np.abs(t3_12)**2 + np.abs(t3_13)**2) / t3_11
        pd_branch1 = trace_T3 - ps_branch1
        
        pd_branch2 = t3_22 + (np.abs(t3_12)**2 + np.abs(t3_23)**2) / t3_22
        ps_branch2 = trace_T3 - pd_branch2
        
        cond_surface = (t3_11 > t3_22)
        
        ps = np.where(cond_surface, ps_branch1, ps_branch2)
        pd = np.where(cond_surface, pd_branch1, pd_branch2)

    Ps = np.abs(ps).reshape(H, W)
    Pd = np.abs(pd).reshape(H, W)
    Pv = np.abs(P_V_flat).reshape(H, W)
    rgb_matrix = np.stack([Pd, Pv, Ps], axis=-1)
    
    return rgb_matrix

if __name__ == "__main__":
    # Synthetic illustration only; this does not reproduce a SAR scene.
    rng = np.random.default_rng(20261027)
    A = rng.normal(size=(4, 5, 3, 3)) + 1j * rng.normal(size=(4, 5, 3, 3))
    T_demo = A @ A.conj().swapaxes(-1, -2)
    span_demo = np.real(np.trace(T_demo, axis1=-2, axis2=-1))
    for name, method in [("SGSD-CMD", decompose1),
                         ("PGPA-CMD", decompose2),
                         ("DSP-CMD", decompose3)]:
        result = method(T_demo)
        error = np.max(np.abs(result.sum(axis=-1) - span_demo))
        print(f"{name}: shape={result.shape}, max absolute SPAN error={error:.3e}")
