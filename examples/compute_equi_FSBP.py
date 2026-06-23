import json
from pathlib import Path
import numpy as np
import scipy.linalg

from src.operators import Operator

CACHE_FILE = Path(__file__).parent / "operator_cache.json"

def load_cache() -> dict:
    if CACHE_FILE.exists():
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return {}

def save_cache(cache: dict) -> None:
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=4)

def commutation_matrix(N: int) -> np.ndarray:
    """
    Generates a commutation matrix C such that C @ vec(A) = vec(A.T).
    Replicates MATLAB's index manipulation for C = speye(N*N); C = C(I,:).
    """
    I = np.arange(N * N).reshape((N, N), order='F')
    I_T = I.T
    idx = I_T.flatten(order='F')
    C = np.eye(N * N)[idx, :]
    return C

def compute_LSQF(x_L: float, x_R: float, span_G: callable, m_G: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Replicates compute_LSQF.m
    Iteratively finds the minimum number of equidistant points required
    to achieve positive, exact weights for the product space G.
    """
    L = len(m_G)
    N = max(L // 2, 2)
    
    exactness_error = 1.0
    w_min = -1.0
    tol_exactness = 1e-13
    
    while w_min < 1e-14 or exactness_error > tol_exactness:
        x = np.linspace(x_L, x_R, N)
        G = np.zeros((L, N))
        
        for n in range(N):
            G[:, n] = span_G(x[n])
            
        # lapack_driver='gelsd' forces the minimum-norm least squares solution (like MATLAB's lsqminnorm)
        w, _, _, _ = scipy.linalg.lstsq(G, m_G, lapack_driver='gelsd')
        
        w_min = np.min(w)
        exactness_error = np.linalg.norm(G @ w - m_G)**2 / L
        
        if w_min >= 1e-14 and exactness_error <= tol_exactness:
            break
            
        N += 1
        
    return x, w

def compute_FSBP(basis_F: callable, dx_basis_F: callable, x: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Replicates compute_FSBP.m
    Constructs the exact, minimal-norm differentiation matrix D.
    """
    N = len(x)
    K = len(basis_F(x[0]))
    
    P = np.diag(w)
    
    F = np.zeros((N, K))
    F_x = np.zeros((N, K))
    for n in range(N):
        F[n, :] = basis_F(x[n])
        F_x[n, :] = dx_basis_F(x[n])
        
    B = np.zeros((N, N))
    B[0, 0] = -1.0
    B[-1, -1] = 1.0
    
    R = P @ F_x - 0.5 * B @ F
    
    # Vectorize (Using Fortran order 'F' to match MATLAB's column-major flattening)
    A = np.kron(F.T, np.eye(N))
    r = R.flatten(order='F')
    
    C = commutation_matrix(N)
    
    A_ext = np.vstack([A, C + np.eye(N * N)])
    r_ext = np.concatenate([r, np.zeros(N * N)])
    
    # Solve for anti-symmetric part
    q_anti, _, _, _ = scipy.linalg.lstsq(A_ext, r_ext, lapack_driver='gelsd')
    Q_anti = q_anti.reshape((N, N), order='F')
    
    Q = Q_anti + 0.5 * B
    P_inv = np.diag(1.0 / w)
    D = P_inv @ Q
    
    return D, P, Q

def get_exact_equispaced_operator(h: float, pe: float, order: int = 2) -> Operator:
    """
    Wrapper function to compute/cache the exact equispaced FSBP operator.
    """
    #epsilon_0 = 0.0125
    #beta = h / epsilon_0
    beta = pe * h
    
    cache_key = f"h{h:g}_beta{beta:g}_equi_exact_p{order}"
    cache = load_cache()
    
    if cache_key in cache:
        data = cache[cache_key]
        return Operator(
            name=f"EXP_{cache_key}",
            basis=data["basis"],
            quad_basis=data["quad_basis"],
            op_type=data["op_type"],
            selector=data.get("selector", 0),
            interval=np.array(data["interval"]),
            nodes=np.array(data["nodes"]),
            D=np.array(data["D"]),
            H=np.array(data["H"]),
            tL=np.array(data["tL"]),
            tR=np.array(data["tR"])
        )

    print(f"  -> Cache miss. Generating EXACT equispaced operator for h={h:g} (beta={beta:g})...")

    if order == 2:
        # F = span{1, x, exp(beta*x)}
        basis_F = lambda x_val: np.array([1.0, x_val, np.exp(beta * x_val)])
        dx_basis_F = lambda x_val: np.array([0.0, 1.0, beta * np.exp(beta * x_val)])
        
        # G = (FF)' = span{1, x, exp(beta*x), x*exp(beta*x), exp(2*beta*x)}
        span_G = lambda x_val: np.array([
            1.0, 
            x_val, 
            np.exp(beta * x_val), 
            x_val * np.exp(beta * x_val), 
            np.exp(2.0 * beta * x_val)
        ])
        
        # Exact analytical integrals over [0, 1] to avoid quadrature error breaking the 1e-14 tolerance
        m_G = np.array([
            1.0,
            0.5,
            (np.exp(beta) - 1.0) / beta,
            ((beta - 1.0) * np.exp(beta) + 1.0) / (beta**2),
            (np.exp(2.0 * beta) - 1.0) / (2.0 * beta)
        ])
        
        basis_labels = ["1", "x", f"exp({beta:g}x)"]
        quad_labels = ["1", "x", f"exp({beta:g}x)", f"x*exp({beta:g}x)", f"exp(2*{beta:g}x)"]
        
    else:
        raise NotImplementedError("Exact equispaced formulation for p3 requires deriving the 7 moments analytically.")

    # 1. Compute Exact Positive Weights
    x, w = compute_LSQF(0.0, 1.0, span_G, m_G)
    
    # 2. Compute Exact Operator
    D, P, Q = compute_FSBP(basis_F, dx_basis_F, x, w)
    
    op_data = {
        "name": f"EXP_{cache_key}",
        "basis": basis_labels,
        "quad_basis": quad_labels, 
        "op_type": "closed",
        "selector": 0,
        "interval": [0.0, 1.0],
        "nodes": x.tolist(),
        "D": D.tolist(),
        "H": w.tolist(),
        "tL": np.eye(len(x))[0].tolist(),  # Left boundary vector
        "tR": np.eye(len(x))[-1].tolist()  # Right boundary vector
    }
    
    cache[cache_key] = op_data
    save_cache(cache)
    
    return Operator(
        name=op_data["name"],
        basis=op_data["basis"],
        quad_basis=op_data["quad_basis"],
        op_type=op_data["op_type"],
        selector=op_data["selector"],
        interval=np.array(op_data["interval"]),
        nodes=np.array(op_data["nodes"]),
        D=np.array(op_data["D"]),
        H=np.array(op_data["H"]),
        tL=np.array(op_data["tL"]),
        tR=np.array(op_data["tR"])
    )