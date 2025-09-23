import os
import sys
import logging
import pickle
import numpy as np
import scipy.optimize
import multiprocessing as mp
from textwrap import dedent

import lib_operations as operations
import lib_data_load as data_load
from lib_models import model_3sr, model_4pr

# ================================================================
# Banner blurb (concise, integrated) + input parameter docs
# ================================================================
BANNER = dedent("""
    ================================================================
    Building Interpretable Climate Emulators for Economics
    ================================================================
    This repository provides a Python-based framework supporting the
    workflow described by Eftekhari, Folini, Friedl, Kübler, Scheidegger,
    and Schenk (2024 https://www.arxiv.org/abs/2411.10768). It enables 
    Integrated Assessment Model (IAM) practitioners to construct 
    interpretable, efficient, and custom-tailored climate emulators 
    without requiring advanced climate science expertise.

    ----------------------------------------------------------------
    Penalty Parameter Grid Search
    ----------------------------------------------------------------
    As described in Eftekhari et al. (2024) [arXiv:2411.10768], this emulator
    fitting routine performs a grid search over penalty parameter triples:
        rho1_set = [0.0, 0.0001, 0.001, 0.01, 0.1]
        rho2_set = [0.0, 0.0001, 0.001, 0.01, 0.1]
        rho3_set = [0.0, 0.0001, 0.001, 0.01, 0.1]
    for a total of 125 (5×5×5) combinations. Each combination is evaluated
    for both the 3SR and 4PR carbon cycle models, resulting in a full grid
    of emulator fits.

    ----------------------------------------------------------------
    Model Framework
    ----------------------------------------------------------------
    Carbon Cycle Models (3SR, 4PR)
    - Linear multi-reservoir structure spanning Atmosphere (A),
      Ocean (O1 upper / O2 deep), and Land biosphere (L).
    - 3SR (serial): A → O1 → O2 (three sequential reservoirs).
    - 4PR (parallel): A splits to Land (L) and Ocean (O1 → O2) in parallel.
                
    ----------------------------------------------------------------
    Script Input Parameters
    ----------------------------------------------------------------
    1. conditions    : 'PI' or 'PD'
         PI = Preindustrial benchmark set
         PD = Present-day benchmark set
    2. T             : integer time horizon (years) for pulse decay fit
    3. benchmark_sel : one of the available benchmark model codes from
                       the selected benchmark set (see catalog below)

    Example:
        python fit_runner.py PD 250 MMM

    ----------------------------------------------------------------
    Benchmark Model Catalog
    ----------------------------------------------------------------
    Available benchmark models vary by 'conditions':
    - PI:  NCAR, BERN3D, BERN25D, CLIMBER2, DCESS, GENIE, LOVECLIM, MESMO,
           UVIC29, BERNSAR, MMM, MMMU, MMMD
    - PD:  NCAR, HADGEM2, MPIESM, BERN3DR, BERN3DE, BERN25D, CLIMBER2, DCESS,
           GENIE, LOVECLIM, MESMO, UVIC29, ACC2, BERNSAR, MAGICC6, TOTEM2,
           MMM, MMMU, MMMD
    """)

# ================================================================
# Benchmark catalogs (no descriptions)
# ================================================================
PI_BENCHMARKS = [
    'NCAR', 'BERN3D', 'BERN25D', 'CLIMBER2', 'DCESS',
    'GENIE', 'LOVECLIM', 'MESMO', 'UVIC29', 'BERNSAR',
    'MMM', 'MMMU', 'MMMD'
]

PD_BENCHMARKS = [
    'NCAR', 'HADGEM2', 'MPIESM', 'BERN3DR', 'BERN3DE', 'BERN25D',
    'CLIMBER2', 'DCESS', 'GENIE', 'LOVECLIM', 'MESMO', 'UVIC29',
    'ACC2', 'BERNSAR', 'MAGICC6', 'TOTEM2', 'MMM', 'MMMU', 'MMMD'
]

# ================================================================
# CLI helpers
# ================================================================

def get_benchmarks(conditions: str):
    if conditions == 'PI':
        return PI_BENCHMARKS
    if conditions == 'PD':
        return PD_BENCHMARKS
    raise ValueError("conditions must be 'PI' or 'PD'.")

# ================================================================
# Globals set at runtime for objective
# ================================================================
_data_pulse = None
_T_global = None
_metric_loss  = []
_metric_pen_1 = []
_metric_pen_2 = []
_metric_pen_3 = []

# ================================================================
# Core fitting logic (constraints + objective + driver)
# ================================================================

def constraints(z: np.array, model: callable) -> float:
    info       = model()
    p          = info['p']
    a_size     = len(info['a_bounds'])
    m_size     = len(info['m_bounds'])
    a          = z[0:a_size]
    m_eq       = z[a_size:a_size+m_size]

    time_scale_max = 10000

    [A, m_eq, x_vec]   = model(a, m_eq)
    [eig_val, eig_vec] = operations.eig(A)
    eig_val_trim       = eig_val[np.abs(eig_val) > 1e-12]
    time_scale         = 1.0 / (np.abs(eig_val_trim))

    c = 0.0
    # Reject complex parts
    c += np.sum(np.abs(eig_val.flatten().imag)) + np.sum(np.abs(eig_vec.flatten().imag))
    # Reject positive eigenvalues
    c += np.sum(np.abs(eig_val[eig_val > 0]))
    # Timescale bounds
    c += np.sum(time_scale[time_scale < .1]) + np.sum(time_scale[time_scale > time_scale_max])
    return float(c)

def obj_fun(z: np.array, model: callable, rho: list) -> float:
    global _data_pulse, _metric_loss, _metric_pen_1, _metric_pen_2, _metric_pen_3, _T_global

    info       = model()
    p          = info['p']
    a_size     = len(info['a_bounds'])
    m_size     = len(info['m_bounds'])
    a          = z[0:a_size]
    m_eq       = z[a_size:a_size+m_size]
    m_eq_base  = np.mean(np.array(info['m_bounds']), 1)

    [A, m_eq, x_vec] = model(a, m_eq)

    m0    = np.zeros(p)
    m0[0] = _data_pulse[0]

    # simulate pulse + loss
    m_sim       = operations.simulate_pulse(A=A, m0=m0, T=_T_global)
    m_benchmark = _data_pulse
    loss        = operations.l2_err(m_sim[0, :], m_benchmark) / _T_global

    # penalties
    pen_1 = -np.trace(A) / A.shape[0]  # q1
    pen_2 = np.linalg.norm((m_eq_base - m_eq) / m_eq_base) / A.shape[0]  # q2

    # q3: ocean vs land partition at ~20 years
    inx_ocean, inx_land = [], []
    for m_i_name in info['m_names']:
        if 'O' in m_i_name:
            inx_ocean.append(info['m_names'].index(m_i_name))
        elif 'L' in m_i_name:
            inx_land.append(info['m_names'].index(m_i_name))

    if len(inx_ocean) == 0 or len(inx_land) == 0:
        pen_3 = 0.0
    else:
        T20   = min(20, m_sim.shape[1] - 1)
        m_T   = m_sim[:, T20]
        m_to_ocean = np.sum(m_T[inx_ocean])
        m_to_land  = np.sum(m_T[inx_land])
        eta = 1.0
        pen_3 = np.linalg.norm(m_to_ocean / m_to_land - eta)

    _metric_loss.append(loss)
    _metric_pen_1.append(pen_1)
    _metric_pen_2.append(pen_2)
    _metric_pen_3.append(pen_3)

    return float(loss + pen_1*rho[0] + pen_2*rho[1] + pen_3*rho[2])

def fit_model(folder_name: str, model: callable, rho: list, seed: int = 1, verbose: bool = False) -> list:
    os.makedirs(folder_name, exist_ok=True)

    logging.root.handlers = []
    handlers = [logging.FileHandler(os.path.join(folder_name, 'opt.log'))]
    if verbose:
        handlers.append(logging.StreamHandler(sys.stdout))

    logging.basicConfig(
        format='%(asctime)s:%(levelname)s:%(message)s',
        level=logging.INFO,
        datefmt='%H:%M:%S',
        handlers=handlers
    )

    logging.info('GENERAL:seed:%s', seed)
    logging.info('GENERAL:rho:%s', rho)

    info   = model()
    a_size = len(info['a_bounds'])
    m_size = len(info['m_bounds'])

    bounds_z = info['a_bounds'] + info['m_bounds']
    z0_1     = np.mean(np.array(info['a_bounds']), 1)
    z0_2     = np.mean(np.array(info['m_bounds']), 1)

    logging.info('GENERAL:bounds_z:%s', bounds_z)

    res = scipy.optimize.differential_evolution(
        func        = obj_fun,
        bounds      = bounds_z,
        args        = (model, rho,),
        maxiter     = int(1e6),
        tol         = 1e-6,
        polish      = False,
        init        ='sobol',
        seed        = seed,
        x0          = np.concatenate((z0_1, z0_2)),
        constraints = (scipy.optimize.NonlinearConstraint(lambda z: constraints(z, model), 0, 0)),
    )

    for key in res:
        if isinstance(res[key], np.ndarray):
            logging.info('OPTIMIZER:%s:\n%s', key, res[key])
        else:
            logging.info('OPTIMIZER:%s:%s', key, res[key])

    z    = res.x
    a    = z[0:a_size]
    m_eq = z[a_size:a_size+m_size]

    [A, m_eq, x_vec]              = model(a, m_eq)
    [eig_val, eig_vec]            = operations.eig(A)
    [eig_val_left, eig_vec_left]  = operations.eig(A.T)

    result = {
        'A': A,
        'a': a,
        'm_eq': m_eq,
        'benchmark_pulse': _data_pulse,
        'success': bool(res.success),
        'time_scale': 1/np.abs(eig_val[:-1]) if len(eig_val) > 1 else np.array([]),
    }

    logging.info('RESULT:A:\n%s', A)
    logging.info('RESULT:m_eq:\n%s (sum=%s)', m_eq, np.sum(m_eq))
    logging.info('RESULT:x_vec:\n%s (sum=%s)', x_vec, np.sum(x_vec) if x_vec is not None else None)
    logging.info('RESULT:time_scale:\n%s', result['time_scale'])

    with open(os.path.join(folder_name, 'result.pkl'), 'wb') as f:
        pickle.dump(result, f)

    return [res.success, result]

def run(_conditions, _T, _benchmark_sel, rho, model, out_root):
    global _metric_loss, _metric_pen_1, _metric_pen_2, _metric_pen_3

    rho = [float(rho_i) for rho_i in rho]
    info = model()
    folder_name = os.path.join(out_root, f"{_benchmark_sel}-{'_'.join(map(str, rho))}", str(info['name']))
    print(f"  -> Running: {folder_name}")

    succ, result = fit_model(folder_name=folder_name, model=model, rho=rho)

    opt_data = {
        '_metric_loss':  _metric_loss,
        '_metric_pen_1': _metric_pen_1,
        '_metric_pen_2': _metric_pen_2,
        '_metric_pen_3': _metric_pen_3,
    }
    with open(os.path.join(folder_name, 'opt_data.pkl'), 'wb') as f:
        pickle.dump(opt_data, f)

    # reset trackers for next run
    _metric_loss.clear()
    _metric_pen_1.clear()
    _metric_pen_2.clear()
    _metric_pen_3.clear()

    print(f"     Done: {folder_name} | success={succ}")

def run_mask(args):
    (_conditions, _T, _benchmark_sel, rho, model, out_root, data_pulse) = args
    # Seed per-process globals used by obj_fun/constraints
    global _data_pulse, _T_global
    _data_pulse = data_pulse
    _T_global   = _T
    return run(_conditions, _T, _benchmark_sel, rho, model, out_root)


# ================================================================
# Entrypoint
# ================================================================

def main():
    global _data_pulse, _T_global

    # Print banner and usage only when no args given
    if len(sys.argv) == 1:
        print(BANNER)
        print("Usage: python solver.py <conditions: PI|PD> <T:int> <benchmark_sel>")
        print("Example: python solver.py PD 250 CLIMBER2")
        sys.exit(0)

    if len(sys.argv) != 4:
        print("Usage: python solver.py <conditions: PI|PD> <T:int> <benchmark_sel>")
        sys.exit(1)

    _conditions = sys.argv[1].strip().upper()
    try:
        _T = int(sys.argv[2])
    except ValueError:
        sys.exit("Error: T must be an integer.")
    _benchmark_sel = sys.argv[3].strip()

    # Validate selection
    try:
        available = get_benchmarks(_conditions)
    except ValueError as e:
        print(f"[FAIL] {e}")
        sys.exit(1)

    if _benchmark_sel not in available:
        print(f"[FAIL] Benchmark '{_benchmark_sel}' not available for {_conditions}.")
        print("       Options:", ", ".join(available))
        sys.exit(1)

    _T_global = _T

    # Selection summary
    print("----------------------------------------------------------------")
    print("Selection Summary")
    print("----------------------------------------------------------------")
    print(f"Conditions      : {_conditions}  (PI=Preindustrial, PD=Present Day)")
    print(f"Time Horizon T  : {_T}")
    print(f"Benchmark       : {_benchmark_sel}")
    print("----------------------------------------------------------------")

    # Load benchmark pulse
    print("[LOAD] Fetching benchmark pulse...")
    _data_pulse, _ = data_load.pulse_fraction(test_type=_benchmark_sel, conditions=_conditions, T=_T)
    if _data_pulse is None or len(_data_pulse) == 0:
        sys.exit("[FAIL] Empty benchmark pulse returned.")

    # Rho grid + models
    rho1_set = np.array([0.0, 0.0001, 0.001, 0.01, 0.1])
    rho2_set = np.array([0.0, 0.0001, 0.001, 0.01, 0.1])
    rho3_set = np.array([0.0, 0.0001, 0.001, 0.01, 0.1])
    rho_set  = [[r1, r2, r3] for r3 in rho3_set for r2 in rho2_set for r1 in rho1_set]

    model_set = [model_4pr, model_3sr]


    out_root = os.path.join("result_NEW", _conditions,f"T-{_T}")
    os.makedirs(out_root, exist_ok=True)

    tasks = [(_conditions, _T, _benchmark_sel, rho, m, out_root, _data_pulse)
             for rho in rho_set for m in model_set]


    print(f"[PLAN] Total tasks: {len(tasks)} (rho grid={len(rho_set)} x models={len(model_set)})")
    print(f"[OUT ] Root directory: {out_root}")
    print("----------------------------------------------------------------")

    # Pool size heuristic: half the cores, but not more than #tasks
    pool_size = min(max(1, mp.cpu_count() // 2), max(1, len(tasks)))
    print(f"[EXEC] Launching pool with {pool_size} workers...")
    with mp.Pool(pool_size) as pool:
        pool.map(run_mask, tasks)

    print("\n[END] Completed all fits.")

if __name__ == "__main__":
    main()
