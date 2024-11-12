import numpy as np
import pickle
from prettytable import PrettyTable
import lib_data_load as data_load
from lib_models import model_3sr, model_4pr

def l2_err(x, x_true):
    temp = (x - x_true) 
    return np.linalg.norm(temp, ord=2) 

def l1_err(x, x_true):
    temp = (x - x_true) 
    return np.linalg.norm(temp, ord=1) 

def linf_err(x, x_true):
    temp = (x - x_true)
    return np.max(np.abs(temp))

def eig(A: np.array) -> list:
    """eigen value decomposion

    Args:
        A (np.array): Matrix

    Returns:
        list: [eig_val,eig_vec] 
        normalized (unit “length”) eigenvectors, such that the column eig_vec[:,i] 
        is the eigenvector corresponding to the eigenvalue eig_val[i]. 
        Sorted biggest to absval (largest to smallest)
    """
    [eig_val, eig_vec] = np.linalg.eig(A)

    idx = np.abs(eig_val).argsort()[::-1]
    eig_val = eig_val[idx]
    eig_vec = eig_vec[:, idx]

    return [eig_val, eig_vec]

def simulate_pulse(A: np.array, m0: np.array, T: int) -> np.array:
    """Simulate the ODE for the given operator starting from state x0 for T steps

    Args:
        A (np.array): Matrix operator
        T (int): Number of time steps
        x0 (np.array): Initial state

    Returns:
        np.array: Simulated path from x0 for T steps 
    """

    [eig_val, eig_vec] = np.linalg.eig(A)

    # idx = np.abs(eig_val).argsort()[::-1]
    # eig_val = eig_val[idx]
    # eig_vec = eig_vec[:,idx]

    # Initial variables
    p = len(m0)
    m = np.zeros(shape=(p, T))

    # solve system Vc=x0 where V is eigen vectors
    c = np.linalg.solve(eig_vec, m0)

    # E[i,j] = exp(\lambda_i x t_j)
    E = np.exp(np.outer(eig_val, np.array(range(0, T))))

    for i in range(0, p):
        m += c[i] * np.outer(eig_vec[:, i], E[i, :])

    return m

def simulate_dynamic_ignore(A_set, m0, T, 
    e=None,
    c1=0.137, 
    c3=0.73, 
    c4=0.00689, 
    F2XCO2=3.45,
    T2XCO2=3.25, 
    TAT0=0,#1.1,
    TOC0=0,#0.27,
    delta_t=1,
    m_A_eq=589
    ) -> list:

    p = A_set[0].shape[0]
    assert p == len(m0)

    # Initial variables
    if e is None:
        e = np.zeros(shape=(p,T))

    assert T == e.shape[1]

    m = np.empty(shape=(p, T))
    temprature_oc = np.zeros(T)
    temprature_at = np.zeros(T)

    # Using m_new = Am_new + m 
    # (I-A)m_new = m 
    #  m_new = inv(I-A) @ m
    # than we add new emission on, i.e., m_new_t += e_t
    # A is now time depenatnt ... we need to do this on every iteration
    I = np.eye(p)

    temprature_at[0] = TAT0
    temprature_oc[0] = TOC0

    m[:, 0] = m0
    m[:, 0] += e[:,0]
    for t in range(1, T):

        C = np.linalg.inv(np.eye(p) - A_set[t])

        m[:, t] = C @ m[:, t - 1] + e[:,t]

        F_t = (1+0.3) * F2XCO2 * np.log(m[0, t-1] / m_A_eq) / np.log(2) 

        delta_temp       = temprature_at[t - 1] - temprature_oc[t - 1]
        temprature_at[t] = temprature_at[t - 1] + delta_t * c1 * (F_t - F2XCO2/T2XCO2*temprature_at[t-1] - c3*(delta_temp))
        temprature_oc[t] = temprature_oc[t - 1] + delta_t * c4 * (delta_temp)

    return [m, temprature_at, temprature_oc]

# see https://www.umr-cnrm.fr/IMG/pdf/2box_i.pdf
def simulate_ignore(A, m0, T, 
    e=None,
    c1=0.137, 
    c3=0.73, 
    c4=0.00689, 
    F2XCO2=3.45,
    T2XCO2=3.25, 
    TAT0=0,#1.1,
    TOC0=0,#0.27,
    m_A_eq = 589
    ) -> list:

    p = A.shape[0]
    assert p == len(m0)

    # Initial variables
    if e is None:
        e = np.zeros(shape=(p,T))

    assert T == e.shape[1]

    m = np.empty(shape=(p, T))
    temprature_oc = np.zeros(T)
    temprature_at = np.zeros(T)

    # Using m_new = Am_new + m 
    # (I-A)m_new = m 
    #  m_new = inv(I-A) @ m
    # than we add new emission on, i.e., m_new_t += e_t
    # A is now time depenatnt ... we need to do this on every iteration
    I = np.eye(p)

    C = np.linalg.inv(np.eye(p) - A)

    temprature_at[0] = TAT0
    temprature_oc[0] = TOC0

    delta_t=1

    m[:, 0] = m0
    m[:, 0] += e[:,0]
    for t in range(1, T):

        m[:, t] = C @ m[:, t - 1] + e[:,t]

        F_t = (1+0.3) * F2XCO2 * np.log(m[0, t-1] / m_A_eq) / np.log(2) 

        delta_temp       = temprature_at[t - 1] - temprature_oc[t - 1]
        temprature_at[t] = temprature_at[t - 1] + delta_t * c1 * (F_t - F2XCO2/T2XCO2*temprature_at[t-1] - c3*(delta_temp))
        temprature_oc[t] = temprature_oc[t - 1] + delta_t * c4 * (delta_temp)

    return [m, temprature_at, temprature_oc]

# see https://www.umr-cnrm.fr/IMG/pdf/2box_i.pdf
def simulate_new(A, m0, T, 
    e=None,
    C=7.3, 
    C0=106, 
    l=1.13, 
    gamma= 0.73,
    FX_ratio = 1.2,
    F2XCO2 = 3.45,
    TAT0=0,#1.1,
    TOC0=0,#0.27,
    m_A_eq = 589
    ) -> list:

    p = A.shape[0]
    assert p == len(m0)

    # Initial variables
    if e is None:
        e = np.zeros(shape=(p,T))

    assert T == e.shape[1]

    if np.isscalar(FX_ratio):
        FX_ratio = np.ones(T)*FX_ratio
    else:
        assert len(FX_ratio)==T

    m = np.empty(shape=(p, T))
    temprature_oc = np.zeros(T)
    temprature_at = np.zeros(T)

    G = np.zeros(shape=(2,2))
    G[0,0] = -(l+gamma)/C
    G[0,1] = gamma/C
    G[1,0] = gamma/C0
    G[1,1] = -gamma/C0

    # Using m_new = Am_new + m 
    # (I-A)m_new = m 
    #  m_new = inv(I-A) @ m
    # than we add new emission on, i.e., m_new_t += e_t
    # A is now time depenatnt ... we need to do this on every iteration
    I = np.eye(p)

    B = np.linalg.inv(np.eye(p) - A)

    temprature = np.zeros(shape=(2,T))
    temprature[0,0] = TAT0
    temprature[1,0] = TOC0

    b = np.zeros(2).T

    m[:, 0] = m0
    m[:, 0] += e[:,0]
    for t in range(1, T):

        m[:, t] = B @ m[:, t - 1] + e[:,t]
        F_t =  FX_ratio[t-1] * F2XCO2 * np.log(m[0, t-1] / m_A_eq) / np.log(2) 
        b[0] = F_t / C
        temprature[:,t]= temprature[:,t-1] +  G@temprature[:,t-1] + b.T


    return [m, temprature[0,:], temprature[1,:]]

# see https://www.umr-cnrm.fr/IMG/pdf/2box_i.pdf
def simulate_dynamic_new(A_set, m0, T, 
    e = None,
    C = 7.3, 
    C0 = 106, 
    l = 1.13, 
    gamma= 0.73,
    FX_ratio = 1.2,
    F2XCO2 = 3.45,
    TAT0 = 0,#1.1,
    TOC0 = 0,#0.27,
    m_A_eq = 589
    ) -> list:

    p = A_set[0].shape[0]
    assert p == len(m0)

    # Initial variables
    if e is None:
        e = np.zeros(shape=(p,T))

    assert T == e.shape[1]

    if np.isscalar(FX_ratio):
        FX_ratio = np.ones(T)*FX_ratio
    else:
        assert len(FX_ratio)==T

    m = np.empty(shape=(p, T))
    temprature_oc = np.zeros(T)
    temprature_at = np.zeros(T)

    G = np.zeros(shape=(2,2))
    G[0,0] = -(l+gamma)/C
    G[0,1] = gamma/C
    G[1,0] = gamma/C0
    G[1,1] = -gamma/C0

    # Using m_new = Am_new + m 
    # (I-A)m_new = m 
    #  m_new = inv(I-A) @ m
    # than we add new emission on, i.e., m_new_t += e_t
    # A is now time depenatnt ... we need to do this on every iteration
    I = np.eye(p)

    temprature = np.zeros(shape=(2,T))
    temprature[0,0] = TAT0
    temprature[1,0] = TOC0

    b = np.zeros(2).T 

    m[:, 0] = m0
    m[:, 0] += e[:,0]
    for t in range(1, T):

        B = np.linalg.inv(I - A_set[t])
        m[:, t] = B @ m[:, t - 1] + e[:,t]
        F_t = FX_ratio[t-1] * F2XCO2 * np.log(m[0, t-1] / m_A_eq) / np.log(2) 
        b[0] = F_t / C
        temprature[:,t]= temprature[:,t-1] + G@temprature[:,t-1] + b

    return [m, temprature[0,:], temprature[1,:]]

def load_results(root, test_name, conditions, model_set=[model_3sr, model_4pr],T_sim_set=[500]):
    
    results = {}
    benchmark_name = test_name.split('-')[0]
    rho = test_name.split('-')[1].split('_')
    rho = np.array([float(rho_i) for rho_i in rho])

    for model_inx, model in enumerate(model_set):

        info       = model()
        a_size     = len(info['a_bounds'])
        m_size     = len(info['m_bounds'])
        model_name = info['name']

        folder = root + test_name + '/' + model_name + '/result.pkl'
        with open(folder, 'rb') as f:
            data = pickle.load(f)

        if data['success'] == False:
            print(folder,'::Failed')

        results[model_name]                   = {}
        results[model_name]['rho']            = rho
        results[model_name]['benchmark_name'] = benchmark_name
        results[model_name]['model']          = model
       
        ############################################################
        # definition of the model
        ############################################################
        A    = data['A']
        m_eq = data['m_eq']
        a    = data['a']

        results[model_name]['A']    = A
        results[model_name]['m_eq'] = np.round(m_eq,0)
        results[model_name]['a']    = np.round(a,7)
        ############################################################
        # Eigen value related
        ############################################################
        [eig_val, eig_vec] = eig(A)
        eig_val_trim = eig_val[np.abs(eig_val) > 1e-12]
        time_scale = 1.0 / np.abs(eig_val_trim)
        
        results[model_name]['time_scale']   = np.round(time_scale,1)

        ############################################################
        # error metrics
        #############################################################
        m0 = np.zeros(A.shape[0])
        m0[0] = data['benchmark_pulse'][0]

        T_max= max(T_sim_set)

        [m_benchmark,_] = data_load.pulse_fraction(test_type=benchmark_name, T=T_max,conditions=conditions)
        m_benchmark    *= m0[0]
        m_sim           = simulate_pulse(A=A, m0=m0, T=T_max)

        results[model_name]['m_sim'] = m_sim
        results[model_name]['benchmark_pulse'] = m_benchmark

        for T in T_sim_set:
            results[model_name]['err_l1_'+str(int(T))]      = l1_err(m_sim[0, 0:T], m_benchmark[0:T])   / (T)
            results[model_name]['err_l2_'+str(int(T))]      = l2_err(m_sim[0, 0:T], m_benchmark[0:T])   / (T)
            results[model_name]['err_linf_'+str(int(T))]    = linf_err(m_sim[0, 0:T], m_benchmark[0:T]) 
            results[model_name]['err_sum_abs_'+str(int(T))] =  np.abs(np.sum(m_sim[0, 0:T] - m_benchmark[0:T]))

        ############################################################
        # compute difference betwee carbon absorved by oceans and lands at t=50
        ############################################################
        
        # identify ocean and land reservoirs
        inx_ocean = []
        inx_land  = []

        for m_i_name in info['m_names']:
            if 'O_' in m_i_name:
                inx_ocean.append(info['m_names'].index(m_i_name))
            elif 'L_' in m_i_name:
                inx_land.append(info['m_names'].index(m_i_name))
        
        # we done have oceans/land ignore not relevant
        if len(inx_ocean) == 0 or len(inx_land) == 0:
            diff_o_l = 0
        else:
            T = 20

            m0[0] = data['benchmark_pulse'][0]
            m_sim = simulate_pulse(A=A, m0=m0, T=T)
            m_T  = m_sim[:,-1]

            m_to_ocean = np.sum(m_T[inx_ocean])
            m_to_land  = np.sum(m_T[inx_land])

            diff_o_l = m_to_ocean/m_to_land
        results[model_name]['diff_o_l'] = diff_o_l 

    return results

def load_opt_data(root, test_name, model_set=[model_3sr, model_4pr]):
    
    opt_data = {}

    for model_inx, model in enumerate(model_set):
        info       = model()
        a_size     = len(info['a_bounds'])
        model_name = info['name']
        folder     = root + test_name + '/' + model_name + '/opt_data.pkl'

        with open(folder, 'rb') as f:
            data = pickle.load(f)

        opt_data[model_name] = {}
        opt_data[model_name] = data

    return opt_data

def tabulate(results, title='Table', vars=['a', 'm_eq', 'time_scale', 'diff_o_l']):

    table = PrettyTable()
    table.field_names = ['model', 'benchmark_name', 'rho'] + vars
    table.float_format = "0.5"
    table.title = title

    data = {}
    data['key'] = table.field_names
    data['val'] = []

    for model_inx, model_name in enumerate(results.keys()):
        row = []
        row += [model_name]
        row += [results[model_name]['benchmark_name']]
        row += [results[model_name]['rho']]
        temp = []
        for u in table.field_names[3:]:
            temp += [results[model_name][u]]
        row += temp
        table.add_row(row)
        data['val'].append(temp)

    print(table)