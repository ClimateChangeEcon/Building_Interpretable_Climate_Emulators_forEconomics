
################################################
# constants from config file

constants = {'Tstep': 1, # Years in one time step
            'Version': '2016', #Version of DICE 2007 or 2016
            ########## EXOGENOUS PROCESSES ###################################
            # Population
            'L0': 7403,  # World population in 2015 [millions]
            'Linfty': 11500,  # Asymptotic world population [millions]
            'deltaL': 0.0268,  # Rate of convergence to asymptotic world population
            # Total factor productivity in effective labor units
            'A0hat': 0.010295,  # Initial level of TFP
            'gA0hat': 0.0217,  # Initial growth rate of TFP
            'deltaA': 0.005,  # Rate of decline in growth of the productivity
            'adj_coef': 1.045, #Adjustment coefficient to make continuous TFP representation matching excatly a recursive one
            # Stochastic TFP
            'varrho': 0.035,  # Stochastic productivity process parameter
            'r': 0.775,  # Stochastic productivity process parameter
            'varsigma': 0.008,  # Stochastic productivity process parameter
            'tfp_scale': 1.,  # Scale the TFP shock
            # Carbon intensity
            'sigma0': 0.0000955592,  # Initial carbon instensity
            'deltaSigma': 0.001, # decline rate of decarbonization per year
            'gSigma0': -0.0152, #initial growth of carbon intensity per year
            # Mitigation
            'theta2': 2.6, # Mitigation cost parameter
            'pback': 0.55, # Cost of backstop 2010 thousand USD per tCo2 2015
            'gback': 0.005, # initial cost decline backstop cost per year
            'c2co2': 3.666, #transformation from c to co2
            # Land emissions
            'ELand0': 0.00070922, # emissions form land in 2005 (1000GtC per year)
            'deltaLand': 0.023, # decline rate of land amissions (per year)
            # Land emissions
            'fex0': 0.5, # 2000 forcing of nonCO2 GHG (Wm-2)
            'fex1': 1.0, #  2100 forcing of nonCO2 GHG (Wm-2)
            'Tyears': 85., # Number of years before 2100
            ########## ECONOMIC PARAMETERS ###################################
            # Numeric parameters
            'vartheta': 0.015,  # Purely numeric parameter to transform time periods
            # Utility function
            'rho': 0.015,  # Discount factor in continuous time
            'psi': 0.68965517,  # CRRA: intertemporal elasticity of substitution/risk aversion (Cai and Lontzek, 2019)
            # Production function
            'alpha': 0.3,  # Capital elasticity
            'delta': 0.1,  # Annual capital depreciation rate
            # Damage function
            'pi1': 0., # Climate damage factor coefficient
            'pi2': 0.00236,  # Climate damage factor coefficient
            'pow1': 1.,
            'pow2': 2.,
            #----------------Carbon mass---------------------------------
            'b12_': 0.0208,   # Rate of carbon diffusion from atmosphere to upper ocean
            'b23_': 0.0025,   # Rate of carbon diffusion from upper ocean to lower ocean
            'b14_': 0.0613,  # Rate of carbon diffusion from atmosphere to land biosphere
            'MATeq': 0.589, # Equilibrium mass of carbon in the Atmosphere
            'MUOeq': 1.078, # Equilibrium mass of carbon in the upper Ocean
            'MLOeq': 37.220, #Equilibrium mass of carbon in the lower ocean

            #-------------------Temperature----------------------
            'c1_': 0.137,    #Temperature coefficient
            'c3_': 0.73,
            'c4_': 0.00689,
            'f2xco2': 3.45,
            't2xco2': 3.1,
            'kappa': 1.2,
            'MATbase': 0.589,  # Preindustrial atmospheric carbon concentration
            # Initial state
            'k0': 2.926,  # K0/(A0L0)
            'MAT0': 0.850, #0.935, #0.851,  # [1000 GtC] 2004
            'MUO0': 1.236, #1.298,  # [1000 GtC] 2004
            'MLO0': 37.236, #37.244, # [1000 GtC] 2004
            'MLF0': 0.371, #0.377, # [1000 GtC] 2004
            'MLFeq0': 0.258, #Equilibrium mass of carbon in the lower ocean
            'TAT0': 1.1,  # [oC relative to the preindustrial]
            'TOC0': 0.27, # [oC relative to the preindustrial]
            'tau0': 0.  # Initial time period
            }


#################################################
### states

### endogenous states
kx_state = []
kx_state.append({'name':'kx', 'init':{'distribution':'truncated_normal','kwargs': {'mean':2.926,'stddev':0.}}})

MATx_state = []
MATx_state.append({'name':'MATx','init':{'distribution':'truncated_normal','kwargs': {'mean':0.850,'stddev':0.}}})

MUOx_state = []
MUOx_state.append({'name':'MUOx','init':{'distribution':'truncated_normal','kwargs': {'mean':1.236,'stddev':0.}}})

MLOx_state = []
MLOx_state.append({'name':'MLOx','init':{'distribution':'truncated_normal','kwargs': {'mean':37.236,'stddev':0.}}})

MLFx_state = []
MLFx_state.append({'name':'MLFx','init':{'distribution':'truncated_normal','kwargs': {'mean':0.371,'stddev':0.}}})

MLFeqx_state = []
MLFeqx_state.append({'name':'MLFeqx','init':{'distribution':'truncated_normal','kwargs': {'mean':0.258,'stddev':0.}}})


TATx_state = []
TATx_state.append({'name':'TATx','init':{'distribution':'truncated_normal','kwargs': {'mean':1.1,'stddev':0.}}})

TOCx_state = []
TOCx_state.append({'name':'TOCx','init':{'distribution':'truncated_normal','kwargs': {'mean':0.27,'stddev':0.}}})

# total endogenous state space
end_state = kx_state + MATx_state + MUOx_state + MLOx_state + MLFx_state + MLFeqx_state + TATx_state + TOCx_state


taux_state = []
taux_state.append({'name':'taux','init':{'distribution':'truncated_normal','kwargs': {'mean':0.,'stddev':0.}}})

# total exogenous state space
ex_state =  taux_state


###############################

### total state space
states = end_state + ex_state


#################################################
### Policies

kplusy_policy = []
kplusy_policy.append({'name':'kplusy','activation': 'tf.keras.activations.softplus'})

lambd_haty_policy = []
lambd_haty_policy.append({'name':'lambd_haty','activation': 'tf.keras.activations.softplus'})





### total number of policies
policies = kplusy_policy + lambd_haty_policy

##################################################
### definitions

tau2t_policy = []
tau2t_policy.append({'name':'tau2t'})

tau2tauplus_policy = []
tau2tauplus_policy.append({'name':'tau2tauplus'})

tfp_policy = []
tfp_policy.append({'name':'tfp'})

gr_tfp_policy = []
gr_tfp_policy.append({'name':'gr_tfp'})

lab_policy = []
lab_policy.append({'name':'lab'})

gr_lab_policy = []
gr_lab_policy.append({'name':'gr_lab'})

sigma_policy = []
sigma_policy.append({'name':'sigma'})

theta1_policy = []
theta1_policy.append({'name':'theta1'})

Eland_policy = []
Eland_policy.append({'name':'Eland'})

Fex_policy = []
Fex_policy.append({'name':'Fex'})

beta_hat_policy = []
beta_hat_policy.append({'name':'beta_hat'})

b11_policy = []
b11_policy.append({'name':'b11'})

b32_policy = []
b32_policy.append({'name':'b32'})

b21_policy = []
b21_policy.append({'name':'b21'})

b22_policy = []
b22_policy.append({'name':'b22'})

b12_policy = []
b12_policy.append({'name':'b12'})

b23_policy = []
b23_policy.append({'name':'b23'})

b33_policy = []
b33_policy.append({'name':'b33'})

b14_policy = []
b14_policy.append({'name':'b14'})

b41_policy = []
b41_policy.append({'name':'b41'})

b44_policy = []
b44_policy.append({'name':'b44'})


c1_policy = []
c1_policy.append({'name':'c1'})

c3_policy = []
c3_policy.append({'name':'c3'})

c4_policy = []
c4_policy.append({'name':'c4'})

con_policy = []
con_policy.append({'name':'con'})

Omega_policy = []
Omega_policy.append({'name':'Omega'})

Omega_prime_policy = []
Omega_prime_policy.append({'name':'Omega_prime'})

Theta_policy = []
Theta_policy.append({'name':'Theta'})

Theta_prime_policy = []
Theta_prime_policy.append({'name':'Theta_prime'})

ygross_policy = []
ygross_policy.append({'name':'ygross'})

ynet_policy = []
ynet_policy.append({'name':'ynet'})

inv_policy = []
inv_policy.append({'name':'inv'})

Eind_policy = []
Eind_policy.append({'name':'Eind'})

carbontax_policy = []
carbontax_policy.append({'name':'carbontax'})

Abatement_policy = []
Abatement_policy.append({'name':'Abatement'})

Dam_policy = []
Dam_policy.append({'name':'Dam'})

Emissions_policy = []
Emissions_policy.append({'name':'Emissions'})

MATplus_policy = []
MATplus_policy.append({'name':'MATplus'})

MUOplus_policy = []
MUOplus_policy.append({'name':'MUOplus'})

MLOplus_policy = []
MLOplus_policy.append({'name':'MLOplus'})

MLFplus_policy = []
MLFplus_policy.append({'name':'MLFplus'})

MLFeqplus_policy = []
MLFeqplus_policy.append({'name':'MLFeqplus'})

TATplus_policy = []
TATplus_policy.append({'name':'TATplus'})

TOCplus_policy = []
TOCplus_policy.append({'name':'TOCplus'})


### total number of definitions
definitions = tau2t_policy + tau2tauplus_policy + tfp_policy + \
                gr_tfp_policy + lab_policy + gr_lab_policy + \
                sigma_policy + theta1_policy + Eland_policy + \
                Fex_policy + beta_hat_policy + b11_policy + \
                b32_policy + b21_policy + b22_policy + \
                b12_policy + b23_policy + b33_policy + \
                b14_policy + b41_policy + b44_policy +\
                c1_policy + c3_policy + c4_policy + con_policy + Omega_policy + \
                Omega_prime_policy + Theta_policy + Theta_prime_policy +\
                ygross_policy + ynet_policy + inv_policy +\
                Eind_policy +  carbontax_policy + \
                Abatement_policy + Dam_policy + Emissions_policy +\
                MATplus_policy + MUOplus_policy + MLOplus_policy +\
                MLFplus_policy + MLFeqplus_policy +TATplus_policy + TOCplus_policy


