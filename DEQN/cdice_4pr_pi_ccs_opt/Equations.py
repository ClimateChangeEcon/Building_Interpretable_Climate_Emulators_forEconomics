import tensorflow as tf
import Definitions
import PolicyState
import Parameters
import State


def equations(state, policy_state):
    """ The dictionary of loss functions """
    # Expectation operator
    E_t = State.E_t_gen(state, policy_state)

    # ----------------------------------------------------------------------- #
    # Parameters
    # ----------------------------------------------------------------------- #
    Tstep = Parameters.Tstep
    Version = Parameters.Version
    # Economic and climate parameters
    delta, alpha, psi = Parameters.delta, Parameters.alpha, Parameters.psi
    b11 = Definitions.b11(state, policy_state)
    b12 = Definitions.b12(state, policy_state)
    b23 = Definitions.b23(state, policy_state)
    b21 = Definitions.b21(state, policy_state)
    b22 = Definitions.b22(state, policy_state)
    b32 = Definitions.b32(state, policy_state)
    b33 = Definitions.b33(state, policy_state)
    b41 = Definitions.b41(state, policy_state)
    b14 = Definitions.b14(state, policy_state)
    b44 = Definitions.b44(state, policy_state)
    c4 = Definitions.c4(state, policy_state)
    c1 = Definitions.c1(state, policy_state)
    c3 = Definitions.c3(state, policy_state)
    f2xco2, MATbase, t2xco2 = Parameters.f2xco2, Parameters.MATbase, Parameters.t2xco2
    theta2 = Parameters.theta2

    # Exogenously evolved parameters
    tfp = Definitions.tfp(state, policy_state)
    gr_tfp = Definitions.gr_tfp(state, policy_state)
    lab = Definitions.lab(state, policy_state)
    gr_lab = Definitions.gr_lab(state, policy_state)
    sigma = Definitions.sigma(state, policy_state)
    Eland = Definitions.Eland(state, policy_state)
    Fex = Definitions.Fex(state, policy_state)
    theta1 = Definitions.theta1(state, policy_state)
    beta_hat = Definitions.beta_hat(state, policy_state)

    # ----------------------------------------------------------------------- #
    # State variables
    # ----------------------------------------------------------------------- #
    # Retlieve the current state
    kx = State.kx(state)
    MATx, MUOx, MLOx, MLFx = State.MATx(state), State.MUOx(state), State.MLOx(state), State.MLFx(state)
    TATx, TOCx = State.TATx(state), State.TOCx(state)

    # States in period t+1
    MATplus = Definitions.MATplus(state, policy_state)
    MUOplus = Definitions.MUOplus(state, policy_state)
    MLOplus = Definitions.MLOplus(state, policy_state)
    MLFplus = Definitions.MLFplus(state, policy_state)
    TATplus = Definitions.TATplus(state, policy_state)
    TOCplus = Definitions.TOCplus(state, policy_state)

    # ----------------------------------------------------------------------- #
    # Pptimal policy functions in period t
    # ----------------------------------------------------------------------- #
    kplusy = PolicyState.kplusy(policy_state)
    lambd_haty = PolicyState.lambd_haty(policy_state)


    # ----------------------------------------------------------------------- #
    # Defined economic variables in period t
    # ----------------------------------------------------------------------- #
    con = Definitions.con(state, policy_state)
    Omega = Definitions.Omega(state, policy_state)
    Theta = Definitions.Theta(state, policy_state)
    Theta_prime = Definitions.Theta_prime(state, policy_state)

    # ----------------------------------------------------------------------- #
    # Loss functions
    # ----------------------------------------------------------------------- #
    loss_dict = {}
    # ----------------------------------------------------------------------- #
    # FOC wrt. kplus for dice 2016
    # ----------------------------------------------------------------------- #
    loss_dict['foc_kplus'] = tf.math.exp(Tstep*(gr_tfp + gr_lab)) * lambd_haty \
        - beta_hat * E_t(
            lambda s, ps:
            PolicyState.lambd_haty(ps) * (
                Tstep*(1 - Definitions.Theta(s, ps)- Definitions.Omega(s, ps))
                * alpha * kplusy**(alpha - 1)
                + (1 - delta)**Tstep)
        )
    # ----------------------------------------------------------------------- #
    # FOC wrt. lambd_haty (budget constraint) for dice 2016
    # ----------------------------------------------------------------------- #
    budget = Tstep*(1 - Theta - Omega) * kx**alpha - Tstep*con \
        + (1 - delta)**Tstep * kx - tf.math.exp(Tstep*(gr_tfp + gr_lab)) * kplusy
    loss_dict['foc_lambd'] = budget




    return loss_dict
