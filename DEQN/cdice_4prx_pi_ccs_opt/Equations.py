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
    # Economic and climate parameters
    delta, alpha, psi = Parameters.delta, Parameters.alpha, Parameters.psi

    # Exogenously evolved parameters
    gr_tfp = Definitions.gr_tfp(state, policy_state)
    gr_lab = Definitions.gr_lab(state, policy_state)
    beta_hat = Definitions.beta_hat(state, policy_state)

    # ----------------------------------------------------------------------- #
    # State variables
    # ----------------------------------------------------------------------- #
    # Retlieve the current state
    kx = State.kx(state)


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
