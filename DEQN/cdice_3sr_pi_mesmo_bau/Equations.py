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
    b11 = Definitions.b11(state, policy_state)
    b12 = Definitions.b12(state, policy_state)
    b23 = Definitions.b23(state, policy_state)
    b21 = Definitions.b21(state, policy_state)
    b22 = Definitions.b22(state, policy_state)
    b32 = Definitions.b32(state, policy_state)
    b33 = Definitions.b33(state, policy_state)
    c4 = Definitions.c4(state, policy_state)
    c1 = Definitions.c1(state, policy_state)
    c3 = Definitions.c3(state, policy_state)
    f2xco2, MATbase, t2xco2 = Parameters.f2xco2, Parameters.MATbase, Parameters.t2xco2

    # Exogenously evolved parameters
    gr_tfp = Definitions.gr_tfp(state, policy_state)
    gr_lab = Definitions.gr_lab(state, policy_state)
    beta_hat = Definitions.beta_hat(state, policy_state)

    # ----------------------------------------------------------------------- #
    # State variables
    # ----------------------------------------------------------------------- #
    # Retlieve the current state
    kx = State.kx(state)

    # States in period t+1
    MATplus = Definitions.MATplus(state, policy_state)

    # ----------------------------------------------------------------------- #
    # Pptimal policy functions in period t
    # ----------------------------------------------------------------------- #
    kplusy = PolicyState.kplusy(policy_state)

    lambd_haty = PolicyState.lambd_haty(policy_state)
    nuAT_haty = PolicyState.nuAT_haty(policy_state)
    nuUO_haty = PolicyState.nuUO_haty(policy_state)
    nuLO_haty = PolicyState.nuLO_haty(policy_state)
    etaAT_haty = PolicyState.etaAT_haty(policy_state)
    etaOC_haty = PolicyState.etaOC_haty(policy_state)

    # ----------------------------------------------------------------------- #
    # Defined economic variables in period t
    # ----------------------------------------------------------------------- #
    con = Definitions.con(state, policy_state)
    Omega = Definitions.Omega(state, policy_state)

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
                Tstep*(1 - Definitions.Omega(s, ps))
                * alpha * kplusy**(alpha - 1)
                + (1 - delta)**Tstep)
            + (-PolicyState.nuAT_haty(ps)) 
            * Tstep * Definitions.sigma(s, ps) * Definitions.tfp(s, ps)
            * Definitions.lab(s, ps)
            * alpha * kplusy**(alpha - 1)
        )
    # ----------------------------------------------------------------------- #
    # FOC wrt. lambd_haty (budget constraint) for dice 2016
    # ----------------------------------------------------------------------- #
    budget = Tstep*(1 - Omega) * kx**alpha - Tstep*con \
        + (1 - delta)**Tstep * kx - tf.math.exp(Tstep*(gr_tfp + gr_lab)) * kplusy
    loss_dict['foc_lambd'] = budget

    # ----------------------------------------------------------------------- #
    # FOC wrt. TATplus for dice 2016
    # ----------------------------------------------------------------------- #
    loss_dict['foc_TATplus'] = etaAT_haty - beta_hat * E_t(
        lambda s, ps:
        PolicyState.lambd_haty(ps)
        * (-Tstep * Definitions.Omega_prime(s, ps)) * kplusy**alpha
        + PolicyState.etaAT_haty(ps) * (1 - Tstep * c1 * c3 - Tstep * c1 * f2xco2 / t2xco2)
        + PolicyState.etaOC_haty(ps) * c4
    )

    # ----------------------------------------------------------------------- #
    # FOC wrt. MATplus
    # ----------------------------------------------------------------------- #
    loss_dict['foc_MATplus'] = (-nuAT_haty) - beta_hat * E_t(
        lambda s, ps:
        (-PolicyState.nuAT_haty(ps)) * b11
        + PolicyState.nuUO_haty(ps) * b12
        + PolicyState.etaAT_haty(ps) * c1 * f2xco2 * (1 / (
            tf.math.log(2.) * MATplus))
    )

    # ----------------------------------------------------------------------- #
    # FOC wrt. MUOplus
    # ----------------------------------------------------------------------- #
    loss_dict['foc_MUOplus'] = nuUO_haty - beta_hat * E_t(
        lambda s, ps:
        (-PolicyState.nuAT_haty(ps)) * b21
        + PolicyState.nuUO_haty(ps) * b22
        + PolicyState.nuLO_haty(ps) * b23
    )

    # ----------------------------------------------------------------------- #
    # FOC wrt. MLOplus
    # ----------------------------------------------------------------------- #
    loss_dict['foc_MLOplus'] = nuLO_haty - beta_hat * E_t(
        lambda s, ps:
        PolicyState.nuUO_haty(ps) * b32
        + PolicyState.nuLO_haty(ps) * b33
    )

    # ----------------------------------------------------------------------- #
    # FOC wrt. TOCplus
    # ----------------------------------------------------------------------- #
    loss_dict['foc_TOCplus'] = etaOC_haty - beta_hat * E_t(
        lambda s, ps:
        PolicyState.etaAT_haty(ps) * Tstep * c1 * c3
        + PolicyState.etaOC_haty(ps) * (1 - c4)
    )

    return loss_dict
