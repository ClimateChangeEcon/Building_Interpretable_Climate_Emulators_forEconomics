import tensorflow as tf
import State
import PolicyState
import Definitions

# There is no random shock, i.e., shock_step_random and shock_step_spec_shock =
# 0 over time.

# Probability of a dummy shock
shock_probs = tf.constant([1.0])  # Dummy probability

def augment_state(state):
    _state = state
    return _state


def total_step_random(prev_state, policy_state):
    """ State dependant random shock to evaluate the expectation operator """
    _ar = AR_step(prev_state)
    _shock = shock_step_random(prev_state)
    _policy = policy_step(prev_state, policy_state)

    _total_random = _ar + _shock + _policy

    return augment_state(_total_random)


def total_step_spec_shock(prev_state, policy_state, shock_index):
    """ State specific shock to run one episode """
    _ar = AR_step(prev_state)
    _shock = shock_step_spec_shock(prev_state, shock_index)
    _policy = policy_step(prev_state, policy_state)

    _total_spec = _ar + _shock + _policy

    return augment_state(_total_spec)


def AR_step(prev_state):
    """ AR(1) shock on zetax and chix """
    _ar_step = tf.zeros_like(prev_state)  # Initialization
    return _ar_step


def shock_step_random(prev_state):
    """ TFP shock zeta and chi """
    _shock_step = tf.zeros_like(prev_state)  # Initialization

    return _shock_step


def shock_step_spec_shock(prev_state, shock_index):
    """ TFP shock zeta and chi """
    _shock_step = tf.zeros_like(prev_state)  # Initialization

    return _shock_step


def policy_step(prev_state, policy_state):
    """ State variables are updated by the optimal policy (capital stock) or
    the laws of motion for carbon masses and temperatures """
    _policy_step = tf.zeros_like(prev_state)  # Initialization

    # Update state variables if needed
    _policy_step = State.update(
        _policy_step, 'kx', PolicyState.kplusy(policy_state))
    _policy_step = State.update(
        _policy_step, 'MATx', Definitions.MATplus(prev_state, policy_state))
    _policy_step = State.update(
        _policy_step, 'MUOx', Definitions.MUOplus(prev_state, policy_state))
    _policy_step = State.update(
        _policy_step, 'MLOx', Definitions.MLOplus(prev_state, policy_state))
    _policy_step = State.update(
        _policy_step, 'MLFx', Definitions.MLFplus(prev_state, policy_state))
    _policy_step = State.update(
        _policy_step, 'MLFeqx', Definitions.MLFeqplus(prev_state, policy_state))
    _policy_step = State.update(
        _policy_step, 'TATx', Definitions.TATplus(prev_state, policy_state))
    _policy_step = State.update(
        _policy_step, 'TOCx', Definitions.TOCplus(prev_state, policy_state))
    _policy_step = State.update(
        _policy_step, 'taux', Definitions.tau2tauplus(prev_state, policy_state)
    )

    return _policy_step
