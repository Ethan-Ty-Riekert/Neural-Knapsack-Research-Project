"""wrapper_chain.py - find the innermost job/rule-selection wrapper under the training env's outer wrappers
(Monitor, ActionMasker, the critic-input wrapper). One definition for every caller that needs the wrapper's
attributes (max_jobs, get_action_mask, ...) -- the walk `env.env.env` used before fails once the critic-input wrapper
is present (--critic-arrivals)."""


def selection_wrapper(env):
    while not (hasattr(env, "max_jobs") and hasattr(env, "get_action_mask")):
        env = env.env
    return env
