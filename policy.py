# from tf_agents.policies import gaussian_policy  # Is intended for integer range Actions, not suitable for binomial
from tf_agents.policies import random_py_policy

class Policy:

    def __init__(self, action_spec, *, policy="random"):
        if policy == "random":
            # Random Policy, choose either 0 or 1
            self.policy_random = random_py_policy.RandomPyPolicy(time_step_spec=None,
                action_spec=action_spec)

        # Change to stochastic Policy

    def get_action(self, step):
        return self.policy_random.action(step)
