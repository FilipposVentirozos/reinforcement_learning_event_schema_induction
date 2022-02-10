from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.policies import random_py_policy
from tf_agents.policies import random_tf_policy
from tf_agents.metrics import py_metrics
from tf_agents.metrics import tf_metrics
from tf_agents.drivers import py_driver
from tf_agents.drivers import dynamic_episode_driver

tf.compat.v1.enable_v2_behavior()

env = suite_gym.load('CartPole-v0')
policy = random_py_policy.RandomPyPolicy(time_step_spec=env.time_step_spec(),
                                         action_spec=env.action_spec())
replay_buffer = []
metric = py_metrics.AverageReturnMetric()
observers = [replay_buffer.append, metric]
driver = py_driver.PyDriver(
    env, policy, observers, max_steps=20, max_episodes=1)

initial_time_step = env.reset()
final_time_step, _ = driver.run(initial_time_step)

print('Replay Buffer:')
for traj in replay_buffer:
  print(traj)

print('Average Return: ', metric.result())