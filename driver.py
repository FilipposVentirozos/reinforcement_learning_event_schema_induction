# coding=utf-8

""" The driver is aimed to be instantiate for each patch of similar recipes to calculate the sequences,
     it calculates the posterior rewards that are stored in the buffer"""

from __future__ import absolute_import
from __future__ import division
# Using Type Annotations.
from __future__ import print_function

import copy
from abc import ABC
from typing import Any, Callable, Optional, Sequence, Tuple

import numpy as np
from tf_agents.drivers import driver
from tf_agents.environments import py_environment
from tf_agents.policies import py_policy
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.trajectories import policy_step

from tf_agents.typing import types
import tensorflow as tf
from sequence_tagger_env import EndOfDataSet
from collections import defaultdict


class IntervalDriver(driver.Driver, ABC):
    def __init__(
            self,
            env: py_environment.PyEnvironment,
            policy: py_policy.PyPolicy,
            # agents_per_recipe: types.Int,
            buffer_observer: Sequence[Callable[[trajectory.Trajectory], Any]],
            observers: Sequence[Callable[[trajectory.Trajectory], Any]] = None,
            transition_observers: Optional[Sequence[Callable[[trajectory.Transition],
                                                             Any]]] = None,
            interval_number_of_recipes: Optional[types.Int] = np.inf,  # Number of recipes to run the driver
            rec_count: Optional[types.Int] = 0  # Start from a latter recipe id
    ):

        super(IntervalDriver, self).__init__(env, policy, observers, transition_observers)

        # Deprecated, A compulsory number of agents per recipe that is pre-defined
        # self._agents_per_recipe = agents_per_recipe
        # The number of interval
        self._interval_number_of_recipes = interval_number_of_recipes
        # Compulsory buffer
        self._buffer_observer = buffer_observer
        # Temporary buffer for each recipe
        self._buffer_template = copy.copy(buffer_observer)  # Do I need to copy
        self._episode_buffer_reset()
        self._recipe_buffer_reset()  # Or have a list
        # Count the recipes to apply reward
        self.env.set_rec_count(rec_count)
        self.rec_count = rec_count

    def _episode_buffer_reset(self):
        self._episode_buffer = self._buffer_template

    def _recipe_buffer_reset(self):
        self._recipe_buffer = self._buffer_template

    # Initial driver.run will reset the environment and initialize the policy.
    # final_time_step, policy_state = driver.run()
    # From https://www.tensorflow.org/agents/tutorials/4_drivers_tutorial
    # def run(self,
    #         time_step: ts.TimeStep,
    #         policy_state: types.NestedArray = ()
    # ) -> Tuple[ts.TimeStep, types.NestedArray]:
    def run(self):
        """ The timestep and Policy are provided outside the scope of this functions

        :return:
        """
        # time_step = self.env.id  # This is the token ID in a recipe, gets zero after each recipe
        # Re-initialise the environment's variables
        self.env.set_rec_count(-1)
        self.env.seed = -1
        self.env.recipe_length = False
        time_step = self.env.reset()  # Check that
        policy_state = self.policy.get_initial_state(self.env.batch_size)
        interval_number_of_recipes = 0
        # num_agents_per_recipe = 0
        prev_recipe_id = self.rec_count
        while interval_number_of_recipes < self._interval_number_of_recipes:
            # Policy
            # [print(i.shape) for i in tf.nest.flatten(time_step)]
            action_step = self.policy.action(time_step, policy_state)
            next_time_step = self.env.step(action_step.action)
            self.env._episode_ended = False
            traj = from_transition(time_step, action_step, next_time_step)
            for observer in self._transition_observers:
                observer((time_step, action_step, next_time_step))
            for observer in self.observers:
                observer(traj)
            # Add trajectory to buffer
            # Trajectory(step_type=TensorSpec(shape=(), dtype=tf.int32, name='step_type'),
            #            observation=BoundedTensorSpec(shape=(768,), dtype=tf.float32, name='observation',
            #                                          minimum=array(-51.842564, dtype=float32),
            #                                          maximum=array(12.627596, dtype=float32)),
            #            action=BoundedTensorSpec(shape=(), dtype=tf.int32, name='action', minimum=array(0, dtype=int32),
            #                                     maximum=array(4, dtype=int32)), policy_info=(),
            #            next_step_type=TensorSpec(shape=(), dtype=tf.int32, name='step_type'),
            #            reward=TensorSpec(shape=(), dtype=tf.float32, name='reward'),
            #            discount=BoundedTensorSpec(shape=(), dtype=tf.float32, name='discount',
            #                                       minimum=array(0., dtype=float32), maximum=array(1., dtype=float32)))

            self._episode_buffer.add_batch(traj)

            # if traj.is_boundary():  # End of episode, evaluates the time_step not the next_time_step
            if next_time_step.is_last().numpy():  # End of episode for the current decision, the above will go to
                # the next step
                try:
                    self.env.reset()
                except EndOfDataSet:
                    break
                # Count recipes
                if prev_recipe_id != self.env.rec_count:
                    interval_number_of_recipes += 1
                    prev_recipe_id = self.env.rec_count

                # # Apply the NE reward estimate
                # # ToDo Get the number of actions
                # for traj, _ in self._episode_buffer.as_dataset(single_deterministic_pass=True):
                #     self._recipe_buffer.add_batch(traj)
                # # ToDo Apply the estimate on the Positive rewards
                # # ToDo call reset from environemnt
                # reward = self.episodic_verb_reward()
                # # ToDo Append to recipe buffer
                # # Re-Init episode buffer
                # self._recipe_buffer_reset()

                # # Is it the end of recipe?, should this be done upon greedy policy?
                # if self.rec_count < self.env.rec_count:
                #     self.rec_count = self.env.rec_count
                #     # Apply the Convolutional rewards, could be from list of episode buffers
                #     self._recipe_buffer.as_dataset(single_deterministic_pass=True)
                #     # Calculate convolution reward
                #     # Reset recipe buffer or liste
            # # Deprecated
            # # Count the agents per recipe
            # num_agents_per_recipe += np.sum(traj.is_boundary())
            # if traj.is_boundary():
            #     self.episodic_verb_reward()
            # if num_agents_per_recipe >= self._agents_per_recipe:
            #     interval_number_of_recipes += 1
            #     num_agents_per_recipe = 0
            #     # Calculate the Convolution reward
            #     self.convolution_reward()
            #     # Pass the Update rewards buffer to the main buffer
            #     for traj in self._temp_buffer:
            #         # Can use the add_batch as suggested from comment
            #         self._buffer_observer(traj)
            #     # Reet the temporary buffer
            #     self._temp_buffer_reset()

            time_step = next_time_step
            policy_state = action_step.state
        # Calculate posterior rewards
        # self.event_sequence_reward()

        return time_step, policy_state, self._interval_number_of_recipes, self.env.rec_count

    @property
    def get_buffer_observer(self):
        return self._observers

    def episodic_verb_reward(self, estimate_discount=0.5):
        """Penalise with a log function if away from the number of verbs divided by the NEs"""
        estimate = self.env.n_NE_estimate
        actual = None
        return np.exp(abs(estimate-actual)) * estimate_discount

    def convolution_reward(self):
        pass

    def event_sequence_reward(self):
        pass


class IntervalDriverEval(IntervalDriver):
    def __init__(
            self,
            env: py_environment.PyEnvironment,
            policy: py_policy.PyPolicy,
            # number_of_episodes: Optional[types.Int] = 10,  # Each recipe has many episodes
            rec_count: Optional[types.Int] = 0,
            interval_number_of_recipes: Optional[types.Int] = np.inf,
    ):
        super(IntervalDriverEval, self).__init__(env=env, policy=policy, buffer_observer=None, rec_count=rec_count,
                                                 interval_number_of_recipes=interval_number_of_recipes)
        # self.number_of_episodes = number_of_episodes

    def run(self):
        """ Get the Eval from the start for now

        :return:
        """
        time_step = self.env.reset()
        policy_state = self.policy.get_initial_state(self.env.batch_size)
        interval_number_of_recipes = 0
        # num_agents_per_recipe = 0
        num_episodes, total_return = 0, 0
        prev_recipe_id = None
        actions = defaultdict(lambda: 0)
        while interval_number_of_recipes <= self._interval_number_of_recipes:
            action_step = self.policy.action(time_step, policy_state)
            next_time_step = self.env.step(action_step.action)
            actions[action_step.action.numpy()[0]] += 1  # Count the actions
            self.env._episode_ended = False
            total_return += next_time_step.reward.numpy()
            if next_time_step.is_last().numpy():  # End of episode
                try:
                    self.env.reset()
                except EndOfDataSet:
                    break
                num_episodes += 1
                # Count recipes
                if prev_recipe_id != self.env.rec_count:
                    interval_number_of_recipes += 1
                    prev_recipe_id = self.env.rec_count

        return total_return / num_episodes, num_episodes, actions


def from_transition(time_step: ts.TimeStep,
                    action_step: policy_step.PolicyStep,
                    next_time_step: ts.TimeStep) -> trajectory.Trajectory:
    """Returns a `Trajectory` given transitions.

    `from_transition` is used by a driver to convert sequence of transitions into
    a `Trajectory` for efficient storage. Then an agent (e.g.
    `ppo_agent.PPOAgent`) converts it back to transitions by invoking
    `to_transition`.

    Note that this method does not add a time dimension to the Tensors in the
    resulting `Trajectory`. This means that if your transitions don't already
    include a time dimension, the `Trajectory` cannot be passed to
    `agent.train()`.

    Args:
      time_step: A `time_step.TimeStep` representing the first step in a
        transition.
      action_step: A `policy_step.PolicyStep` representing actions corresponding
        to observations from time_step.
      next_time_step: A `time_step.TimeStep` representing the second step in a
        transition.
    """
    return trajectory.Trajectory(
        step_type=tf.reshape(time_step.step_type, (1,)),
        observation=time_step.observation,
        action=tf.reshape(action_step.action, (1,)),
        policy_info=action_step.info,
        next_step_type=tf.reshape(next_time_step.step_type, (1,)),
        reward=tf.reshape(next_time_step.reward, (1,)),
        discount=tf.reshape(next_time_step.discount, (1,)))
