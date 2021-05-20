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

from tf_agents.typing import types


class IntervalDriver(driver.Driver, ABC):
    def __init__(
            self,
            env: py_environment.PyEnvironment,
            policy: py_policy.PyPolicy,
            agents_per_recipe: types.Int,
            buffer_observer: Sequence[Callable[[trajectory.Trajectory], Any]],
            observers: Sequence[Callable[[trajectory.Trajectory], Any]],
            transition_observers: Optional[Sequence[Callable[[trajectory.Transition],
                                                             Any]]] = None,
            interval_number_of_recipes: Optional[types.Int] = None):

        super(IntervalDriver, self).__init__(env, policy, observers, transition_observers)
        # A compulsory number of agents per recipe that is pre-defined
        self._agents_per_recipe = agents_per_recipe
        # The number of interval
        self._interval_number_of_recipes = interval_number_of_recipes or np.inf
        # Compulsory buffer
        self._buffer_observer = buffer_observer
        # Temporary buffer for each recipe
        self._buffer_template = copy.copy(buffer_observer)  # Do I need to copy
        self._temp_buffer_reset()

    def _temp_buffer_reset(self):
        self._temp_buffer = self._buffer_template

    def run(self,
            time_step: ts.TimeStep,
            policy_state: types.NestedArray = ()
    ) -> Tuple[ts.TimeStep, types.NestedArray]:

        interval_number_of_recipes = 0
        num_agents_per_recipe = 0
        while interval_number_of_recipes < self._interval_number_of_recipes:
            action_step = self.policy.action(time_step, policy_state)
            next_time_step = self.env.step(action_step.action)

            traj = trajectory.from_transition(time_step, action_step, next_time_step)
            for observer in self._transition_observers:
                observer((time_step, action_step, next_time_step))
            for observer in self.observers:
                observer(traj)
            # Add trajectory to buffer
            self._temp_buffer(traj)

            # Count the agents per recipe
            num_agents_per_recipe += np.sum(traj.is_boundary())
            if traj.is_boundary():
                self.episodic_verb_reward()
            if num_agents_per_recipe >= self._agents_per_recipe:
                interval_number_of_recipes += 1
                num_agents_per_recipe = 0
                # Calculate the Convolution reward
                self.convolution_reward()
                # Pass the Update rewards buffer to the main buffer
                for traj in self._temp_buffer:
                    # Can use the add_batch as suggested from comment
                    self._buffer_observer(traj)
                # Reet the temporary buffer
                self._temp_buffer_reset()

            time_step = next_time_step
            policy_state = action_step.state
        # Calculate posterior rewards
        self.event_sequence_reward()

        return time_step, self._interval_number_of_recipes, policy_state

    @property
    def get_buffer_observer(self):
        return self._observers

    def episodic_verb_reward(self):
        """Penalise with a log function if away from the number of verbs divided by the NEs"""
        pass

    def convolution_reward(self):
        pass

    def event_sequence_reward(self):
        pass
