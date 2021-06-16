from abc import ABC

import numpy as np
from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.specs.array_spec import BoundedArraySpec
from tf_agents.trajectories import time_step as ts


class EndOfDataSet(Exception):
    """ Raise when the end of Dataset

    """
    pass


class EndOfEpisode(Exception):
    """ Raise when end of episode, no more tokens to visite, start with a new word token seed

    """
    pass


class SequenceTaggerEnv(PyEnvironment, ABC):
    """
    Custom `PyEnvironment` environment for imbalanced classification.
    Based on https://www.tensorflow.org/agents/tutorials/2_environments_tutorial
    """

    def __init__(self, X_train: np.ndarray, y_train: np.ndarray):  # , per_rec: int
        """Initialization of environment with X_train and y_train.

        :param X_train: Features shaped: [samples, ..., ]
        :type  X_train: np.ndarray
        :param y_train: Labels shaped: [samples]
        :type  y_train: np.ndarray
        :param imb_ratio: Imbalance ratio of the data
        :type  imb_ratio: float

        :returns: None
        :rtype: NoneType

        """
        # Action is 0, 1  to whether take into account an NE or not
        # Updated to O (0), Verb (1), Device (2), Ingr (3)
        # -1 would mean that is non existent, so it's omitted below
        self._action_spec = BoundedArraySpec(shape=(), dtype=y_train.dtype, minimum=0, maximum=3, name="action")
        # Observation is the embedding of the NE, which is between 0 and 1
        # self._observation_spec = ArraySpec(shape=X_train.shape[0, 0, :], dtype=X_train.dtype, name="observation")
        # self._observation_spec = BoundedArraySpec(shape=X_train[0, 0, :].shape, minimum=0, maximum=1,
        #                                           dtype=X_train.dtype, name="observation")
        self._observation_spec = BoundedArraySpec(shape=X_train[0, 0, :].shape, minimum=X_train.min(), maximum=X_train.max(),
                                                  dtype=X_train.dtype, name="observation")
        self._episode_ended = False

        self.X_train = X_train
        self.y_train = y_train

        # Count the recipes for env reset and also to establish posterior sequence reward
        self.rec_count = -1  # It will increment to 0 index
        # self.seed_buffer = list()
        self.token_trajectory = list()
        # Sample an id to start with

        self.seed = -1  # It will increment to 0 index
        self.set_seed_sequential()
        self.episode_step = 0  # Episode step, resets every episode
        self.recipe_length = np.inf

        # Inside recipe counters
        self._maxed_increment = self.maxed_increment()
        self.balance = 0
        self.increment = 0
        # Each episode is a recipe (1 of 50)
        self._reset()
        self._state = self.X_train[self.rec_count, self.seed, :]

        # # The below variables are used for random sampling, the previous version
        # self.search_space = 10  # The number of candidate states
        # self.per_rec = per_rec  # The amount of agents to run in a single recipe, but with different starting points
        # self.per_rec_counter = 0  # Counter for the above
        # self.used_NEs = list()

        # # Replay buffer, to store variables and train accordingly
        # batch_size = 32
        # replay_buffer_capacity = 1000 * batch_size
        # buffer_unit = (tf.TensorSpec([1], tf.bool, 'action'),  # Binary is 0 or 1
        #                (tf.TensorSpec([5], tf.float32, 'lidar'),
        #                 # ToDo set the NEs values instead, add index info as well, reward?
        #                 tf.TensorSpec([3, 2], tf.float32, 'camera')))
        # py_replay_buffer = py_uniform_replay_buffer.PyUniformReplayBuffer(
        #     capacity=replay_buffer_capacity,
        #     data_spec=tensor_spec.to_nest_array_spec(buffer_unit))

    def action_spec(self):
        """
        Definition of the discrete actionspace.
        1 for the positive/minority class, 0 for the negative/majority class.
        """
        return self._action_spec

    def observation_spec(self):
        """Definition of the continuous statespace e.g. the observations in typical RL environments."""
        return self._observation_spec

    @DeprecationWarning
    def set_seed_random_NE(self):
        """The seed should be a random NE, we do sampling without replacement since we do not want to have
         the same NE seed and we want to prioritise NE samples that have not been tagged yet."""

        NE_samples = np.nonzero(self.y_train[self.per_rec_counter, :])[0]
        # Get the seed IDs positions
        seed_buffer_ids = [np.argwhere(NE_samples == i) for i in self.seed_buffer]
        NE_samples = np.delete(NE_samples, seed_buffer_ids)
        try:
            # Prioritse a sample NE that has not been tagged yet
            used_NEs_buffer_ids = [np.argwhere(NE_samples == i) for i in self.used_NEs]
            NE_used_samples = np.delete(NE_samples, used_NEs_buffer_ids)
            self.seed = np.random.choice(NE_used_samples)
        except ValueError:
            self.seed = np.random.choice(NE_samples)
        self.seed_buffer.append(self.seed)

    def set_seed_sequential(self):
        """ Set the seed one after one, no matter is an NE or not

        :return:
        """
        try:
            self.seed += 1
            self.id = self.seed
            self._state = self.X_train[self.rec_count, self.id, :]
            # Re-initialise inside recipe counters
            self._maxed_increment = self.maxed_increment()  # Get the farthest away token for the end of episode
            self.balance = 0
            self.increment = 0
            if np.sum(self._state) == 0:  # Have exhausted the tokens on the current recipe, move to next recipe,
                # 0 would denote an empty cell
                self.next_recipe()
                return
            ts.restart(self._state)
        except IndexError:
            try:
                assert self.X_train[self.rec_count, 0, :]
                raise EndOfDataSet
            except IndexError:  # If the error is not caused by end of data, then the index means we should go to the
                # next recipe
                self.next_recipe()

    def set_seed_random(self):
        """ Set the seed random, without replacement from a recipe, no matter is an NE or not

        :return:
        """
        pass

    @DeprecationWarning
    def set_id_via_normal_distr(self, sigma=3):
        while True:
            new_id = round(np.random.normal(self.seed, sigma, 1)[0])
            if new_id in self.seed_buffer:
                continue

    def set_id_zig_zag(self):

        if self.balance == 0:
            self.increment += 1
            offset = self.increment
            self.balance = self.increment
        else:
            offset = -self.increment
            self.balance = 0
        self.id = self.seed + offset
        # End of possible trajectory, end of episode
        if self.id == self._maxed_increment:
            raise EndOfEpisode

    def get_recipe_token_length(self):
        for i in range(self.X_train.shape[1]):
            if np.sum(self.X_train[0, i, :]) == 0:
                break
        self.recipe_length = i

    def maxed_increment(self):
        """ Return the index that is farthest from the seed id.
        Actually we return the index after that to count for the last action.

        :return:
        """
        max_token_index = self.recipe_length - 1
        if max_token_index - self.seed > self.seed:
            return -1  # Return the next to maximum, here is max_token_index
        else:
            return max_token_index + 1  # Return the next to maximum

    @DeprecationWarning
    def _reset_old(self, ):
        """Shuffles data and returns the first state of the shuffled data to begin training on new episode."""

        # np.random.shuffle(self.id)  # Shuffle the X and y data
        # self.episode_step = 0

        # self._episode_ended = False  # Reset terminal condition
        self.per_rec_counter += 1
        # Check if the number of passes in a recipe have been achieved
        # If yes proceed to the new recipe, if not start from a new seed in the current recipe
        if self.per_rec_counter >= self.per_rec:
            self.rec_count += 1  # Next recipe
            self.per_rec_counter = 0
            self.seed_buffer, self.used_NEs = list()
        # self.set_id()
        self.set_seed()
        self._state = self.X_train[self.per_rec_counter, self.id, :]

        return ts.restart(self._state)

    def next_recipe(self):
        self.rec_count += 1  # Go to next recipe
        self.seed = 0  # Start from the first token
        self.get_recipe_token_length()
        self.recipe_number_of_NEs_estimate()  # For posterior reward

        self.set_seed_sequential()

    def _reset(self):
        """ Go to the next token and start exploring. If no tokens left, then go to the next recipe.

        :return:
        """
        self.set_seed_sequential()

    def _step(self, action: int):
        """
        Take one step in the environment.
        If the action is correct, the environment will either return 1 or `imb_ratio` depending on the current class.
        If the action is incorrect, the environment will either return -1 or -`imb_ratio` depending on the current class.
        """
        # Deprecated way
        # if self.episode_step > self.search_space:  # or used_NEs is all
        #     # self.episode_step = True
        #     return self.reset()
        # if self._episode_ended:
        #     # The last action ended the episode. Ignore the current action and start a new episode
        #     return self.reset()

        # env_action = self.y_train[self.id[self.episode_step]]  # The label of the current state

        try:
            env_action = self.y_train[self.rec_count, self.id]  # The label of the current state
            # Not an NE Reward
            if action == env_action:
                reward = 50
            else:
                reward = -50
        except IndexError:
            reward = 0
        self.episode_step += 1

        # Deprecated
        # if self.episode_step == self.X_train.shape[0] - 1:  # If last step in data
        #     self._episode_ended = True
        # self._state = self.X_train[self.id[self.episode_step]]  # Update state with new datapoint

        try:
            self.set_id_zig_zag()
        except EndOfEpisode:
            self._episode_ended = True
            # try:
            #     self.set_seed_sequential()
            # except EndOfDataSet:

        self._state = self.X_train[self.rec_count, self.id, :]

        # ToDo Currently it does not take into account the action for the last token
        if self._episode_ended:
            return ts.termination(self._state, reward)
        else:
            return ts.transition(self._state, reward)

        # https://github.com/tensorflow/agents/blob/master/docs/tutorials/2_environments_tutorial.ipynb
        # if self._episode_ended or self._state >= 21:
        #     reward = self._state - 21 if self._state <= 21 else -21
        #     return ts.termination(np.array([self._state], dtype=np.int32), reward)
        # else:
        #     return ts.transition(
        #         np.array([self._state], dtype=np.int32), reward=0.0, discount=1.0)

    @DeprecationWarning  # This function will be at the Drivers level
    def retro_rewards(self, trajectory):
        return None

    def recipe_number_of_NEs_estimate(self):
        """Call when starting a new recipe, we hypothesize that the label 1 refers to verbs.
        The assumption is that the number of verbs should be the number of events.
        Here we take this assumption and we divide by the number of NEs to estimate how many NEs an event should have."""
        NE_sum = 0
        verb_sum = 0
        for k, v in dict(zip(*np.unique(self.y_train[self.rec_count, :], return_counts=True))):
            if k > 0:
                NE_sum += v
            if k == 1:
                verb_sum = float(v)
        self.n_NE_estimate = NE_sum/verb_sum
