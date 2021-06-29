from abc import ABC

import numpy as np
import tensorflow as tf
from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.specs.array_spec import BoundedArraySpec
from tf_agents.specs.tensor_spec import BoundedTensorSpec, TensorSpec

from tf_agents.trajectories import time_step as ts
import logging
from tf_agents.typing import types
logger = logging.getLogger('sequence_tagger_env')
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


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

    def __init__(self, X_train: np.ndarray, y_train: np.ndarray, ne_reward=50, non_ne_traj_mult=2):  # , per_rec: int
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
        # self._action_spec = BoundedArraySpec(shape=(), dtype=y_train.dtype, minimum=0, maximum=3, name="action")
        self.X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
        self.y_train = tf.convert_to_tensor(y_train, dtype=tf.int8)
        self.obs_shape_ = (1, X_train[0, 0, :].shape[0])
        # The three NEs and the option to stop the threadk, a tf.int8 produces an error is not supported
        # Also include the action of zero rule, means when all zero then make it to decide a -1
        # self._action_spec = BoundedTensorSpec(shape=(), dtype=tf.int32, minimum=0, maximum=4, name="action") # Maybe change to tf dtype
        # self._action_spec = BoundedTensorSpec(shape=(), dtype=tf.int32, minimum=-1, maximum=4, name="action")
        self._action_spec = BoundedTensorSpec(shape=(), dtype=tf.int32, minimum=0, maximum=5, name="action")
        # Observation is the embedding of the NE, which is between 0 and 1
        # self._observation_spec = ArraySpec(shape=X_train.shape[0, 0, :], dtype=X_train.dtype, name="observation")
        # self._observation_spec = BoundedArraySpec(shape=X_train[0, 0, :].shape, minimum=0, maximum=1,
        #                                           dtype=X_train.dtype, name="observation")
        self._observation_spec = BoundedTensorSpec(shape=X_train[0, 0, :].shape,
                                                   minimum=X_train.min(axis=0).min(axis=0),
                                                   maximum=X_train.max(axis=0).max(axis=0),
                                                   dtype=tf.float32,
                                                   name="observation")
        self._observation_spec_pol = BoundedTensorSpec(shape=self.obs_shape_,
                                                   minimum=X_train.min(axis=0).min(axis=0),
                                                   maximum=X_train.max(axis=0).max(axis=0),
                                                   dtype=tf.float32,
                                                   name="observation")
        # self.empty_observation = tf.convert_to_tensor(np.zeros(shape=X_train[0, 0, :].shape,
        #                                                        dtype=X_train.dtype),
        #                                               dtype=tf.float32)
        self.empty_observation = tf.convert_to_tensor(np.zeros(shape=self.obs_shape_,
                                                               dtype=X_train.dtype),
                                                      dtype=tf.float32)

        # self._reward_spec = TensorSpec(shape=(), dtype=tf.float32, name='reward')
        self._episode_ended = False


        self.ne_reward = ne_reward
        self.non_ne_traj_mult = non_ne_traj_mult
        # Count the recipes for env reset and also to establish posterior sequence reward
        self.rec_count = -1  #0  # False  #0  # It was -1 # Lead to produce error if not used
        # self.seed_buffer = list()
        self.token_trajectory = list()
        # Sample an id to start with

        self.seed = -1  # It will increment to 0 index

        self.recipe_length = False

        # self.set_seed_sequential()  # Is the self._reset()
        self.episode_step = 0  # Episode step, resets every episode


        # Inside recipe counters
        # self._maxed_increment = self.maxed_increment() # Is Updated on the function above self.set_seed_sequential()
        self.balance = 0
        self.increment = 0
        # Each episode is a recipe (1 of 50)
        # self._reset()
        # self._state = self.X_train[self.rec_count, self.seed, :]  # Is Updated on the function above self.set_seed_sequential()

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

    def set_rec_count(self, rec_count):
        self.rec_count = rec_count

    def reward_spec(self) -> types.NestedTensorSpec:
        """Defines the rewards that are returned by `step()`.

        Override this method to define an environment that uses non-standard reward
        values, for example an environment with array-valued rewards.

        Returns:
          An `ArraySpec`, or a nested dict, list or tuple of `ArraySpec`s.
        """
        return TensorSpec(shape=(), dtype=tf.float32, name='reward')

    def action_spec(self):
        """
        Definition of the discrete actionspace.
        1 for the positive/minority class, 0 for the negative/majority class.
        """
        return self._action_spec

    def observation_spec(self):
        """Definition of the continuous statespace e.g. the observations in typical RL environments."""
        # env.time_step_spec().observation == env._observation_spec
        return self._observation_spec

    def get_recipe_length(self):
        """ Get a recipe's last token before the padding which are zero. Check fast for when zero, with chance of zero
        embedding.

        :return: The token index that is zero or out of bound for the non padded recipe
        """
        try:
            assert np.argwhere(self.X_train[self.rec_count, :, 0] == 0)[0] == \
                   np.argwhere(self.X_train[self.rec_count, :, 3] == 0)[0]
        except AssertionError as e:
            logger.info("Embedding zero found\n" + str(e))
            return max(np.argwhere(self.X_train[self.rec_count, :, 0] == 0)[0][0],
                       np.argwhere(self.X_train[self.rec_count, :, 3] == 0)[0][0])
        except IndexError:
            return len(self.X_train[self.rec_count, :, 0])
        return np.argwhere(self.X_train[self.rec_count, :, 0] == 0)[0][0]

    def set_seed_sequential(self):
        """ Set the seed one after one, no matter is an NE or not

        :return:
        """
        logger.info("\nSet seed.")
        try:
            self.seed += 1
            logger.info("Seed is: " + str(self.seed))
            self.id = self.seed
            # try:
            # except ValueError as e:
            #     logger.error("Need to set a " + str(e))

            if self.seed >= self.recipe_length or not self.recipe_length:
                return self.next_recipe()
            self._state = tf.reshape(self.X_train[self.rec_count, self.id, :], self.obs_shape_)
            # self._state = self.X_train[self.rec_count, self.id, :]
            # Re-initialise inside recipe counters
            self._maxed_increment = self.maxed_increment()  # Get the farthest away token for the end of episode
            self.balance = 0
            self.increment = 0

            # if np.sum(self._state) == 0:  # Have exhausted the tokens on the current recipe, move to next recipe,
            #     # 0 would denote an empty cell
            #     self.next_recipe()
            #     # return
            # # If it is a NE
            # if self.y_train[self.rec_count, self.id, :] > 0:
            #     return ts.restart(self._state, self.ne_reward)
            # # If it's an O then skip it and move to the next token
            # else:
            #     return ts.restart(self._state, -self.ne_reward)
            return ts.restart(self._state)

        except (IndexError, tf.errors.InvalidArgumentError):
            try:
                assert self.X_train[self.rec_count, 0, :]
                # raise EndOfDataSet
            except (IndexError, tf.errors.InvalidArgumentError):  # If the error is not caused by end of data, then the index means we should go to the
                # next recipe
                raise EndOfDataSet
                # return self.next_recipe()

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

    def maxed_increment(self):
        """ Return the index that is farthest from the seed id.
        Actually we return the index after that to count for the last action.

        :return:
        """
        # max_token_index = self.recipe_length - 1
        if self.recipe_length - self.seed > self.seed:
            # According to zigzag that will be the next after the recipe max (recipe_length)
            return -self.recipe_length
        else:
            # According to zigzag that will be the next after the recipe max (0). Hence,
            # the distance after the self.seed plus 1
            return (2*self.seed) + 1  # max_token_index + 1  # Return the next to maximum

    def next_recipe(self):
        self.rec_count += 1  # Go to next recipe
        self.seed = -1  # Start from the first token
        self.recipe_length = self.get_recipe_length()
        self.recipe_number_of_NEs_estimate()  # For posterior reward
        logger.info("Next recipe...")
        return self.set_seed_sequential()

    def _reset(self):
        """ Go to the next token and start exploring. If no tokens left, then go to the next recipe.

        :return:
        """
        logger.info("Resetting...")
        return self.set_seed_sequential()

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
        # Deprecated
        # if self.episode_step == self.X_train.shape[0] - 1:  # If last step in data
        #     self._episode_ended = True
        # self._state = self.X_train[self.id[self.episode_step]]  # Update state with new datapoint
        # Do it bit inversely, calculate the new state first
        prev_id = self.id
        reward = None  # Safeguard
        self.episode_step += 1
        # Update self.id to get the new state
        try:
            self.set_id_zig_zag()
        except EndOfEpisode:
            self._episode_ended = True
            self._state = self.empty_observation
            # try:
            #     self.set_seed_sequential()
            # except EndOfDataSet:
        try:
            logger.info("Recipe: " + str(self.rec_count) + " with id: " + str(self.id))
            self._state = tf.reshape(self.X_train[self.rec_count, self.id, :], self.obs_shape_)  # self.X_train[self.rec_count, self.id, :]
            # self._state = self.X_train[self.rec_count, self.id, :]  # self.X_train[self.rec_count, self.id, :]
        # Due to being out of bounds
        except (IndexError, tf.errors.InvalidArgumentError):  # ToDo Update Exception to TF
            self._state = self.empty_observation

        # try:
        try:
            # The label of the previous state where the action was taken upon
            env_action = self.y_train[self.rec_count, prev_id].numpy()
        # Due to being out of bounds
        except (IndexError,  tf.errors.InvalidArgumentError):
            # env_action = 0
            env_action = -1
        # Not an NE Reward
        try:
            logger.info("selected: " + str(action.numpy()[0]))
            action = action.numpy()[0]
        except IndexError:
            logger.info("selected: " + str(action.numpy()))
            action = action.numpy()
        logger.info("true: " + str(env_action))
        if action.astype(np.int8) == env_action:
            reward = self.ne_reward
        elif action == 4:  # The choice action to stop the episode
            if self.y_train[self.rec_count, self.seed].numpy() > 0:  # An NE
                return ts.termination(self._state, reward=self.non_ne_traj_mult * (-self.ne_reward))
            else:
                return ts.termination(self._state, reward=self.non_ne_traj_mult * self.ne_reward)
        else:
            reward = -self.ne_reward
        # except IndexError:
        #     reward = 0

        # Set continuous negative reward if non-NE seed, for every time the agent does not choose to stop
        if self.y_train[self.rec_count, self.seed].numpy() == 0:
            if self._episode_ended:
                return ts.termination(self._state, reward=self.non_ne_traj_mult * (-self.ne_reward))
            else:
                return ts.transition(self._state, reward=self.non_ne_traj_mult * (-self.ne_reward))

        if self._episode_ended:
            return ts.termination(self._state, reward)
        else:
            # print(self._state)
            # print(reward)
            return ts.transition(self._state, reward)

        # https://github.com/tensorflow/agents/blob/master/docs/tutorials/2_environments_tutorial.ipynb
        # if self._episode_ended or self._state >= 21:
        #     reward = self._state - 21 if self._state <= 21 else -21
        #     return ts.termination(np.array([self._state], dtype=np.int32), reward)
        # else:
        #     return ts.transition(
        #         np.array([self._state], dtype=np.int32), reward=0.0, discount=1.0)

    def recipe_number_of_NEs_estimate(self):
        """Call when starting a new recipe, we hypothesize that the label 1 refers to verbs.
        The assumption is that the number of verbs should be the number of events.
        Here we take this assumption and we divide by the number of NEs to estimate how many NEs an event should have."""
        NE_sum = 0
        verb_sum = 0
        for k, v in dict(zip(*np.unique(self.y_train[self.rec_count, :].numpy(), return_counts=True))).items():
            if k > 0:
                NE_sum += v
            if k == 1:
                verb_sum = float(v)
        self.n_NE_estimate = NE_sum/verb_sum

    def time_step_spec_pol(self) -> ts.TimeStep:
        """Describes the `TimeStep` fields returned by `step()`.

        Override this method to define an environment that uses non-standard values
        for any of the items returned by `step()`. For example, an environment with
        array-valued rewards.

        Returns:
          A `TimeStep` namedtuple containing (possibly nested) `ArraySpec`s defining
          the step_type, reward, discount, and observation structure.
        """
        return ts.time_step_spec(self._observation_spec_pol, self.reward_spec())
    #
    # @DeprecationWarning
    # def set_seed_random(self):
    #     """ Set the seed random, without replacement from a recipe, no matter is an NE or not
    #
    #     :return:
    #     """
    #     pass
    # @DeprecationWarning
    # def set_id_via_normal_distr(self, sigma=3):
    #     while True:
    #         new_id = round(np.random.normal(self.seed, sigma, 1)[0])
    #         if new_id in self.seed_buffer:
    #             continue
    # @DeprecationWarning  # This function will be at the Drivers level
    # def retro_rewards(self, trajectory):
    #     return None
    # @DeprecationWarning
    # def _reset_old(self, ):
    #     """Shuffles data and returns the first state of the shuffled data to begin training on new episode."""
    #
    #     # np.random.shuffle(self.id)  # Shuffle the X and y data
    #     # self.episode_step = 0
    #
    #     # self._episode_ended = False  # Reset terminal condition
    #     self.per_rec_counter += 1
    #     # Check if the number of passes in a recipe have been achieved
    #     # If yes proceed to the new recipe, if not start from a new seed in the current recipe
    #     if self.per_rec_counter >= self.per_rec:
    #         self.rec_count += 1  # Next recipe
    #         self.per_rec_counter = 0
    #         self.seed_buffer, self.used_NEs = list()
    #     # self.set_id()
    #     self.set_seed()
    #     self._state = self.X_train[self.per_rec_counter, self.id, :]
    #
    #     return ts.restart(self._state)
    # @DeprecationWarning
    # def get_recipe_token_length(self):
    #     for i in range(self.X_train.shape[1]):
    #         if np.sum(self.X_train[0, i, :]) == 0:
    #             break
    #     self.recipe_length = i
    # @DeprecationWarning
    # def set_seed_random_NE(self):
    #     """The seed should be a random NE, we do sampling without replacement since we do not want to have
    #      the same NE seed and we want to prioritise NE samples that have not been tagged yet."""
    #
    #     NE_samples = np.nonzero(self.y_train[self.per_rec_counter, :])[0]
    #     # Get the seed IDs positions
    #     seed_buffer_ids = [np.argwhere(NE_samples == i) for i in self.seed_buffer]
    #     NE_samples = np.delete(NE_samples, seed_buffer_ids)
    #     try:
    #         # Prioritse a sample NE that has not been tagged yet
    #         used_NEs_buffer_ids = [np.argwhere(NE_samples == i) for i in self.used_NEs]
    #         NE_used_samples = np.delete(NE_samples, used_NEs_buffer_ids)
    #         self.seed = np.random.choice(NE_used_samples)
    #     except ValueError:
    #         self.seed = np.random.choice(NE_samples)
    #     self.seed_buffer.append(self.seed)
