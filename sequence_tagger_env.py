import numpy as np
from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.specs.array_spec import ArraySpec, BoundedArraySpec
from tf_agents.trajectories import time_step as ts


class ClassifierEnv(PyEnvironment):
    """
    Custom `PyEnvironment` environment for imbalanced classification.
    Based on https://www.tensorflow.org/agents/tutorials/2_environments_tutorial
    """

    def __init__(self, X_train: np.ndarray, y_train: np.ndarray, per_rec: int):  # , imb_ratio: float
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
        self._action_spec = BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=1, name="action")
        # Observation is the embedding of the NE,
        self._observation_spec = ArraySpec(shape=X_train.shape[0, 0, :], dtype=X_train.dtype, name="observation")
        self._episode_ended = False

        self.X_train = X_train
        self.y_train = y_train
        self.per_rec = per_rec
        self.per_rec_counter = 0
        self.rec_count = 0
        self.seed_buffer, self.used_NEs = list(), list()
        # self.imb_ratio = imb_ratio  # Imbalance ratio: 0 < imb_ratio < 1
        # self.id = np.arange(self.X_train.shape[0])  # List of IDs to connect X and y data
        # Sample an id to start with
        self.seed = 0
        self.set_seed()
        self.episode_step = 0  # Episode step, resets every episode
        # Each episode is a recipe (1 of 50)
        self._state = self.X_train[self.per_rec_counter, self.seed, :]

    def action_spec(self):
        """
        Definition of the discrete actionspace.
        1 for the positive/minority class, 0 for the negative/majority class.
        """
        return self._action_spec

    def observation_spec(self):
        """Definition of the continous statespace e.g. the observations in typical RL environments."""
        return self._observation_spec

    def set_seed(self):
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

    def set_id_via_normal_distr(self, sigma=3):
        while True:
            new_id = round(np.random.normal(self.seed, sigma, 1)[0])
            if new_id in self.seed_buffer:
                continue


    def _reset(self):  # ToDo
        """Shuffles data and returns the first state of the shuffled data to begin training on new episode."""

        np.random.shuffle(self.id)  # Shuffle the X and y data
        # self.episode_step = 0

        # self._episode_ended = False  # Reset terminal condition
        self.per_rec_counter += 1
        # Check if the number of passes in a recipe have been acheived
        if self.per_rec_counter >= self.per_rec:
            self.rec_count += 1  # Next recipe
            self.per_rec_counter = 0
            self.seed_buffer, self.used_NEs = list()
        self.set_id()
        self._state = self.X_train[self.per_rec_counter, self.id, :]

        return ts.restart(self._state)

    def _step(self, action: int):
        """
        Take one step in the environment.
        If the action is correct, the environment will either return 1 or `imb_ratio` depending on the current class.
        If the action is incorrect, the environment will either return -1 or -`imb_ratio` depending on the current class.
        """
        # If the
        if (self.episode_step % self.per_rec) == 0:
            # self.episode_step = True
            return self.reset()
        # if self._episode_ended:
        #     # The last action ended the episode. Ignore the current action and start a new episode
        #     return self.reset()

        env_action = self.y_train[self.id[self.episode_step]]  # The label of the current state
        self.episode_step += 1

        if action == env_action:  # Correct action
            if env_action:  # Minority
                reward = 1  # True Positive
            else:  # Majority
                reward = self.imb_ratio  # True Negative

        else:  # Incorrect action
            if env_action:  # Minority
                reward = -1  # False Negative
                self._episode_ended = True  # Stop episode when minority class is misclassified
            else:  # Majority
                reward = -self.imb_ratio  # False Positive

        if self.episode_step == self.X_train.shape[0] - 1:  # If last step in data
            self._episode_ended = True

        self._state = self.X_train[self.id[self.episode_step]]  # Update state with new datapoint

        if self._episode_ended:
            return ts.termination(self._state, reward)
        else:
            return ts.transition(self._state, reward)
