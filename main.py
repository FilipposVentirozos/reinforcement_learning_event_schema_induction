import numpy as np
from tf_agents.trajectories import trajectory
import tensorflow as tf
from tf_agents.specs import tensor_spec
from tf_agents.replay_buffers import py_uniform_replay_buffer
import embeddings, policy, sequence_tagger_env


# Hyper-Parameters
num_iterations = 20000  # @param {type:"integer"}

initial_collect_steps = 100  # @param {type:"integer"}
collect_steps_per_iteration = 1  # @param {type:"integer"}
replay_buffer_max_length = 100000  # @param {type:"integer"}

batch_size = 64  # @param {type:"integer"}
learning_rate = 1e-3  # @param {type:"number"}
log_interval = 200  # @param {type:"integer"}

num_eval_episodes = 10  # @param {type:"integer"}
eval_interval = 1000  # @param {type:"integer"}


# Get the Data
data = embeddings.ToyData()

# Environment
env = sequence_tagger_env.SequenceTaggerEnv(data.X_train, data.y_train)

# Initialise the Policy
policy_ = policy.Policy(env.action_spec())

# Replay buffer
# Replay buffer, to store variables and train accordingly
batch_size = 32
replay_buffer_capacity = 1000 * batch_size
buffer_unit = (tf.TensorSpec([1], tf.bool, 'action'),  # Binary is 0 or 1
               (tf.TensorSpec([5], tf.float32, 'lidar'),
                # ToDo set the NEs values instead, add index info as well, reward?
                tf.TensorSpec([3, 2], tf.float32, 'camera')))
py_replay_buffer = py_uniform_replay_buffer.PyUniformReplayBuffer(
    capacity=replay_buffer_capacity,
    data_spec=tensor_spec.to_nest_array_spec(buffer_unit))

# Data Collection
# @test {"skip": true}
def collect_step(environment, policy, buffer):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)

    # Add trajectory to the replay buffer
    buffer.add_batch(traj)

def collect_data(env, policy, buffer, steps):
    for _ in range(steps):
        collect_step(env, policy, buffer)

collect_data(sequence_tagger_env.SequenceTaggerEnv, random_policy, replay_buffer, initial_collect_steps)




