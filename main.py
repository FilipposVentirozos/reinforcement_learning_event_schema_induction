import tensorflow as tf
from tf_agents.replay_buffers import py_uniform_replay_buffer
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import trajectory

# Local
from archive import embeddings
import policy
import sequence_tagger_env
from dataset import DataSet
from driver import IntervalDriver

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

# Domain Specific Hyper-Parameters

# Can be an integer or a lambda function. If lambda function then is the number of verbs of a recipe +/- the function
# agents_per_recipe = lambda x: x + 1
agents_per_recipe = 5
# The interval is when the trajectory ends and the rewards are calculated
# Is an integer denoting how many recipes, or could be a signal from the data
interval = 25



# Get the Data
dataset_path = "/home/chroner/PhD_remote/RL_Event_Schema_Induction/data/processed/recipes_0_0_proc"
dat = DataSet()
dat.fetch_from_path(dataset_path)

# Environment
env = sequence_tagger_env.SequenceTaggerEnv(dat.X, dat.label)

# Initialise the Policy
policy_ = policy.Policy(env.action_spec(), policy="random")

# Create PPO agent


# Replay buffer
# Replay buffer, to store variables and train accordingly
batch_size = 32
replay_buffer_capacity = 10_000 * batch_size
# Use agent's traj unit for the buffer
buffer_unit = (tf.TensorSpec([1], tf.bool, 'action'),  # Binary is 0 or 1
               (tf.TensorSpec([5], tf.float32, 'lidar'),
                # ToDo set the NEs values instead, add index info as well, reward?
                tf.TensorSpec([3, 2], tf.float32, 'camera')))
py_replay_buffer = py_uniform_replay_buffer.PyUniformReplayBuffer(
    capacity=replay_buffer_capacity,
    data_spec=tensor_spec.to_nest_array_spec(buffer_unit))

driver = IntervalDriver(env=env, policy=policy_, buffer_observer=py_replay_buffer)

for _ in range(num_iterations):
    driver.run()
    # loss_info = agent.train(replay_buffer.gather_all())
    # replay_buffer.clear()
    # regret_values.append(regret_metric.result())


# # Data Collection
# # @test {"skip": true}
# def collect_step(environment, policy, buffer):
#     time_step = environment.current_time_step()
#     action_step = policy.action(time_step)
#     next_time_step = environment.step(action_step.action)
#     traj = trajectory.from_transition(time_step, action_step, next_time_step)
#
#     # Add trajectory to the replay buffer
#     buffer.add_batch(traj)
#
# def collect_data(env, policy, buffer, steps):
#     for _ in range(steps):
#         collect_step(env, policy, buffer)

# collect_data(sequence_tagger_env.SequenceTaggerEnv, random_policy, replay_buffer, initial_collect_steps)




